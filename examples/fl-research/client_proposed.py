import os
import time
import random
import psutil
from dataclasses import dataclass
from typing import Dict, List, Union

import flwr as fl
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import datasets, transforms, models

try:
    from flwr.common import Context
except Exception:
    Context = object


# ============================================================
# Reproducibility
# ============================================================

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Model
# ============================================================

def build_vgg16(num_classes: int = 10) -> nn.Module:
    weights = models.VGG16_Weights.IMAGENET1K_V1
    model = models.vgg16(weights=weights)

    model.classifier[6] = nn.Linear(
        in_features=model.classifier[6].in_features,
        out_features=num_classes,
    )

    return model


def get_params(model: nn.Module) -> List[np.ndarray]:
    return [p.detach().cpu().numpy() for p in model.parameters()]


def set_params(model: nn.Module, params: List[np.ndarray]) -> None:
    with torch.no_grad():
        for p, np_p in zip(model.parameters(), params):
            p.copy_(
                torch.tensor(
                    np_p,
                    dtype=p.dtype,
                    device=p.device,
                )
            )


def model_size_bytes(model: nn.Module) -> int:
    return sum(
        p.numel() * p.element_size()
        for p in model.parameters()
    )


# ============================================================
# Experiment configuration
# ============================================================

@dataclass
class DataConfig:
    num_clients: int = int(
        os.environ.get("NUM_CLIENTS", 20)
    )

    batch_size: int = int(
        os.environ.get("BATCH_SIZE", 16)
    )

    data_dir: str = os.environ.get(
        "CINIC10_DIR",
        os.path.expanduser("~/.data/CINIC-10"),
    )

    base_seed: int = int(
        os.environ.get("BASE_SEED", 1234)
    )

    lr: float = float(
        os.environ.get("LR", 0.001)
    )

    local_epochs: int = int(
        os.environ.get("LOCAL_EPOCHS", 1)
    )

    num_workers: int = int(
        os.environ.get("NUM_WORKERS", 2)
    )

    # Batch size used only when computing the local data profile.
    profile_batch_size: int = int(
        os.environ.get("PROFILE_BATCH_SIZE", 64)
    )
    partition_mode: str = os.environ.get("PARTITION_MODE", "iid").lower()
    dirichlet_alpha: float = float(os.environ.get("DIRICHLET_ALPHA", "0.5"))
    # --------------------------------------------------------
# System eligibility requirements
#
# These are configurable experimental parameters.
# The current defaults are only for implementation
# validation and are NOT the final research thresholds.
# --------------------------------------------------------

    min_cpu_count: int = int(
        os.environ.get(
           "MIN_CPU_COUNT",
            "1",
        )
    )

    min_available_ram_mb: float = float(
        os.environ.get(
            "MIN_AVAILABLE_RAM_MB",
            "1024",
        )
    )

    require_gpu: bool = (
        os.environ.get(
            "REQUIRE_GPU",
            "false",
        ).lower()
        == "true"
    )

    min_free_gpu_memory_mb: float = float(
        os.environ.get(
            "MIN_FREE_GPU_MEMORY_MB",
            "0",
        )
    )
        # --------------------------------------------------------
        # Network capability configuration
        #
        # "simulated":
        #     Use deterministic heterogeneous network values.
        #     This is appropriate for the current Ray simulation
        #     because virtual clients run on the same EC2 host.
    # --------------------------------------------------------
        # "measured":
    #     Reserved for future real-client experiments where
    #     network capability will be measured between each
    #     physical client and the FL server.
    # --------------------------------------------------------

    network_mode: str = os.environ.get(
        "NETWORK_MODE",
        "simulated",
    ).lower()

    simulated_upload_min_mbps: float = float(
        os.environ.get(
            "SIM_UPLOAD_MIN_MBPS",
            "10.0",
        )
    )

    simulated_upload_max_mbps: float = float(
        os.environ.get(
            "SIM_UPLOAD_MAX_MBPS",
            "100.0",
        )
    )

    simulated_download_min_mbps: float = float(
        os.environ.get(
            "SIM_DOWNLOAD_MIN_MBPS",
            "50.0",
        )
    )

    simulated_download_max_mbps: float = float(
        os.environ.get(
            "SIM_DOWNLOAD_MAX_MBPS",
            "300.0",
        )
    )

    simulated_latency_min_ms: float = float(
        os.environ.get(
            "SIM_LATENCY_MIN_MS",
            "10.0",
        )
    )

    simulated_latency_max_ms: float = float(
        os.environ.get(
            "SIM_LATENCY_MAX_MS",
            "150.0",
        )
    )


# ============================================================
# CINIC-10 datasets
# ============================================================

def load_cinic10(data_dir: str):
    """
    Create two views of the local training data:

    1. federated_train_ds:
       Used for actual local model training.
       Includes augmentation and ImageNet normalization.

    2. federated_profile_ds:
       Used only for statistical data profiling.
       No random augmentation and no ImageNet normalization.

    Both datasets have the same sample ordering, which allows
    the same client partition indices to be used for both.
    """

    # --------------------------------------------------------
    # Training transform
    # --------------------------------------------------------

    train_tfm = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )

    # --------------------------------------------------------
    # Test transform
    # --------------------------------------------------------

    test_tfm = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )

    # --------------------------------------------------------
    # Profiling transform
    #
    # IMPORTANT:
    # We intentionally do NOT use RandomHorizontalFlip here.
    # We also do NOT apply ImageNet normalization.
    #
    # Therefore RGB statistics are computed in [0, 1].
    # --------------------------------------------------------

    profile_tfm = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    # --------------------------------------------------------
    # Datasets used for training
    # --------------------------------------------------------

    train_ds = datasets.ImageFolder(
        root=os.path.join(data_dir, "train"),
        transform=train_tfm,
    )

    valid_ds = datasets.ImageFolder(
        root=os.path.join(data_dir, "valid"),
        transform=train_tfm,
    )

    federated_train_ds = ConcatDataset(
        [train_ds, valid_ds]
    )

    # --------------------------------------------------------
    # Same samples, but deterministic profiling transform
    # --------------------------------------------------------

    profile_train_ds = datasets.ImageFolder(
        root=os.path.join(data_dir, "train"),
        transform=profile_tfm,
    )

    profile_valid_ds = datasets.ImageFolder(
        root=os.path.join(data_dir, "valid"),
        transform=profile_tfm,
    )

    federated_profile_ds = ConcatDataset(
        [profile_train_ds, profile_valid_ds]
    )

    # --------------------------------------------------------
    # Global test dataset
    # --------------------------------------------------------

    test_ds = datasets.ImageFolder(
        root=os.path.join(data_dir, "test"),
        transform=test_tfm,
    )

    return (
        federated_train_ds,
        federated_profile_ds,
        test_ds,
    )


# ============================================================
# Client partitioning
# ============================================================

def partition_indices(
    num_samples: int,
    num_clients: int,
    seed: int,
) -> List[np.ndarray]:

    rng = np.random.default_rng(seed)

    idx = np.arange(num_samples)

    rng.shuffle(idx)

    return np.array_split(
        idx,
        num_clients,
    )

def dirichlet_partition_indices(
    dataset,
    num_clients: int,
    alpha: float,
    seed: int,
):
    """
    Partition dataset indices across clients using a
    class-wise Dirichlet distribution.

    Smaller alpha -> stronger Non-IID.
    Larger alpha  -> closer to IID.
    """

    if alpha <= 0:
        raise ValueError("DIRICHLET_ALPHA must be > 0.")

    rng = np.random.default_rng(seed)

    # Extract labels from the complete dataset
    labels = np.array(
        [dataset[i][1] for i in range(len(dataset))],
        dtype=np.int64,
    )

    num_classes = int(labels.max()) + 1

    client_indices = [
        [] for _ in range(num_clients)
    ]

    for class_id in range(num_classes):

        # Find all samples belonging to this class
        class_indices = np.where(
            labels == class_id
        )[0]

        rng.shuffle(class_indices)

        # Draw client proportions for this class
        proportions = rng.dirichlet(
            np.full(num_clients, alpha)
        )

        # Convert proportions to split positions
        split_points = (
            np.cumsum(proportions)[:-1]
            * len(class_indices)
        ).astype(int)

        class_splits = np.split(
            class_indices,
            split_points,
        )

        # Assign class samples to clients
        for cid, split in enumerate(class_splits):
            client_indices[cid].extend(
                split.tolist()
            )

    # Shuffle each client's final local dataset
    for cid in range(num_clients):
        rng.shuffle(client_indices[cid])

        client_indices[cid] = np.array(
            client_indices[cid],
            dtype=np.int64,
        )

    return client_indices
# ============================================================
# Proposed client
# ============================================================

class ProposedClient(fl.client.NumPyClient):

    def __init__(
        self,
        cid: int,
        cfg: DataConfig,
    ):

        self.cid = cid
        self.cfg = cfg

        seed_everything(
            cfg.base_seed + cid
        )

        # ----------------------------------------------------
        # Build local VGG16 model
        # ----------------------------------------------------

        self.model = build_vgg16(
            num_classes=10
        ).to(DEVICE)

        self.criterion = nn.CrossEntropyLoss()

        self.opt = optim.SGD(
            self.model.parameters(),
            lr=cfg.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )

        # ----------------------------------------------------
        # Load datasets
        # ----------------------------------------------------

        (
            train_ds,
            profile_ds,
            test_ds,
        ) = load_cinic10(cfg.data_dir)

        # ----------------------------------------------------
        # Create the client partition once.
        #
        # The SAME indices are used for:
        #
        #   training dataset
        #   profiling dataset
        #
        # Therefore the profile describes exactly the same
        # samples owned by this client.
        # ----------------------------------------------------

        if cfg.partition_mode == "iid":

            all_parts = partition_indices(
                len(train_ds),
                cfg.num_clients,
                cfg.base_seed,
            )

        elif cfg.partition_mode == "dirichlet":

            all_parts = dirichlet_partition_indices(
                profile_ds,
                cfg.num_clients,
                cfg.dirichlet_alpha,
                cfg.base_seed,
            )

        else:
            raise ValueError(
                f"Unknown PARTITION_MODE: {cfg.partition_mode}. "
                "Use 'iid' or 'dirichlet'."
            )

        client_train_idx = all_parts[self.cid]

        print(
            f"[CLIENT {self.cid}] "
            f"Partition mode: {cfg.partition_mode}"
        )

        if cfg.partition_mode == "dirichlet":
            print(
                f"[CLIENT {self.cid}] "
                f"Dirichlet alpha: {cfg.dirichlet_alpha}"
            )

        # ----------------------------------------------------
        # Training loader
        # ----------------------------------------------------

        self.train_loader = DataLoader(
            Subset(
                train_ds,
                client_train_idx,
            ),
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        # ----------------------------------------------------
        # Profiling loader
        #
        # shuffle=False because profiling should be
        # deterministic.
        # ----------------------------------------------------

        self.profile_loader = DataLoader(
            Subset(
                profile_ds,
                client_train_idx,
            ),
            batch_size=cfg.profile_batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=False,
        )

        # ----------------------------------------------------
        # Test loader
        # ----------------------------------------------------

        self.test_loader = DataLoader(
            test_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        self.model_bytes = model_size_bytes(
            self.model
        )

        # ----------------------------------------------------
        # Compute the profile once.
        #
        # CINIC-10 partitions are static in this experiment,
        # so there is no reason to recompute these statistics
        # every training round.
        # ----------------------------------------------------

        self.data_profile = self.compute_data_profile()

        self.print_data_profile()


    # ========================================================
    # Local Statistical Data Profiling
    # ========================================================

    def compute_data_profile(self) -> Dict[str, object]:
        """
        Compute a compact statistical description of the
        client's local CINIC-10 partition.

        V1 profile contains:

        1. Number of local samples
        2. Class distribution
        3. RGB channel means
        4. RGB channel standard deviations

        Raw images never leave the client.
        """

        num_classes = 10

        class_counts = torch.zeros(
            num_classes,
            dtype=torch.long,
        )

        # Sum of all pixel values for R, G, B.
        channel_sum = torch.zeros(
            3,
            dtype=torch.float64,
        )

        # Sum of squared pixel values for R, G, B.
        channel_squared_sum = torch.zeros(
            3,
            dtype=torch.float64,
        )

        total_pixels_per_channel = 0
        total_samples = 0

        # No gradients are required for profiling.
        with torch.no_grad():

            for images, labels in self.profile_loader:

                images = images.to(
                    dtype=torch.float64
                )

                batch_size = images.shape[0]

                total_samples += batch_size

                # --------------------------------------------
                # Class counts
                # --------------------------------------------

                class_counts += torch.bincount(
                    labels,
                    minlength=num_classes,
                )

                # --------------------------------------------
                # RGB statistics
                #
                # images shape:
                #
                # [B, C, H, W]
                #
                # Sum over:
                # batch, height, width
                #
                # Result:
                # [R_sum, G_sum, B_sum]
                # --------------------------------------------

                channel_sum += images.sum(
                    dim=(0, 2, 3)
                )

                channel_squared_sum += (
                    images ** 2
                ).sum(
                    dim=(0, 2, 3)
                )

                total_pixels_per_channel += (
                    batch_size
                    * images.shape[2]
                    * images.shape[3]
                )

        # ----------------------------------------------------
        # Safety check
        # ----------------------------------------------------

        if total_samples == 0:
            raise RuntimeError(
                f"Client {self.cid} has no local samples."
            )

        # ----------------------------------------------------
        # Class distribution
        #
        # Example:
        #
        # [0.10, 0.08, ..., 0.12]
        #
        # Sum should be approximately 1.
        # ----------------------------------------------------

        class_distribution = (
            class_counts.double()
            / total_samples
        )

        # ----------------------------------------------------
        # RGB mean
        #
        # E[X]
        # ----------------------------------------------------

        channel_mean = (
            channel_sum
            / total_pixels_per_channel
        )

        # ----------------------------------------------------
        # RGB variance
        #
        # Var(X) = E[X^2] - E[X]^2
        # ----------------------------------------------------

        channel_variance = (
            channel_squared_sum
            / total_pixels_per_channel
        ) - channel_mean ** 2

        # Protect against tiny negative values caused by
        # floating-point precision.
        channel_variance = torch.clamp(
            channel_variance,
            min=0.0,
        )

        channel_std = torch.sqrt(
            channel_variance
        )

        # ----------------------------------------------------
        # Build local profile
        # ----------------------------------------------------

        profile = {
            "num_samples": int(total_samples),

            "class_counts":
                class_counts.tolist(),

            "class_distribution":
                class_distribution.tolist(),

            "channel_mean":
                channel_mean.tolist(),

            "channel_std":
                channel_std.tolist(),
        }

        return profile


    # ========================================================
    # Print profile for verification
    # ========================================================

    def print_data_profile(self) -> None:

        profile = self.data_profile

        print(
            f"\n"
            f"========================================\n"
            f"[CLIENT {self.cid}] LOCAL DATA PROFILE\n"
            f"========================================"
        )

        print(
            f"Number of samples: "
            f"{profile['num_samples']}"
        )

        print(
            "Class counts: "
            f"{profile['class_counts']}"
        )

        print(
            "Class distribution: "
            + str(
                [
                    round(x, 4)
                    for x in profile[
                        "class_distribution"
                    ]
                ]
            )
        )

        print(
            "RGB mean: "
            + str(
                [
                    round(x, 6)
                    for x in profile[
                        "channel_mean"
                    ]
                ]
            )
        )

        print(
            "RGB std: "
            + str(
                [
                    round(x, 6)
                    for x in profile[
                        "channel_std"
                    ]
                ]
            )
        )

        print(
            "Distribution sum: "
            f"{sum(profile['class_distribution']):.6f}"
        )

        print(
            "========================================\n"
        )

        # ========================================================
    # Network Capability Profiling
    # ========================================================

    def compute_network_profile(self) -> Dict[str, float]:
        """
        Return the client's network capability profile.

        Current implementation supports:

        1. simulated:
           Deterministic heterogeneous network characteristics
           for Ray-based experiments where virtual clients run
           on the same physical EC2 host.

        2. measured:
           Reserved for future real-client experiments.

        The simulated profile is deterministic with respect to
        BASE_SEED and logical client ID. Therefore, the same
        client receives the same network characteristics when
        the experiment is repeated with the same seed.
        """

        network_mode = self.cfg.network_mode

        # ----------------------------------------------------
        # Simulation mode
        # ----------------------------------------------------

        if network_mode == "simulated":

            network_seed = (
                self.cfg.base_seed
                + 100000
                + self.cid
            )

            rng = np.random.default_rng(
                network_seed
            )

            upload_bandwidth_mbps = float(
                rng.uniform(
                    self.cfg.simulated_upload_min_mbps,
                    self.cfg.simulated_upload_max_mbps,
                )
            )

            download_bandwidth_mbps = float(
                rng.uniform(
                    self.cfg.simulated_download_min_mbps,
                    self.cfg.simulated_download_max_mbps,
                )
            )

            latency_ms = float(
                rng.uniform(
                    self.cfg.simulated_latency_min_ms,
                    self.cfg.simulated_latency_max_ms,
                )
            )

            return {
                "upload_bandwidth_mbps":
                    upload_bandwidth_mbps,

                "download_bandwidth_mbps":
                    download_bandwidth_mbps,

                "latency_ms":
                    latency_ms,
            }

        # ----------------------------------------------------
        # Real-client measurement mode
        # ----------------------------------------------------

        if network_mode == "measured":

            raise NotImplementedError(
                "NETWORK_MODE='measured' is reserved for "
                "future real-client experiments. "
                "Client-to-server bandwidth and latency "
                "measurement has not been implemented yet."
            )

        # ----------------------------------------------------
        # Invalid mode
        # ----------------------------------------------------

        raise ValueError(
            f"Unknown NETWORK_MODE: {network_mode}. "
            "Use 'simulated' or 'measured'."
        )


        # ========================================================
    # Current System Capability Profiling
    # ========================================================

    def compute_system_profile(self) -> Dict[str, object]:
        """
        Measure the client's current system capability.

        Unlike the local data profile, the system profile is
        dynamic and should be measured again when needed
        because CPU load, RAM availability, and GPU state
        can change over time.

        V1 system profile contains:

        1. Logical CPU count
        2. Current CPU utilization
        3. Total RAM
        4. Available RAM
        5. RAM utilization
        6. GPU availability
        7. GPU count
        8. GPU total memory
        9. GPU currently allocated memory
        10. GPU currently reserved memory
        """

        # ----------------------------------------------------
        # CPU information
        # ----------------------------------------------------

        cpu_count = psutil.cpu_count(
            logical=True
        )

        cpu_utilization = psutil.cpu_percent(
            interval=0.2
        )

        # ----------------------------------------------------
        # RAM information
        # ----------------------------------------------------

        memory = psutil.virtual_memory()

        total_ram_mb = (
            memory.total
            / (1024 ** 2)
        )

        available_ram_mb = (
            memory.available
            / (1024 ** 2)
        )

        ram_utilization = float(
            memory.percent
        )

        # ----------------------------------------------------
        # GPU information
        # ----------------------------------------------------

        gpu_available = torch.cuda.is_available()

        gpu_count = (
            torch.cuda.device_count()
            if gpu_available
            else 0
        )

        gpu_total_memory_mb = 0.0
        gpu_allocated_memory_mb = 0.0
        gpu_reserved_memory_mb = 0.0
        gpu_free_memory_mb = 0.0

        if gpu_available:

            device_index = torch.cuda.current_device()

            gpu_properties = (
                torch.cuda.get_device_properties(
                    device_index
                )
            )

            gpu_total_memory_mb = (
                gpu_properties.total_memory
                / (1024 ** 2)
            )
            free_bytes, total_bytes = torch.cuda.mem_get_info(
                device_index
            )

            gpu_free_memory_mb = (
                free_bytes
              / (1024 ** 2)
            )

            gpu_allocated_memory_mb = (
                torch.cuda.memory_allocated(
                    device_index
                )
                / (1024 ** 2)
            )

            gpu_reserved_memory_mb = (
                torch.cuda.memory_reserved(
                    device_index
                )
                / (1024 ** 2)
            )

        # ----------------------------------------------------
        # Network information
        # ----------------------------------------------------

        network_profile = (
            self.compute_network_profile()
        )

        # ----------------------------------------------------
        # Build system profile
        # ----------------------------------------------------

        system_profile = {

            "cpu_count":
                int(cpu_count or 0),

            "cpu_utilization_percent":
                float(cpu_utilization),

            "total_ram_mb":
                float(total_ram_mb),

            "available_ram_mb":
                float(available_ram_mb),

            "ram_utilization_percent":
                float(ram_utilization),

            "gpu_available":
                bool(gpu_available),

            "gpu_count":
                int(gpu_count),

            "gpu_total_memory_mb":
                float(gpu_total_memory_mb),

            "gpu_free_memory_mb":
                float(gpu_free_memory_mb),

            "gpu_allocated_memory_mb":
                float(gpu_allocated_memory_mb),

            "gpu_reserved_memory_mb":
                float(gpu_reserved_memory_mb),


             "upload_bandwidth_mbps":
                float(
                    network_profile[
                        "upload_bandwidth_mbps"
                    ]
                ),

            "download_bandwidth_mbps":
                float(
                    network_profile[
                        "download_bandwidth_mbps"
                    ]
                ),

            "latency_ms":
                float(
                    network_profile[
                        "latency_ms"
                    ]
                ),
        }

        return system_profile

        # ========================================================
    # System Eligibility Check
    # ========================================================

    def check_eligibility(
        self,
        system_profile: Dict[str, object],
    ) -> Dict[str, object]:
        """
        Check whether the client currently satisfies the
        minimum system requirements for participation.

        IMPORTANT:
        The thresholds are configurable experimental
        parameters. The current default values are used
        only to validate the implementation.

        A failed client is deferred for the current round,
        not permanently removed from the FL system.
        """

        reasons = []

        # ----------------------------------------------------
        # CPU requirement
        # ----------------------------------------------------

        cpu_ok = (
            system_profile["cpu_count"]
            >= self.cfg.min_cpu_count
        )

        if not cpu_ok:
            reasons.append(
                "insufficient_cpu_count"
            )

        # ----------------------------------------------------
        # RAM requirement
        # ----------------------------------------------------

        ram_ok = (
            system_profile["available_ram_mb"]
            >= self.cfg.min_available_ram_mb
        )

        if not ram_ok:
            reasons.append(
                "insufficient_available_ram"
            )

        # ----------------------------------------------------
        # GPU requirement
        # ----------------------------------------------------

        gpu_available_ok = True
        gpu_memory_ok = True

        if self.cfg.require_gpu:

            gpu_available_ok = bool(
                system_profile["gpu_available"]
            )

            if not gpu_available_ok:

                reasons.append(
                    "gpu_required_but_unavailable"
                )

                gpu_memory_ok = False

            else:

                gpu_memory_ok = (
                    system_profile[
                        "gpu_free_memory_mb"
                    ]
                    >= self.cfg.min_free_gpu_memory_mb
                )

                if not gpu_memory_ok:
                    reasons.append(
                        "insufficient_free_gpu_memory"
                    )

        # ----------------------------------------------------
        # Final eligibility decision
        # ----------------------------------------------------

        eligible = (
            cpu_ok
            and ram_ok
            and gpu_available_ok
            and gpu_memory_ok
        )

        return {
            "eligible":
                bool(eligible),

            "cpu_ok":
                bool(cpu_ok),

            "ram_ok":
                bool(ram_ok),

            "gpu_available_ok":
                bool(gpu_available_ok),

            "gpu_memory_ok":
                bool(gpu_memory_ok),

            "reasons":
                reasons,
        }

        # ========================================================
    # Print eligibility result for verification
    # ========================================================

    def print_eligibility(
        self,
        result: Dict[str, object],
    ) -> None:

        print(
            f"\n"
            f"========================================\n"
            f"[CLIENT {self.cid}] ELIGIBILITY CHECK\n"
            f"========================================"
        )

        print(
            f"Eligible: "
            f"{result['eligible']}"
        )

        print(
            f"CPU requirement passed: "
            f"{result['cpu_ok']}"
        )

        print(
            f"RAM requirement passed: "
            f"{result['ram_ok']}"
        )

        print(
            f"GPU availability passed: "
            f"{result['gpu_available_ok']}"
        )

        print(
            f"GPU memory requirement passed: "
            f"{result['gpu_memory_ok']}"
        )

        if result["reasons"]:

            print(
                "Reason(s): "
                + ", ".join(
                    result["reasons"]
                )
            )

        else:

            print(
                "Reason(s): None"
            )

        print(
            "========================================\n"
        )
    


    # ========================================================
    # Print system profile for verification
    # ========================================================

    def print_system_profile(
        self,
        profile: Dict[str, object],
    ) -> None:

        print(
            f"\n"
            f"========================================\n"
            f"[CLIENT {self.cid}] SYSTEM CAPABILITY PROFILE\n"
            f"========================================"
        )

        print(
            f"CPU count: "
            f"{profile['cpu_count']}"
        )

        print(
            f"CPU utilization: "
            f"{profile['cpu_utilization_percent']:.2f}%"
        )

        print(
            f"Total RAM: "
            f"{profile['total_ram_mb']:.2f} MB"
        )

        print(
            f"Available RAM: "
            f"{profile['available_ram_mb']:.2f} MB"
        )

        print(
            f"RAM utilization: "
            f"{profile['ram_utilization_percent']:.2f}%"
        )

        print(
            f"GPU available: "
            f"{profile['gpu_available']}"
        )

        print(
            f"GPU count: "
            f"{profile['gpu_count']}"
        )

        print(
            f"GPU total memory: "
            f"{profile['gpu_total_memory_mb']:.2f} MB"
        )
        print(
            f"GPU free memory: "
            f"{profile['gpu_free_memory_mb']:.2f} MB"
        )

        print(
            f"GPU allocated memory: "
            f"{profile['gpu_allocated_memory_mb']:.2f} MB"
        )

        print(
            f"GPU reserved memory: "
            f"{profile['gpu_reserved_memory_mb']:.2f} MB"
        )

        print(
            f"Network mode: "
            f"{self.cfg.network_mode}"
        )

        print(
            f"Upload bandwidth: "
            f"{profile['upload_bandwidth_mbps']:.2f} Mbps"
        )

        print(
            f"Download bandwidth: "
            f"{profile['download_bandwidth_mbps']:.2f} Mbps"
        )

        print(
            f"Latency: "
            f"{profile['latency_ms']:.2f} ms"
        )

        print(
            "========================================\n"
        )
            # ========================================================
    # Pre-selection client properties
    # ========================================================

    def get_properties(self, config):
        """
        Return lightweight client information required by
        the server before client selection.

        The local data profile is cached because the local
        dataset is static.

        The system profile is recomputed because system
        conditions can change between rounds.
        """

        # ----------------------------------------------------
        # Static local data profile
        # ----------------------------------------------------

        data_profile = self.data_profile

        # ----------------------------------------------------
        # Dynamic system profile
        # ----------------------------------------------------

        system_profile = (
            self.compute_system_profile()
        )

        # ----------------------------------------------------
        # Current eligibility
        # ----------------------------------------------------

        eligibility = (
            self.check_eligibility(
                system_profile
            )
        )

        # ----------------------------------------------------
        # Flatten properties into scalar values
        # ----------------------------------------------------

        properties = {
            "cid":
                int(self.cid),

            "eligible":
                bool(
                    eligibility["eligible"]
                ),

            "num_samples":
                int(
                    data_profile["num_samples"]
                ),

            "cpu_count":
                int(
                    system_profile["cpu_count"]
                ),

            "cpu_utilization_percent":
                float(
                    system_profile[
                        "cpu_utilization_percent"
                    ]
                ),

            "total_ram_mb":
                float(
                    system_profile[
                        "total_ram_mb"
                    ]
                ),

            "available_ram_mb":
                float(
                    system_profile[
                        "available_ram_mb"
                    ]
                ),

            "ram_utilization_percent":
                float(
                    system_profile[
                        "ram_utilization_percent"
                    ]
                ),

            "gpu_available":
                bool(
                    system_profile[
                        "gpu_available"
                    ]
                ),

            "gpu_count":
                int(
                    system_profile[
                        "gpu_count"
                    ]
                ),

            "gpu_total_memory_mb":
                float(
                    system_profile[
                        "gpu_total_memory_mb"
                    ]
                ),

            "gpu_free_memory_mb":
                float(
                    system_profile[
                        "gpu_free_memory_mb"
                    ]
                ),

            "upload_bandwidth_mbps":
                float(
                    system_profile[
                        "upload_bandwidth_mbps"
                    ]
                ),

            "download_bandwidth_mbps":
                float(
                    system_profile[
                        "download_bandwidth_mbps"
                    ]
                ),

            "latency_ms":
                float(
                    system_profile[
                        "latency_ms"
                    ]
                ),
        }

        # ----------------------------------------------------
        # Class distribution
        # ----------------------------------------------------

        for class_id, ratio in enumerate(
            data_profile["class_distribution"]
        ):

            properties[
                f"class_{class_id}_ratio"
            ] = float(ratio)

        # ----------------------------------------------------
        # RGB mean
        # ----------------------------------------------------

        channel_names = [
            "r",
            "g",
            "b",
        ]

        for channel_name, value in zip(
            channel_names,
            data_profile["channel_mean"],
        ):

            properties[
                f"rgb_mean_{channel_name}"
            ] = float(value)

        # ----------------------------------------------------
        # RGB standard deviation
        # ----------------------------------------------------

        for channel_name, value in zip(
            channel_names,
            data_profile["channel_std"],
        ):

            properties[
                f"rgb_std_{channel_name}"
            ] = float(value)

        return properties
        


    # ========================================================
    # Flower parameters
    # ========================================================

    def get_parameters(self, config):
        return get_params(self.model)


    # ========================================================
    # Local training
    # ========================================================

    def fit(
        self,
        parameters,
        config,
    ):

        set_params(
            self.model,
            parameters,
        )

        self.model.train()

        local_epochs = int(
            config.get(
                "local_epochs",
                self.cfg.local_epochs,
            )
        )

        start = time.time()

        total_loss = 0.0
        num_examples = 0
        num_batches = 0

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        for _ in range(local_epochs):

            for x, y in self.train_loader:

                x = x.to(DEVICE)
                y = y.to(DEVICE)

                self.opt.zero_grad()

                logits = self.model(x)

                loss = self.criterion(
                    logits,
                    y,
                )

                loss.backward()

                self.opt.step()

                total_loss += float(
                    loss.detach().cpu()
                )

                num_examples += x.shape[0]

                num_batches += 1

        fit_duration = (
            time.time() - start
        )

        avg_loss = (
            total_loss
            / max(1, num_batches)
        )

        samples_per_second = (
            num_examples
            / max(fit_duration, 1e-9)
        )

        gpu_memory_mb = 0.0

        if torch.cuda.is_available():

            gpu_memory_mb = (
                torch.cuda.max_memory_allocated()
                / (1024 ** 2)
            )

        metrics: Dict[str, float] = {

            "cid":
                int(self.cid),

            "train_loss":
                float(avg_loss),

            "fit_duration":
                float(fit_duration),

            "num_examples":
                int(num_examples),

            "samples_per_second":
                float(samples_per_second),

            "model_bytes":
                int(self.model_bytes),

            "upload_bytes":
                int(self.model_bytes),

            "download_bytes":
                int(self.model_bytes),

            "gpu_memory_mb":
                float(gpu_memory_mb),
        }

        print(
            f"[CLIENT {self.cid}] fit: "
            f"loss={avg_loss:.4f}, "
            f"time={fit_duration:.2f}s, "
            f"examples={num_examples}, "
            f"samples/s={samples_per_second:.2f}"
        )

        return (
            get_params(self.model),
            num_examples,
            metrics,
        )


    # ========================================================
    # Evaluation
    # ========================================================

    def evaluate(
        self,
        parameters,
        config,
    ):

        set_params(
            self.model,
            parameters,
        )

        self.model.eval()

        loss_sum = 0.0
        correct = 0
        total = 0

        start = time.time()

        with torch.no_grad():

            for x, y in self.test_loader:

                x = x.to(DEVICE)
                y = y.to(DEVICE)

                logits = self.model(x)

                loss = self.criterion(
                    logits,
                    y,
                )

                loss_sum += float(
                    loss.detach().cpu()
                )

                pred = logits.argmax(
                    dim=1
                )

                correct += int(
                    (pred == y).sum().cpu()
                )

                total += x.shape[0]

        eval_duration = (
            time.time() - start
        )

        avg_loss = (
            loss_sum
            / max(1, len(self.test_loader))
        )

        acc = (
            correct
            / max(1, total)
        )

        return (
            float(avg_loss),
            int(total),
            {
                "accuracy":
                    float(acc),

                "eval_loss":
                    float(avg_loss),

                "eval_duration":
                    float(eval_duration),
            },
        )


# ============================================================
# Global configuration
# ============================================================

CFG = DataConfig()


# ============================================================
# Flower client factory
# ============================================================

def client_fn(
    arg: Union[str, "Context"]
) -> fl.client.Client:

    if isinstance(arg, str):

        cid = int(arg)

    else:

        if (
            hasattr(arg, "node_config")
            and "partition-id"
            in arg.node_config
        ):

            cid = int(
                arg.node_config[
                    "partition-id"
                ]
            )

        else:

            cid = int(
                getattr(
                    arg,
                    "node_id",
                    0,
                )
            )

    return ProposedClient(
        cid=cid,
        cfg=CFG,
    ).to_client()