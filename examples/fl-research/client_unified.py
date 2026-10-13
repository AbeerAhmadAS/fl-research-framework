import psutil
import gc
import json
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Union

import flwr as fl
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision import datasets, models, transforms

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

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


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
    return [
        p.detach().cpu().numpy()
        for p in model.parameters()
    ]


def set_params(
    model: nn.Module,
    params: List[np.ndarray],
) -> None:

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

    # --------------------------------------------------------
    # Data Profiling Configuration
    # --------------------------------------------------------

    profile_batch_size: int = int(
        os.environ.get("PROFILE_BATCH_SIZE", 64)
    )

    data_profile_cache_dir: str = os.environ.get(
        "DATA_PROFILE_CACHE_DIR",
        os.path.expanduser(
            "~/.cache/fl-research/data_profiles"
        ),
    )
    # --------------------------------------------------------
    # Data Partition Configuration
    # --------------------------------------------------------

    partition_mode: str = "iid"

    dirichlet_alpha: float = 0.5


    # --------------------------------------------------------
    # Proposed: System and Network Configuration
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
# CINIC-10
# ============================================================

def load_cinic10(data_dir: str):

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
    # Deterministic, without augmentation or normalization.
    # RGB values remain in [0, 1].
    # --------------------------------------------------------

    profile_tfm = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    train_ds = datasets.ImageFolder(
        root=os.path.join(data_dir, "train"),
        transform=train_tfm,
    )

    valid_ds = datasets.ImageFolder(
        root=os.path.join(data_dir, "valid"),
        transform=train_tfm,
    )

    test_ds = datasets.ImageFolder(
        root=os.path.join(data_dir, "test"),
        transform=test_tfm,
    )

    federated_train_ds = ConcatDataset(
        [train_ds, valid_ds]
    )

        # --------------------------------------------------------
    # Profiling dataset
    # Same samples and ordering as the training dataset.
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

    return (
        federated_train_ds,
        federated_profile_ds,
        test_ds,
    )


# ============================================================
# IID partitioning
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


# ============================================================
# Common Flower client
# ============================================================

class CommonFlowerClient(fl.client.NumPyClient):

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
        # Lazy model lifecycle
        # ----------------------------------------------------

        self.model = None
        self.criterion = None
        self.opt = None
        self.model_bytes = None
        # ----------------------------------------------------
        # Lazy Data Profiling
        #
        # The profile is not computed during initialization.
        # It will be loaded or computed only when requested.
        # ----------------------------------------------------

        self.data_profile = None

        # ----------------------------------------------------
        # Dataset
        # ----------------------------------------------------

        
        (
            train_ds,
            profile_ds,
            test_ds,
        ) = load_cinic10(
            cfg.data_dir
        )

        shards = partition_indices(
            num_samples=len(train_ds),
            num_clients=cfg.num_clients,
            seed=cfg.base_seed,
        )

        client_train_idx = shards[cid]

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
        # Profiling DataLoader
        #
        # Uses exactly the same client partition as training.
        # No shuffle to ensure deterministic profiling.
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

        self.test_loader = DataLoader(
            test_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    # ========================================================
    # Model lifecycle
    # ========================================================

    def _ensure_model_initialized(self) -> None:

        if self.model is not None:
            return

        print(
            f"[CLIENT {self.cid}] "
            "Initializing VGG16 model..."
        )

        self.model = build_vgg16(
            num_classes=10
        ).to(DEVICE)

        self.criterion = nn.CrossEntropyLoss()

        self.opt = optim.SGD(
            self.model.parameters(),
            lr=self.cfg.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )

        self.model_bytes = model_size_bytes(
            self.model
        )

        print(
            f"[CLIENT {self.cid}] "
            "VGG16 model initialized."
        )

    def _release_model(self) -> None:

        self.opt = None
        self.criterion = None
        self.model = None
        self.model_bytes = None

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(
            f"[CLIENT {self.cid}] "
            "Released VGG16 model resources."
        )

    # ========================================================
    # Flower get_parameters
    # ========================================================

    def _data_profile_cache_path(self) -> str:
        """
        Return a cache path uniquely associated with the
        client's data partition configuration.

        The cache key includes all settings that determine
        which samples belong to the logical client.
        """

        os.makedirs(
            self.cfg.data_profile_cache_dir,
            exist_ok=True,
        )

        if self.cfg.partition_mode == "dirichlet":
            partition_tag = (
                f"dirichlet_alpha_{self.cfg.dirichlet_alpha}"
            )
        else:
            partition_tag = "iid"

        filename = (
            f"cinic10_"
            f"clients_{self.cfg.num_clients}_"
            f"{partition_tag}_"
            f"seed_{self.cfg.base_seed}_"
            f"cid_{self.cid}.json"
        )

        return os.path.join(
            self.cfg.data_profile_cache_dir,
            filename,
        )


    def _load_or_compute_data_profile(
        self,
    ) -> Dict[str, object]:
        """
        Load the static local data profile from disk when it
        already exists.

        Otherwise compute it once and save it for reuse.

        This avoids repeating expensive image-based profiling
        when Ray recreates the same logical client.
        """

        cache_path = (
            self._data_profile_cache_path()
        )

        # ----------------------------------------------------
        # Reuse previously computed profile
        # ----------------------------------------------------

        if os.path.exists(cache_path):

            try:

                with open(
                    cache_path,
                    "r",
                    encoding="utf-8",
                ) as file:

                    profile = json.load(file)

                print(
                    f"[CLIENT {self.cid}] "
                    f"Loaded data profile from cache."
                )

                return profile

            except (
                OSError,
                json.JSONDecodeError,
            ) as exc:

                print(
                    f"[CLIENT {self.cid}] "
                    f"Could not read cached data profile: "
                    f"{exc}"
                )

                print(
                    f"[CLIENT {self.cid}] "
                    f"Recomputing data profile."
                )

        # ----------------------------------------------------
        # First use: compute profile normally
        # ----------------------------------------------------

        profile = self.compute_data_profile()

        # ----------------------------------------------------
        # Save the result
        # ----------------------------------------------------

        temporary_path = (
            cache_path
            + f".tmp.{os.getpid()}"
        )

        try:

            with open(
                temporary_path,
                "w",
                encoding="utf-8",
            ) as file:

                json.dump(
                    profile,
                    file,
                    indent=2,
                )

            os.replace(
                temporary_path,
                cache_path,
            )

            print(
                f"[CLIENT {self.cid}] "
                f"Saved data profile to cache."
            )

        finally:

            if os.path.exists(
                temporary_path
            ):
                try:
                    os.remove(
                        temporary_path
                    )
                except OSError:
                    pass

        return profile


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


    def get_parameters(self, config):

        self._ensure_model_initialized()

        parameters = get_params(
            self.model
        )

        self._release_model()

        return parameters

    # ========================================================
    # Local training
    # ========================================================

    def fit(
        self,
        parameters,
        config,
    ):

        self._ensure_model_initialized()

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

        # Oort utility is calculated only when required.
        enable_oort_utility = bool(
            config.get("enable_oort_utility", False)
        )

        oort_loss_sq_sum = 0.0

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


                if enable_oort_utility:
                    with torch.no_grad():
                        per_sample_losses = torch.nn.functional.cross_entropy(
                            logits.detach(),
                            y,
                            reduction="none",
                        )

                        oort_loss_sq_sum += float(
                            per_sample_losses.square().sum().item()
                        )

                total_loss += float(
                    loss.detach().cpu()
                )

                num_examples += x.shape[0]
                num_batches += 1

        fit_duration = (
            time.time() - start
        )

        avg_loss = (
            total_loss /
            max(1, num_batches)
        )

        samples_per_second = (
            num_examples /
            max(fit_duration, 1e-9)
        )

        gpu_memory_mb = 0.0

        if torch.cuda.is_available():
            gpu_memory_mb = (
                torch.cuda.max_memory_allocated()
                / (1024 ** 2)
            )

        current_model_bytes = int(
            self.model_bytes
        )

        metrics: Dict[str, float] = {
            "cid": int(self.cid),

            "train_loss":
                float(avg_loss),

            "fit_duration":
                float(fit_duration),

            "num_examples":
                int(num_examples),

            "samples_per_second":
                float(samples_per_second),

            "model_bytes":
                current_model_bytes,

            "upload_bytes":
                current_model_bytes,

            "download_bytes":
                current_model_bytes,

            "gpu_memory_mb":
                float(gpu_memory_mb),
        }
        
        if enable_oort_utility:
            oort_stat_utility = (
                num_examples
                * (
                    oort_loss_sq_sum
                    / max(1, num_examples)
                ) ** 0.5
            )

            metrics["oort_stat_utility"] = float(
                oort_stat_utility
            )

            metrics["oort_loss_sq_sum"] = float(
                oort_loss_sq_sum
            )

        print(
            f"[CLIENT {self.cid}] fit: "
            f"loss={avg_loss:.4f}, "
            f"time={fit_duration:.2f}s, "
            f"examples={num_examples}, "
            f"samples/s={samples_per_second:.2f}"
        )

        updated_parameters = get_params(
            self.model
        )

        self._release_model()

        return (
            updated_parameters,
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

        self._ensure_model_initialized()

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
            loss_sum /
            max(1, len(self.test_loader))
        )

        accuracy = (
            correct /
            max(1, total)
        )

        metrics = {
            "accuracy":
                float(accuracy),

            "eval_loss":
                float(avg_loss),

            "eval_duration":
                float(eval_duration),
        }

        self._release_model()

        return (
            float(avg_loss),
            int(total),
            metrics,
        )

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

    def get_properties(self, config):
        """
        Return client properties based on the strategy.

        Standard FedAvg and Oort:
            Return only the logical client ID.

        Proposed and Proposed+Oort:
            Load or compute the data profile lazily,
            measure current system capability,
            and check eligibility.
        """

        properties = {
            "cid": int(self.cid),
        }

        if not config.get("enable_proposed_profile", False):
            return properties

        # ----------------------------------------------------
        # Static data profile: lazy loading with cache
        # ----------------------------------------------------

        if self.data_profile is None:
            self.data_profile = (
                self._load_or_compute_data_profile()
            )

        data_profile = self.data_profile

        # ----------------------------------------------------
        # Dynamic system capability
        # ----------------------------------------------------

        system_profile = self.compute_system_profile()

        # ----------------------------------------------------
        # Eligibility
        # ----------------------------------------------------

        eligibility = self.check_eligibility(
            system_profile
        )

        # ----------------------------------------------------
        # Scalar properties expected by Proposed
        # ----------------------------------------------------

        properties.update({
            "eligible": bool(eligibility["eligible"]),
            "num_samples": int(data_profile["num_samples"]),

            "cpu_count": int(system_profile["cpu_count"]),
            "cpu_utilization_percent": float(
                system_profile["cpu_utilization_percent"]
            ),
            "total_ram_mb": float(
                system_profile["total_ram_mb"]
            ),
            "available_ram_mb": float(
                system_profile["available_ram_mb"]
            ),
            "ram_utilization_percent": float(
                system_profile["ram_utilization_percent"]
            ),

            "gpu_available": bool(
                system_profile["gpu_available"]
            ),
            "gpu_count": int(
                system_profile["gpu_count"]
            ),
            "gpu_total_memory_mb": float(
                system_profile["gpu_total_memory_mb"]
            ),
            "gpu_free_memory_mb": float(
                system_profile["gpu_free_memory_mb"]
            ),

            "upload_bandwidth_mbps": float(
                system_profile["upload_bandwidth_mbps"]
            ),
            "download_bandwidth_mbps": float(
                system_profile["download_bandwidth_mbps"]
            ),
            "latency_ms": float(
                system_profile["latency_ms"]
            ),
        })

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
        # RGB mean and standard deviation
        # ----------------------------------------------------

        for channel_name, value in zip(
            ["r", "g", "b"],
            data_profile["channel_mean"],
        ):
            properties[
                f"rgb_mean_{channel_name}"
            ] = float(value)

        for channel_name, value in zip(
            ["r", "g", "b"],
            data_profile["channel_std"],
        ):
            properties[
                f"rgb_std_{channel_name}"
            ] = float(value)

        return properties


# ============================================================
# Global configuration
# ============================================================

CFG = DataConfig()


# ============================================================
# Flower client factory
# ============================================================

def client_fn(
    arg: Union[str, "Context"],
) -> fl.client.Client:

    if isinstance(arg, str):
        cid = int(arg)

    else:
        if (
            hasattr(arg, "node_config")
            and
            "partition-id" in arg.node_config
        ):
            cid = int(
                arg.node_config["partition-id"]
            )

        else:
            cid = int(
                getattr(
                    arg,
                    "node_id",
                    0,
                )
            )

    return CommonFlowerClient(
        cid=cid,
        cfg=CFG,
    ).to_client()