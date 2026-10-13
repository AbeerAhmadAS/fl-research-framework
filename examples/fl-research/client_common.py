import gc
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

    return federated_train_ds, test_ds


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
        # Dataset
        # ----------------------------------------------------

        train_ds, test_ds = load_cinic10(
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

    def get_properties(self, config):
        """
        Return only the stable logical client ID.

        This is intentionally lightweight and does not compute
        data profiles, system capability, eligibility, or any
        Proposed-specific information.
        """
        return {
            "cid": int(self.cid),
        }


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