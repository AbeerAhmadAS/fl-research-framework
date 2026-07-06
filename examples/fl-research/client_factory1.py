import os
import time
import math
import random
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


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Download VGG16
# However, the original VGG16 is designed to classify 1000 classes, while CINIC-10 only has 10 classes.
def build_vgg16(num_classes: int = 10) -> nn.Module:
    weights = models.VGG16_Weights.IMAGENET1K_V1
    model = models.vgg16(weights=weights)

    model.classifier[6] = nn.Linear(
        in_features=model.classifier[6].in_features,
        out_features=num_classes,
    )

    return model


# This function takes the model weights from PyTorch and converts them to NumPy arrays so that Flower can send them to the server.
def get_params(model: nn.Module) -> List[np.ndarray]:
    return [p.detach().cpu().numpy() for p in model.parameters()]


# It does the opposite; it takes the weights coming from the server and places them inside the client model.
def set_params(model: nn.Module, params: List[np.ndarray]) -> None:
    with torch.no_grad():
        for p, np_p in zip(model.parameters(), params):
            p.copy_(torch.tensor(np_p, dtype=p.dtype, device=p.device))


# This function calculates the model's weights in bytes.
# We will later use it to calculate: "download_bytes", "upload_bytes", and "total_transmitted_mb".
# This is important for the "Communication Efficiency" metric.
def model_size_bytes(model: nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters())


# Data and Experiment setup
@dataclass
class DataConfig:
    num_clients: int = int(os.environ.get("NUM_CLIENTS", 20))
    batch_size: int = int(os.environ.get("BATCH_SIZE", 16))
    data_dir: str = os.environ.get(
        "CINIC10_DIR",
        os.path.expanduser("~/.data/CINIC-10"),
    )
    base_seed: int = int(os.environ.get("BASE_SEED", 1234))
    lr: float = float(os.environ.get("LR", 0.001))
    local_epochs: int = int(os.environ.get("LOCAL_EPOCHS", 1))
    num_workers: int = int(os.environ.get("NUM_WORKERS", 2))


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

    federated_train_ds = ConcatDataset([train_ds, valid_ds])

    return federated_train_ds, test_ds


# This function divides the training data by the number of client.
# "seed" makes the division constant in every experiment
def partition_indices(num_samples: int, num_clients: int, seed: int) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(num_samples)
    rng.shuffle(idx)
    return np.array_split(idx, num_clients)


class FlowerClient(fl.client.NumPyClient):
    # It works when the client is created
    def __init__(self, cid: int, cfg: DataConfig):
        self.cid = cid
        self.cfg = cfg

        # "Stabilizing randomness" so that each client has consistent and retestable behavior.
        seed_everything(cfg.base_seed + cid)

        # "Building the model" means that each client gets a copy of VGG16.
        self.model = build_vgg16(num_classes=10).to(DEVICE)

        # Definition of losses, because the task is classification.
        # This one is used for actual model training.
        self.criterion = nn.CrossEntropyLoss()

        # Used only to compute Oort statistical utility.
        # This does not affect training because training still uses self.criterion.
        self.oort_criterion = nn.CrossEntropyLoss(reduction="none")

        # We use Stochastic Gradient Descent to update the model's weights during local training. (optimizer)
        self.opt = optim.SGD(
            self.model.parameters(),
            lr=cfg.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )

        # Preparing client data
        train_ds, test_ds = load_cinic10(cfg.data_dir)
        shards = partition_indices(
            num_samples=len(train_ds),
            num_clients=cfg.num_clients,
            seed=cfg.base_seed,
        )

        client_train_idx = shards[cid]

        self.train_loader = DataLoader(
            Subset(train_ds, client_train_idx),
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

        self.model_bytes = model_size_bytes(self.model)

    # The client's weights are sent to the server. This is often used initially to obtain initial parameters.
    def get_parameters(self, config):
        return get_params(self.model)

    def fit(self, parameters, config):
        set_params(self.model, parameters)
        self.model.train()

        local_epochs = int(config.get("local_epochs", self.cfg.local_epochs))

        # FedFS deadline-based local training.
        # If fit_deadline_sec > 0, the client stops training when the deadline is reached.
        # The client still returns the partially trained model, which represents partial work.
        fit_deadline_sec = float(config.get("fit_deadline_sec", 0.0))

        start = time.time()
        total_loss = 0.0
        num_examples = 0
        num_batches = 0
        stopped_by_deadline = False

        # Oort metric only:
        # stores sum of squared per-sample losses.
        oort_loss_sq_sum = 0.0

        max_possible_examples = len(self.train_loader.dataset) * local_epochs

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        for _ in range(local_epochs):
            for x, y in self.train_loader:
                if fit_deadline_sec > 0.0 and (time.time() - start) >= fit_deadline_sec:
                    stopped_by_deadline = True
                    break

                x, y = x.to(DEVICE), y.to(DEVICE)

                self.opt.zero_grad()
                logits = self.model(x)
                loss = self.criterion(logits, y)
                loss.backward()
                self.opt.step()

                # Oort statistical utility:
                # U(i) = |B_i| * sqrt(mean(Loss(k)^2))
                # This is computed separately and does not affect model training.
                with torch.no_grad():
                    sample_losses = self.oort_criterion(logits.detach(), y)
                    oort_loss_sq_sum += float(torch.sum(sample_losses ** 2).cpu())

                total_loss += float(loss.detach().cpu())
                num_examples += x.shape[0]
                num_batches += 1

            if stopped_by_deadline:
                break

        fit_duration = time.time() - start
        avg_loss = total_loss / max(1, num_batches)

        # Oort statistical utility based on the paper approximation.
        oort_stat_utility = num_examples * math.sqrt(
            oort_loss_sq_sum / max(1, num_examples)
        )

        samples_per_second = num_examples / max(fit_duration, 1e-9)

        # FedFS work contribution ratio.
        # wk = actual processed samples / maximum possible samples
        work_ratio = num_examples / max(1, max_possible_examples)

        gpu_memory_mb = 0.0
        if torch.cuda.is_available():
            gpu_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

        # to send to server
        metrics: Dict[str, float] = {
            "cid": int(self.cid),
            "train_loss": float(avg_loss),
            "fit_duration": float(fit_duration),
            "num_examples": int(num_examples),
            "max_possible_examples": int(max_possible_examples),
            "work_ratio": float(work_ratio),
            "stopped_by_deadline": bool(stopped_by_deadline),
            "fit_deadline_sec": float(fit_deadline_sec),
            "samples_per_second": float(samples_per_second),
            "model_bytes": int(self.model_bytes),
            "upload_bytes": int(self.model_bytes),
            "download_bytes": int(self.model_bytes),
            "gpu_memory_mb": float(gpu_memory_mb),

            # Extra metrics for Oort.
            # FedAvg and FedFS can safely ignore these values.
            "oort_stat_utility": float(oort_stat_utility),
            "oort_loss_sq_sum": float(oort_loss_sq_sum),
        }

        print(
            f"[CLIENT {self.cid}] fit: "
            f"loss={avg_loss:.4f}, "
            f"time={fit_duration:.2f}s, "
            f"examples={num_examples}/{max_possible_examples}, "
            f"work_ratio={work_ratio:.4f}, "
            f"deadline={fit_deadline_sec:.2f}s, "
            f"partial={stopped_by_deadline}, "
            f"samples/s={samples_per_second:.2f}, "
            f"oort_stat_utility={oort_stat_utility:.4f}"
        )

        # In FedFS, aggregation should be weighted by actual work done.
        # Therefore, we return num_examples as the number of processed samples.
        return get_params(self.model), num_examples, metrics

    def evaluate(self, parameters, config):
        set_params(self.model, parameters)
        self.model.eval()

        loss_sum = 0.0
        correct = 0
        total = 0

        start = time.time()

        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)

                logits = self.model(x)
                loss = self.criterion(logits, y)

                loss_sum += float(loss.detach().cpu())
                pred = logits.argmax(dim=1)
                correct += int((pred == y).sum().cpu())
                total += x.shape[0]

        eval_duration = time.time() - start
        avg_loss = loss_sum / max(1, len(self.test_loader))
        acc = correct / max(1, total)
        return float(avg_loss), int(total), {
            "accuracy": float(acc),
            "eval_loss": float(avg_loss),
            "eval_duration": float(eval_duration),
        }


# The results are used by the server to calculate average accuracy and loss.
CFG = DataConfig()


# This is the function that Flower uses to create virtual clients.
def client_fn(arg: Union[str, "Context"]) -> fl.client.Client:
    if isinstance(arg, str):
        cid = int(arg)
    else:
        if hasattr(arg, "node_config") and "partition-id" in arg.node_config:
            cid = int(arg.node_config["partition-id"])
        else:
            cid = int(getattr(arg, "node_id", 0))

    return FlowerClient(cid=cid, cfg=CFG).to_client()
