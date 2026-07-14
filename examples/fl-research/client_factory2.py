# Client factory dedicated to the FAVOR MNIST experiment.
#
# This file is independent from client_factory1.py and does not modify it.
# It contains only:
# - the MNIST CNN
# - the paper-style non-IID partition
# - local client training/evaluation
# - centralized server evaluation helpers

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Union

import flwr as fl
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

try:
    from flwr.common import Context
except Exception:
    Context = object


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@dataclass(frozen=True)
class FavorTaskConfig:
    num_clients: int = int(os.environ.get("NUM_CLIENTS", 20))
    local_epochs: int = int(os.environ.get("LOCAL_EPOCHS", 5))
    batch_size: int = int(os.environ.get("BATCH_SIZE", 10))
    samples_per_client: int = int(
        os.environ.get("SAMPLES_PER_CLIENT", 600)
    )
    dominant_fraction: float = float(
        os.environ.get("DOMINANT_FRACTION", 0.8)
    )
    learning_rate: float = float(os.environ.get("CLIENT_LR", 0.01))
    momentum: float = float(os.environ.get("CLIENT_MOMENTUM", 0.0))
    base_seed: int = int(os.environ.get("BASE_SEED", 1234))
    num_workers: int = int(os.environ.get("NUM_WORKERS", 0))
    data_dir: str = os.environ.get(
        "MNIST_DIR", os.path.expanduser("~/.data")
    )
    client_device: str = os.environ.get(
        "CLIENT_DEVICE",
        "cuda" if torch.cuda.is_available() else "cpu",
    )
    server_device: str = os.environ.get("SERVER_DEVICE", "cpu")


class MNISTFavorCNN(nn.Module):
    """Small CNN matching the architecture described for MNIST in FAVOR."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(50 * 4 * 4, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


def get_parameters(model: nn.Module) -> List[np.ndarray]:
    return [
        parameter.detach().cpu().numpy()
        for parameter in model.state_dict().values()
    ]


def set_parameters(model: nn.Module, parameters: List[np.ndarray]) -> None:
    state_dict = model.state_dict()
    if len(parameters) != len(state_dict):
        raise ValueError("Received an incorrect number of model tensors.")

    new_state = {
        key: torch.tensor(value, dtype=state_dict[key].dtype)
        for key, value in zip(state_dict.keys(), parameters)
    }
    model.load_state_dict(new_state, strict=True)


def _build_label_count_matrix(
    num_clients: int,
    samples_per_client: int,
    dominant_fraction: float,
    num_classes: int = 10,
) -> np.ndarray:
    """Create deterministic 80%-dominant per-client class counts.

    The remaining 20% is distributed as evenly as possible over the other
    labels. Rotation prevents the same labels from always receiving the
    extra remainder samples.
    """
    if num_clients <= 0 or samples_per_client <= 0:
        raise ValueError("Client and sample counts must be positive.")
    if not 0.0 < dominant_fraction <= 1.0:
        raise ValueError("dominant_fraction must be in (0, 1].")

    dominant_count = int(round(samples_per_client * dominant_fraction))
    remaining = samples_per_client - dominant_count

    counts = np.zeros((num_clients, num_classes), dtype=np.int64)

    for cid in range(num_clients):
        dominant_label = cid % num_classes
        counts[cid, dominant_label] = dominant_count

        other_labels = [
            (dominant_label + 1 + offset) % num_classes
            for offset in range(num_classes - 1)
        ]

        base = remaining // (num_classes - 1)
        extra = remaining % (num_classes - 1)

        rotation = (cid // num_classes) % (num_classes - 1)
        other_labels = (
            other_labels[rotation:] + other_labels[:rotation]
        )

        for position, label in enumerate(other_labels):
            counts[cid, label] = base + (
                1 if position < extra else 0
            )

    if not np.all(counts.sum(axis=1) == samples_per_client):
        raise RuntimeError("Invalid per-client allocation was generated.")

    return counts


def create_noniid_partitions(
    targets: np.ndarray,
    num_clients: int,
    samples_per_client: int,
    dominant_fraction: float,
    seed: int,
) -> List[np.ndarray]:
    """Create non-overlapping paper-style non-IID MNIST partitions."""
    targets = np.asarray(targets, dtype=np.int64)
    num_classes = int(targets.max()) + 1
    rng = np.random.default_rng(seed)

    counts = _build_label_count_matrix(
        num_clients=num_clients,
        samples_per_client=samples_per_client,
        dominant_fraction=dominant_fraction,
        num_classes=num_classes,
    )

    class_pools: Dict[int, np.ndarray] = {}
    for label in range(num_classes):
        indices = np.where(targets == label)[0]
        rng.shuffle(indices)
        class_pools[label] = indices

    required_per_label = counts.sum(axis=0)
    available_per_label = np.array(
        [len(class_pools[label]) for label in range(num_classes)]
    )

    if np.any(required_per_label > available_per_label):
        details = {
            label: {
                "required": int(required_per_label[label]),
                "available": int(available_per_label[label]),
            }
            for label in range(num_classes)
            if required_per_label[label] > available_per_label[label]
        }
        raise ValueError(
            "The requested non-IID split requires more samples than "
            f"MNIST provides for some labels: {details}. "
            "For the exact 600-sample setting, use 20 clients as configured. "
            "A 100-client reproduction requires a separate policy for the "
            "small natural MNIST class-count imbalance."
        )

    cursors = np.zeros(num_classes, dtype=np.int64)
    partitions: List[np.ndarray] = []

    for cid in range(num_clients):
        selected_parts = []
        for label in range(num_classes):
            amount = int(counts[cid, label])
            start = int(cursors[label])
            end = start + amount
            selected_parts.append(class_pools[label][start:end])
            cursors[label] = end

        client_indices = np.concatenate(selected_parts)
        rng.shuffle(client_indices)
        partitions.append(client_indices.astype(np.int64))

    all_used = np.concatenate(partitions)
    if len(np.unique(all_used)) != len(all_used):
        raise RuntimeError("The generated client partitions overlap.")

    return partitions


def load_mnist(config: FavorTaskConfig):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_set = datasets.MNIST(
        config.data_dir,
        train=True,
        download=True,
        transform=transform,
    )
    test_set = datasets.MNIST(
        config.data_dir,
        train=False,
        download=True,
        transform=transform,
    )
    return train_set, test_set


class FavorMNISTClient(fl.client.NumPyClient):
    def __init__(
        self,
        cid: int,
        train_set,
        train_indices: np.ndarray,
        test_set,
        config: FavorTaskConfig,
    ):
        self.cid = int(cid)
        self.config = config
        seed_everything(config.base_seed + self.cid)

        self.device = torch.device(config.client_device)
        self.model = MNISTFavorCNN().to(self.device)
        self.criterion = nn.CrossEntropyLoss()

        generator = torch.Generator()
        generator.manual_seed(config.base_seed + self.cid)

        self.train_loader = DataLoader(
            Subset(train_set, train_indices.tolist()),
            batch_size=config.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=config.num_workers,
            pin_memory=self.device.type == "cuda",
        )
        self.test_loader = DataLoader(
            test_set,
            batch_size=256,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=self.device.type == "cuda",
        )

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        self.model.train()

        local_epochs = int(
            config.get("local_epochs", self.config.local_epochs)
        )
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.config.learning_rate,
            momentum=self.config.momentum,
        )

        start = time.time()
        total_loss = 0.0
        total_examples = 0
        total_batches = 0

        for _ in range(local_epochs):
            for images, labels in self.train_loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                optimizer.zero_grad()
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                loss.backward()
                optimizer.step()

                total_loss += float(loss.detach().cpu())
                total_examples += int(labels.shape[0])
                total_batches += 1

        duration = time.time() - start
        average_loss = total_loss / max(total_batches, 1)

        metrics = {
            "logical_cid": self.cid,
            "train_loss": float(average_loss),
            "fit_duration": float(duration),
            "samples_per_second": float(
                total_examples / max(duration, 1e-9)
            ),
        }

        return (
            get_parameters(self.model),
            len(self.train_loader.dataset),
            metrics,
        )

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        loss, accuracy = evaluate_model(
            self.model, self.test_loader, self.device
        )
        return loss, len(self.test_loader.dataset), {
            "accuracy": accuracy,
            "eval_loss": loss,
        }


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            total_loss += float(criterion(logits, labels).cpu())
            correct += int((logits.argmax(dim=1) == labels).sum().cpu())
            total += int(labels.shape[0])

    return (
        total_loss / max(total, 1),
        correct / max(total, 1),
    )


def prepare_favor_task(config: FavorTaskConfig):
    train_set, test_set = load_mnist(config)
    targets = np.asarray(train_set.targets, dtype=np.int64)

    partitions = create_noniid_partitions(
        targets=targets,
        num_clients=config.num_clients,
        samples_per_client=config.samples_per_client,
        dominant_fraction=config.dominant_fraction,
        seed=config.base_seed,
    )

    server_device = torch.device(config.server_device)
    test_loader = DataLoader(
        test_set,
        batch_size=256,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=server_device.type == "cuda",
    )

    def client_fn(arg: Union[str, "Context"]) -> fl.client.Client:
        if isinstance(arg, str):
            cid = int(arg)
        elif (
            hasattr(arg, "node_config")
            and "partition-id" in arg.node_config
        ):
            cid = int(arg.node_config["partition-id"])
        else:
            cid = int(getattr(arg, "node_id", 0))

        return FavorMNISTClient(
            cid=cid,
            train_set=train_set,
            train_indices=partitions[cid],
            test_set=test_set,
            config=config,
        ).to_client()

    def centralized_evaluate(server_round, parameters, config_dict):
        model = MNISTFavorCNN().to(server_device)
        set_parameters(model, parameters)
        loss, accuracy = evaluate_model(
            model, test_loader, server_device
        )
        return float(loss), {"accuracy": float(accuracy)}

    seed_everything(config.base_seed)
    initial_model = MNISTFavorCNN()
    initial_parameters = get_parameters(initial_model)

    return client_fn, centralized_evaluate, initial_parameters
