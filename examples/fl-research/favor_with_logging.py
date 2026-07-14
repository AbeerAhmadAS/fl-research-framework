# Experiment-only logging layer for FAVOR.
#
# This class inherits the core FAVOR strategy and records results.
# It does not alter PCA, DDQN, reward, selection, or aggregation logic.

from __future__ import annotations

import csv
import os
import platform
import time
from typing import Dict

import torch

from flwr.server.strategy import Favor


class FavorWithLogging(Favor):
    def __init__(self, log_dir: str, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        self.round_csv = os.path.join(
            self.log_dir, "round_metrics.csv"
        )
        self.hardware_csv = os.path.join(
            self.log_dir, "hardware_specs.csv"
        )
        self.round_start: Dict[int, float] = {}

        self._initialize_round_csv()
        self._write_hardware_specs()

    def _initialize_round_csv(self) -> None:
        if os.path.exists(self.round_csv):
            return

        with open(self.round_csv, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "episode",
                    "round",
                    "mode",
                    "phase",
                    "selected_clients",
                    "num_selected_clients",
                    "accuracy",
                    "eval_loss",
                    "reward",
                    "dqn_loss",
                    "epsilon",
                    "replay_size",
                    "round_duration_sec",
                    "target_reached",
                ]
            )

    def _write_hardware_specs(self) -> None:
        if os.path.exists(self.hardware_csv):
            return

        cuda_available = torch.cuda.is_available()
        values = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "cuda_available": str(cuda_available),
            "cuda_version": str(torch.version.cuda),
            "gpu_count": str(
                torch.cuda.device_count() if cuda_available else 0
            ),
            "gpu_name": (
                torch.cuda.get_device_name(0)
                if cuda_available
                else "None"
            ),
            "cpu_count": str(os.cpu_count()),
            "strategy": "FAVOR",
            "num_clients": str(self.num_clients),
            "selected_clients": str(self.selected_clients),
            "pca_components": str(
                self.state_encoder.n_components
            ),
            "target_accuracy": str(self.target_accuracy),
        }

        with open(self.hardware_csv, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["key", "value"])
            writer.writerows(values.items())

    def configure_fit(self, server_round, parameters, client_manager):
        self.round_start[server_round] = time.time()
        selected = super().configure_fit(
            server_round, parameters, client_manager
        )

        print(
            f"[FAVOR][EPISODE {self.episode}][ROUND {server_round}] "
            f"mode={self.mode}, phase={self.last_phase}, "
            f"selected={self.last_selected_cids}"
        )
        return selected

    def evaluate(self, server_round, parameters):
        result = super().evaluate(server_round, parameters)
        if result is None:
            return None

        loss, metrics = result

        # Round zero is the initial centralized evaluation and has no fit time.
        if server_round == 0:
            return result

        duration = time.time() - self.round_start.get(
            server_round, time.time()
        )

        row = [
            self.episode,
            server_round,
            self.mode,
            self.last_phase,
            "|".join(self.last_selected_cids),
            len(self.last_selected_cids),
            self.last_accuracy,
            float(loss),
            self.last_reward,
            self.last_dqn_loss,
            self.agent.epsilon if self.mode == "train" else 0.0,
            len(self.agent.replay),
            duration,
            self.episode_done,
        ]

        with open(self.round_csv, "a", newline="") as file:
            csv.writer(file).writerow(row)

        print(
            f"[FAVOR][EVAL][EPISODE {self.episode}] "
            f"round={server_round}, accuracy={self.last_accuracy:.4f}, "
            f"loss={float(loss):.4f}, reward={self.last_reward}, "
            f"dqn_loss={self.last_dqn_loss}, done={self.episode_done}"
        )

        return result
