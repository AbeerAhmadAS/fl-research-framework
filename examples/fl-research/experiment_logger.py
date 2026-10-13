# experiment_logger.py
#
# Common experiment logging and evaluation utilities.
#
# This file is strategy-independent.
# It does NOT implement client selection logic.
#
# Responsibilities:
# 1) Record selected/successful clients
# 2) Track participation
# 3) Calculate Jain's Fairness Index
# 4) Calculate communication cost
# 5) Measure training round duration
# 6) Measure time-to-target accuracy
# 7) Store common experiment metrics in CSV
# 8) Store hardware specifications
#
# Strategy-specific metrics should NOT be calculated here.

import csv
import os
import platform
import time
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch


def jain_fairness(values: List[int]) -> float:
    """Calculate Jain's Fairness Index."""

    arr = np.array(values, dtype=np.float64)

    if arr.size == 0 or arr.sum() == 0:
        return 0.0

    denominator = len(arr) * np.sum(arr ** 2)

    if denominator == 0:
        return 0.0

    return float((arr.sum() ** 2) / denominator)


class ExperimentLogger:
    """Common logger for federated learning experiments."""

    def __init__(
        self,
        num_clients: int,
        log_dir: str,
        strategy_name: str,
        target_accuracy: float = 0.70,
    ):
        self.num_clients = int(num_clients)
        self.log_dir = log_dir
        self.strategy_name = strategy_name
        self.target_accuracy = float(target_accuracy)

        os.makedirs(self.log_dir, exist_ok=True)

        # ---------------------------------------------------------
        # Global experiment timing
        # ---------------------------------------------------------
        self.global_start_time = time.time()

        self.first_target_round: Optional[int] = None
        self.first_target_time: Optional[float] = None

        # ---------------------------------------------------------
        # Round timing
        #
        # Round duration is defined as:
        #
        # start:
        #   immediately before strategy.configure_fit()
        #
        # end:
        #   immediately after strategy.aggregate_fit()
        #
        # Evaluation time is NOT part of round_duration_sec.
        # ---------------------------------------------------------
        self.round_start_time: Dict[int, float] = {}
        self.round_duration: Dict[int, float] = {}

        # ---------------------------------------------------------
        # Participation
        #
        # IMPORTANT:
        # Keys must be LOGICAL client IDs:
        # "0", "1", ..., str(num_clients - 1)
        #
        # This guarantees that clients with zero participation are
        # included in Jain's Fairness Index.
        # ---------------------------------------------------------
        self.participation = defaultdict(int)

        # ---------------------------------------------------------
        # Per-round state
        # ---------------------------------------------------------
        self.round_data: Dict[int, Dict] = {}

        # ---------------------------------------------------------
        # Output files
        # ---------------------------------------------------------
        self.round_csv = os.path.join(
            self.log_dir,
            "round_metrics.csv",
        )

        self.hardware_csv = os.path.join(
            self.log_dir,
            "hardware_specs.csv",
        )

        self._init_round_csv()
        self._write_hardware_specs()

    # =============================================================
    # CSV INITIALIZATION
    # =============================================================

    def _init_round_csv(self):
        fieldnames = [
            "round",
            "selected_clients",
            "successful_clients",
            "num_selected_clients",
            "num_successful_clients",
            "avg_train_loss",
            "avg_accuracy",
            "avg_eval_loss",
            "round_duration_sec",
            "avg_fit_duration_sec",
            "total_fit_duration_sec",
            "avg_samples_per_second",
            "avg_gpu_memory_mb",
            "model_bytes",
            "download_bytes",
            "upload_bytes",
            "total_transmitted_bytes",
            "total_transmitted_mb",
            "jain_fairness",
            "participation_std",
            "min_participation",
            "max_participation",
            "converged",
            "time_to_target_sec",
        ]

        with open(
            self.round_csv,
            "w",
            newline="",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
            )
            writer.writeheader()

    # =============================================================
    # HARDWARE INFORMATION
    # =============================================================

    def _write_hardware_specs(self):
        gpu_name = "None"
        gpu_count = 0

        cuda_available = torch.cuda.is_available()

        if cuda_available:
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)

        specs = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "cuda_available": str(cuda_available),
            "cuda_version": str(torch.version.cuda),
            "gpu_count": str(gpu_count),
            "gpu_name": gpu_name,
            "cpu_count": str(os.cpu_count()),
            "strategy": self.strategy_name,
            "num_clients": str(self.num_clients),
            "target_accuracy": str(self.target_accuracy),
        }

        with open(
            self.hardware_csv,
            "w",
            newline="",
        ) as f:
            writer = csv.writer(f)
            writer.writerow(["key", "value"])

            for key, value in specs.items():
                writer.writerow([key, value])

    # =============================================================
    # ROUND START
    # =============================================================

    def start_round(self, server_round: int):
        """Start timing a training round."""

        self.round_start_time[server_round] = time.time()

        if server_round not in self.round_data:
            self.round_data[server_round] = {}

    # =============================================================
    # CLIENT SELECTION
    # =============================================================

    def record_selection(
        self,
        server_round: int,
        selected_client_ids: List[str],
    ):
        """
        Record selected LOGICAL client IDs.

        Example:
            ["0", "4", "7", "11", "15"]

        Do not pass Flower/Ray proxy IDs here.
        """

        selected_client_ids = [
            str(cid)
            for cid in selected_client_ids
        ]

        if server_round not in self.round_data:
            self.round_data[server_round] = {}

        self.round_data[server_round][
            "selected_clients"
        ] = selected_client_ids

        for cid in selected_client_ids:
            self.participation[cid] += 1

    # =============================================================
    # FIT RESULTS
    # =============================================================

    def record_fit_results(
        self,
        server_round: int,
        successful_client_ids: List[str],
        train_losses: List[float],
        num_examples: List[int],
        fit_durations: List[float],
        samples_per_second: List[float],
        gpu_memory_mb: List[float],
        model_bytes: int,
    ):
        """
        Record common metrics after local training and aggregation.

        Round duration ends here.

        Communication:
            download = model_bytes * selected clients
            upload   = model_bytes * successful clients
        """

        if server_round not in self.round_data:
            self.round_data[server_round] = {}

        data = self.round_data[server_round]

        successful_client_ids = [
            str(cid)
            for cid in successful_client_ids
        ]

        selected_client_ids = data.get(
            "selected_clients",
            [],
        )

        # ---------------------------------------------------------
        # End training-round timer
        # ---------------------------------------------------------
        start_time = self.round_start_time.get(server_round)

        if start_time is None:
            round_duration_sec = 0.0
        else:
            round_duration_sec = (
                time.time() - start_time
            )

        self.round_duration[
            server_round
        ] = round_duration_sec

        # ---------------------------------------------------------
        # Common fit metrics
        # ---------------------------------------------------------
        total_examples = sum(num_examples)

        if train_losses and total_examples > 0:
            avg_train_loss = float(
                sum(
                    n * loss
                    for n, loss in zip(
                        num_examples,
                        train_losses,
                    )
                )
                / total_examples
            )
        else:
            avg_train_loss = 0.0
      
        avg_fit_duration = (
            float(np.mean(fit_durations))
            if fit_durations
            else 0.0
        )

        total_fit_duration = (
            float(np.sum(fit_durations))
            if fit_durations
            else 0.0
        )

        avg_samples_per_second = (
            float(np.mean(samples_per_second))
            if samples_per_second
            else 0.0
        )

        avg_gpu_memory = (
            float(np.mean(gpu_memory_mb))
            if gpu_memory_mb
            else 0.0
        )

        # ---------------------------------------------------------
        # Communication
        #
        # Every selected client receives the global model.
        # Only successful clients return an updated model.
        # ---------------------------------------------------------
        num_selected = len(selected_client_ids)
        num_successful = len(successful_client_ids)

        model_bytes = int(model_bytes)

        download_bytes = (
            model_bytes * num_selected
        )

        upload_bytes = (
            model_bytes * num_successful
        )

        total_transmitted_bytes = (
            download_bytes + upload_bytes
        )

        total_transmitted_mb = (
            total_transmitted_bytes / (1024 ** 2)
        )

        # ---------------------------------------------------------
        # Jain fairness
        #
        # ALWAYS include all clients:
        # 0 ... num_clients-1
        # ---------------------------------------------------------
        participation_values = [
            self.participation.get(str(i), 0)
            for i in range(self.num_clients)
        ]

        fairness = jain_fairness(
            participation_values
        )

        participation_std = (
            float(np.std(participation_values))
            if participation_values
            else 0.0
        )

        min_participation = (
            int(np.min(participation_values))
            if participation_values
            else 0
        )

        max_participation = (
            int(np.max(participation_values))
            if participation_values
            else 0
        )

        # ---------------------------------------------------------
        # Store round data
        # ---------------------------------------------------------
        data.update(
            {
                "successful_clients":
                    successful_client_ids,

                "num_selected_clients":
                    num_selected,

                "num_successful_clients":
                    num_successful,

                "avg_train_loss":
                    avg_train_loss,

                "round_duration_sec":
                    float(round_duration_sec),

                "avg_fit_duration_sec":
                    avg_fit_duration,

                "total_fit_duration_sec":
                    total_fit_duration,

                "avg_samples_per_second":
                    avg_samples_per_second,

                "avg_gpu_memory_mb":
                    avg_gpu_memory,

                "model_bytes":
                    model_bytes,

                "download_bytes":
                    download_bytes,

                "upload_bytes":
                    upload_bytes,

                "total_transmitted_bytes":
                    total_transmitted_bytes,

                "total_transmitted_mb":
                    float(total_transmitted_mb),

                "jain_fairness":
                    fairness,

                "participation_std":
                    participation_std,

                "min_participation":
                    min_participation,

                "max_participation":
                    max_participation,
            }
        )

    # =============================================================
    # EVALUATION RESULTS
    # =============================================================

    def record_evaluation(
        self,
        server_round: int,
        accuracy: float,
        eval_loss: float,
    ):
        """
        Record global evaluation metrics and calculate
        time-to-target.
        """

        if server_round not in self.round_data:
            self.round_data[server_round] = {}

        data = self.round_data[server_round]

        accuracy = float(accuracy)
        eval_loss = float(eval_loss)

        converged = (
            accuracy >= self.target_accuracy
        )

        # ---------------------------------------------------------
        # First time target accuracy is reached
        # ---------------------------------------------------------
        if (
            converged
            and self.first_target_round is None
        ):
            self.first_target_round = server_round

            self.first_target_time = (
                time.time()
                - self.global_start_time
            )

        data.update(
            {
                "avg_accuracy":
                    accuracy,

                "avg_eval_loss":
                    eval_loss,

                "converged":
                    converged,

                "time_to_target_sec":
                    (
                        ""
                        if self.first_target_time is None
                        else float(
                            self.first_target_time
                        )
                    ),
            }
        )

        # A round is complete for common logging after evaluation.
        self._write_all_rounds()

    # =============================================================
    # WRITE CSV
    # =============================================================

    def _write_all_rounds(self):
        fieldnames = [
            "round",
            "selected_clients",
            "successful_clients",
            "num_selected_clients",
            "num_successful_clients",
            "avg_train_loss",
            "avg_accuracy",
            "avg_eval_loss",
            "round_duration_sec",
            "avg_fit_duration_sec",
            "total_fit_duration_sec",
            "avg_samples_per_second",
            "avg_gpu_memory_mb",
            "model_bytes",
            "download_bytes",
            "upload_bytes",
            "total_transmitted_bytes",
            "total_transmitted_mb",
            "jain_fairness",
            "participation_std",
            "min_participation",
            "max_participation",
            "converged",
            "time_to_target_sec",
        ]

        with open(
            self.round_csv,
            "w",
            newline="",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            for server_round in sorted(
                self.round_data.keys()
            ):
                data = self.round_data[
                    server_round
                ]

                row = {
                    "round":
                        server_round,

                    "selected_clients":
                        "|".join(
                            data.get(
                                "selected_clients",
                                [],
                            )
                        ),

                    "successful_clients":
                        "|".join(
                            data.get(
                                "successful_clients",
                                [],
                            )
                        ),

                    "num_selected_clients":
                        data.get(
                            "num_selected_clients",
                            0,
                        ),

                    "num_successful_clients":
                        data.get(
                            "num_successful_clients",
                            0,
                        ),

                    "avg_train_loss":
                        data.get(
                            "avg_train_loss",
                            "",
                        ),

                    "avg_accuracy":
                        data.get(
                            "avg_accuracy",
                            "",
                        ),

                    "avg_eval_loss":
                        data.get(
                            "avg_eval_loss",
                            "",
                        ),

                    "round_duration_sec":
                        data.get(
                            "round_duration_sec",
                            "",
                        ),

                    "avg_fit_duration_sec":
                        data.get(
                            "avg_fit_duration_sec",
                            "",
                        ),

                    "total_fit_duration_sec":
                        data.get(
                            "total_fit_duration_sec",
                            "",
                        ),

                    "avg_samples_per_second":
                        data.get(
                            "avg_samples_per_second",
                            "",
                        ),

                    "avg_gpu_memory_mb":
                        data.get(
                            "avg_gpu_memory_mb",
                            "",
                        ),

                    "model_bytes":
                        data.get(
                            "model_bytes",
                            0,
                        ),

                    "download_bytes":
                        data.get(
                            "download_bytes",
                            0,
                        ),

                    "upload_bytes":
                        data.get(
                            "upload_bytes",
                            0,
                        ),

                    "total_transmitted_bytes":
                        data.get(
                            "total_transmitted_bytes",
                            0,
                        ),

                    "total_transmitted_mb":
                        data.get(
                            "total_transmitted_mb",
                            0.0,
                        ),

                    "jain_fairness":
                        data.get(
                            "jain_fairness",
                            0.0,
                        ),

                    "participation_std":
                        data.get(
                            "participation_std",
                            0.0,
                        ),

                    "min_participation":
                        data.get(
                            "min_participation",
                            0,
                        ),

                    "max_participation":
                        data.get(
                            "max_participation",
                            0,
                        ),

                    "converged":
                        data.get(
                            "converged",
                            False,
                        ),

                    "time_to_target_sec":
                        data.get(
                            "time_to_target_sec",
                            "",
                        ),
                }

                writer.writerow(row)