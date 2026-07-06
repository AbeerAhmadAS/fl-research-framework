#This file represents the server with FedFS + results logging.
#FedFS = Federating Fast and Slow.
#This file is used for research experiments.
#It keeps FedFS algorithm logic inside flwr.server.strategy.FedFS
#and adds experiment-specific logging and metrics here.
#
#This near-complete implementation follows the FedFS logic:
#1) deadline-based local training
#2) partial work return from clients
#3) work contribution ratio wk
#4) importance sampling using probability proportional to 1 - wk + epsilon
#5) alternating fast and slow timeouts

import csv
import os
import time
import platform
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from flwr.common import FitRes, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedFS


def jain_fairness(values: List[int]) -> float:
    arr = np.array(values, dtype=np.float64)
    if arr.sum() == 0:
        return 0.0
    return float((arr.sum() ** 2) / (len(arr) * np.sum(arr ** 2)))


class FedFSWithClientLogging(FedFS):
    def __init__(
        self,
        num_clients: int,
        log_dir: str = "results",
        target_accuracy: float = 0.70,
        r_fast: int = 1,
        r_slow: int = 1,
        delta_fast: float = 90.0,
        delta_slow: float = 180.0,
        epsilon: float = 0.05,
        *args,
        **kwargs,
    ):
        super().__init__(
            r_fast=r_fast,
            r_slow=r_slow,
            delta_fast=delta_fast,
            delta_slow=delta_slow,
            epsilon=epsilon,
            *args,
            **kwargs,
        )

        self.num_clients = num_clients
        self.log_dir = log_dir
        self.target_accuracy = target_accuracy

        os.makedirs(self.log_dir, exist_ok=True)

        self.round_start_time: Dict[int, float] = {}

        #Counts how many times each client participated.
        self.participation = defaultdict(int)

        self.global_start_time = time.time()
        self.first_target_round: Optional[int] = None
        self.first_target_time: Optional[float] = None

        self.round_csv = os.path.join(self.log_dir, "round_metrics.csv")
        self.hardware_csv = os.path.join(self.log_dir, "hardware_specs.csv")

        self._init_round_csv()
        self._write_hardware_specs()

    # Results file
    def _init_round_csv(self):
        with open(self.round_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "round",
                    "fedfs_round_type",
                    "fit_deadline_sec",
                    "selected_clients",
                    "num_selected_clients",
                    "avg_train_loss",
                    "avg_accuracy",
                    "avg_eval_loss",
                    "round_duration_sec",
                    "avg_fit_duration_sec",
                    "total_fit_duration_sec",
                    "avg_samples_per_second",
                    "avg_gpu_memory_mb",
                    "avg_work_ratio",
                    "total_actual_work",
                    "total_max_possible_work",
                    "num_partial_clients",
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
            )

    # Save hardware specifications
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
            "strategy": "FedFS",
            "r_fast": str(self.r_fast),
            "r_slow": str(self.r_slow),
            "delta_fast": str(self.delta_fast),
            "delta_slow": str(self.delta_slow),
            "epsilon": str(self.epsilon),
        }

        with open(self.hardware_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["key", "value"])
            for k, v in specs.items():
                writer.writerow([k, v])

    # This works before the start of each training round.
    def configure_fit(self, server_round, parameters, client_manager):
        # Record the start time of round
        self.round_start_time[server_round] = time.time()

        selected = super().configure_fit(
            server_round,
            parameters,
            client_manager,
        )

        selected_cids = [client.cid for client, _ in selected]

        # Update the participation counter
        for cid in selected_cids:
            self.participation[cid] += 1

        round_type = self._fedfs_round_type(server_round)
        fit_deadline_sec = self._fedfs_deadline(server_round)

        print(f"[ROUND {server_round}] Selected clients: {selected_cids}")
        print(
            f"[SERVER][FedFS][ROUND {server_round}] "
            f"type={round_type}, "
            f"deadline={fit_deadline_sec:.2f}s, "
            f"epsilon={self.epsilon:.4f}"
        )

        return selected

    # This works after clients have completed local training.
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round,
            results,
            failures,
        )

        #Calculating the time of the round
        round_duration = time.time() - self.round_start_time.get(
            server_round,
            time.time(),
        )
        

        selected_cids = [client.cid for client, _ in results]
        num_selected = len(selected_cids)

        fit_durations = [
            float(fit_res.metrics.get("fit_duration", 0.0))
            for _, fit_res in results
        ]

        train_losses = [
            float(fit_res.metrics.get("train_loss", 0.0))
            for _, fit_res in results
        ]

        samples_per_second = [
            float(fit_res.metrics.get("samples_per_second", 0.0))
            for _, fit_res in results
        ]

        gpu_memory = [
            float(fit_res.metrics.get("gpu_memory_mb", 0.0))
            for _, fit_res in results
        ]

        work_ratios = [
            float(fit_res.metrics.get("work_ratio", 0.0))
            for _, fit_res in results
        ]

        actual_work_values = [
            float(fit_res.metrics.get("num_examples", 0.0))
            for _, fit_res in results
        ]

        max_work_values = [
            float(fit_res.metrics.get("max_possible_examples", 0.0))
            for _, fit_res in results
        ]

        partial_flags = [
            bool(fit_res.metrics.get("stopped_by_deadline", False))
            for _, fit_res in results
        ]

        model_bytes = 0
        if results:
            model_bytes = int(results[0][1].metrics.get("model_bytes", 0))

        download_bytes = sum(
            int(fit_res.metrics.get("download_bytes", 0))
            for _, fit_res in results
        )

        upload_bytes = sum(
            int(fit_res.metrics.get("upload_bytes", 0))
            for _, fit_res in results
        ) 

        total_transmitted_bytes = download_bytes + upload_bytes
        total_transmitted_mb = total_transmitted_bytes / (1024 ** 2)
        participation_values = [
            self.participation.get(str(i), 0)
            for i in range(self.num_clients)
        ]

        # Ray/Flower sometimes uses large internal CIDs.
        # This fallback measures fairness over actually observed CIDs.
        if sum(participation_values) == 0:
            participation_values = list(self.participation.values())

        # fairness calculation
        fairness = jain_fairness(participation_values)

        round_type = self._fedfs_round_type(server_round)
        fit_deadline_sec = self._fedfs_deadline(server_round)

        # Save results in CSV
        row = {
            "round": server_round,
            "fedfs_round_type": round_type,
            "fit_deadline_sec": fit_deadline_sec,
            "selected_clients": "|".join(selected_cids),
            "num_selected_clients": num_selected,
            "avg_train_loss": float(np.mean(train_losses)) if train_losses else 0.0,
            "avg_accuracy": "",
            "avg_eval_loss": "",
            "round_duration_sec": float(round_duration),
            "avg_fit_duration_sec": float(np.mean(fit_durations)) if fit_durations else 0.0,
            "total_fit_duration_sec": float(np.sum(fit_durations)) if fit_durations else 0.0,
            "avg_samples_per_second": float(np.mean(samples_per_second)) if samples_per_second else 0.0,
            "avg_gpu_memory_mb": float(np.mean(gpu_memory)) if gpu_memory else 0.0,
            "avg_work_ratio": float(np.mean(work_ratios)) if work_ratios else 0.0,
            "total_actual_work": float(np.sum(actual_work_values)) if actual_work_values else 0.0,
            "total_max_possible_work": float(np.sum(max_work_values)) if max_work_values else 0.0,
            "num_partial_clients": int(np.sum(partial_flags)) if partial_flags else 0,
            "model_bytes": model_bytes,
            "download_bytes": download_bytes,
            "upload_bytes": upload_bytes,
            "total_transmitted_bytes": total_transmitted_bytes,
            "total_transmitted_mb": float(total_transmitted_mb),
            "jain_fairness": fairness,
            "participation_std": float(np.std(participation_values)) if participation_values else 0.0,
            "min_participation": int(np.min(participation_values)) if participation_values else 0,
            "max_participation": int(np.max(participation_values)) if participation_values else 0,
            "converged": False,
            "time_to_target_sec": "",
        }

        self._append_partial_row(row)

        print(
            f"[SERVER][COMMUNICATION][ROUND {server_round}] "
            f"download={download_bytes / (1024 ** 2):.2f} MB, "
            f"upload={upload_bytes / (1024 ** 2):.2f} MB, "
            f"total={total_transmitted_mb:.2f} MB "
            f"(≈ {total_transmitted_mb / 1024:.2f} GB)"
        )

        print(
            f"[SERVER][ROUND {server_round}] "
            f"loss={row['avg_train_loss']:.4f}, "
            f"round_time={round_duration:.2f}s, "
            f"comm={total_transmitted_mb:.2f}MB, "
            f"fairness={fairness:.4f}, "
            f"avg_work_ratio={row['avg_work_ratio']:.4f}, "
            f"partial_clients={row['num_partial_clients']}"
        )

        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(self, server_round, results, failures):
        loss, metrics = super().aggregate_evaluate(
            server_round,
            results,
            failures,
        )

        # Measuring Convergence Time
        accuracy = float(metrics.get("accuracy", 0.0)) if metrics else 0.0
        eval_loss = float(loss) if loss is not None else 0.0

        converged = accuracy >= self.target_accuracy

        if converged and self.first_target_round is None:
            self.first_target_round = server_round
            self.first_target_time = time.time() - self.global_start_time

        self._update_eval_metrics(
            server_round=server_round,
            accuracy=accuracy,
            eval_loss=eval_loss,
            converged=converged,
            time_to_target_sec=self.first_target_time,
        )

        print(
            f"[SERVER][CONVERGENCE][ROUND {server_round}] "
            f"target_accuracy={self.target_accuracy:.4f}, "
            f"accuracy={accuracy:.4f}, "
            f"converged={converged}, "
            f"time_to_target_sec={self.first_target_time}"
        )

        print(
            f"[SERVER][EVAL][ROUND {server_round}] "
            f"accuracy={accuracy:.4f}, "
            f"eval_loss={eval_loss:.4f}"
        )

        return loss, metrics

    def _append_partial_row(self, row: Dict):
        with open(self.round_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    row["round"],
                    row["fedfs_round_type"],
                    row["fit_deadline_sec"],
                    row["selected_clients"],
                    row["num_selected_clients"],
                    row["avg_train_loss"],
                    row["avg_accuracy"],
                    row["avg_eval_loss"],
                    row["round_duration_sec"],
                    row["avg_fit_duration_sec"],
                    row["total_fit_duration_sec"],
                    row["avg_samples_per_second"],
                    row["avg_gpu_memory_mb"],
                    row["avg_work_ratio"],
                    row["total_actual_work"],
                    row["total_max_possible_work"],
                    row["num_partial_clients"],
                    row["model_bytes"],
                    row["download_bytes"],
                    row["upload_bytes"],
                    row["total_transmitted_bytes"],
                    row["total_transmitted_mb"],
                    row["jain_fairness"],
                    row["participation_std"],
                    row["min_participation"],
                    row["max_participation"],
                    row["converged"],
                    row["time_to_target_sec"],
                ]
            )

    def _update_eval_metrics(
        self,
        server_round: int,
        accuracy: float,
        eval_loss: float,
        converged: bool,
        time_to_target_sec,
    ):
        rows = []

        with open(self.round_csv, "r", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            for row in reader:
                if int(row["round"]) == server_round:
                    row["avg_accuracy"] = accuracy
                    row["avg_eval_loss"] = eval_loss
                    row["converged"] = converged
                    row["time_to_target_sec"] = (
                        "" if time_to_target_sec is None else time_to_target_sec
                    )
                rows.append(row)

        with open(self.round_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
