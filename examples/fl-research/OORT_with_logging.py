# This file represents Oort + results logging.
# It inherits the core OORT strategy and adds only:
# - CSV logging
# - hardware_specs.csv
# - round_metrics.csv
# - communication metrics
# - fairness metrics
# - convergence/time-to-target tracking

import csv
import os
import time
import platform
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from flwr.common import FitRes, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy

from flwr.server.strategy.OORT import OORT


def jain_fairness(values: List[int]) -> float:
    arr = np.array(values, dtype=np.float64)
    if arr.sum() == 0:
        return 0.0
    return float((arr.sum() ** 2) / (len(arr) * np.sum(arr ** 2)))


class OORTWithClientLogging(OORT):
    def __init__(
        self,
        num_clients: int,
        log_dir: str = "results",
        target_accuracy: float = 0.70,
        *args,
        **kwargs,
    ):
        super().__init__(
            num_clients=num_clients,
            *args,
            **kwargs,
        )

        self.log_dir = log_dir
        self.target_accuracy = target_accuracy

        os.makedirs(self.log_dir, exist_ok=True)

        self.round_start_time: Dict[int, float] = {}

        self.global_start_time = time.time()
        self.first_target_round: Optional[int] = None
        self.first_target_time: Optional[float] = None

        self.round_csv = os.path.join(self.log_dir, "round_metrics.csv")
        self.hardware_csv = os.path.join(self.log_dir, "hardware_specs.csv")

        self._init_round_csv()
        self._write_hardware_specs()

    def _init_round_csv(self):
        with open(self.round_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "round",
                    "strategy",
                    "selected_clients",
                    "num_selected_clients",
                    "num_exploit_clients",
                    "num_explore_clients",
                    "oort_exploration_factor",
                    "oort_preferred_duration_T",
                    "avg_oort_utility",
                    "total_oort_stat_utility",
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
            )

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
            "strategy": "Oort",
            "exploration_factor_initial": str(self.current_round_exploration_factor),
            "exploration_decay": str(self.exploration_decay),
            "min_exploration_factor": str(self.min_exploration_factor),
            "pacer_window": str(self.pacer_window),
            "pacer_step": str(self.pacer_step),
            "straggler_penalty_alpha": str(self.straggler_penalty_alpha),
            "cutoff_percentage": str(self.cutoff_percentage),
            "max_selection_per_client": str(self.max_selection_per_client),
        }

        with open(self.hardware_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["key", "value"])
            for k, v in specs.items():
                writer.writerow([k, v])

    def configure_fit(self, server_round, parameters, client_manager):
        self.round_start_time[server_round] = time.time()

        selected = super().configure_fit(
            server_round,
            parameters,
            client_manager,
        )

        selected_cids = [client.cid for client, _ in selected]

        print(f"[ROUND {server_round}] Selected clients: {selected_cids}")
        print(
            f"[SERVER][OORT][ROUND {server_round}] "
            f"exploit={self.current_num_exploit}, "
            f"explore={self.current_num_explore}, "
            f"epsilon={self.current_round_exploration_factor:.4f}, "
            f"T={self.preferred_duration_T:.2f}s"
        )

        return selected

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

        total_oort_stat_utility = 0.0

        for _, fit_res in results:
            train_loss = float(fit_res.metrics.get("train_loss", 0.0))
            num_examples = int(fit_res.metrics.get("num_examples", 0))

            stat_utility = float(
                fit_res.metrics.get(
                    "oort_stat_utility",
                    num_examples * max(train_loss, 0.0),
                )
            )
            total_oort_stat_utility += stat_utility

        oort_utilities = [
            float(self.client_oort_utility.get(cid, 0.0))
            for cid in selected_cids
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

        fairness = jain_fairness(participation_values)

        row = {
            "round": server_round,
            "strategy": "Oort",
            "selected_clients": "|".join(selected_cids),
            "num_selected_clients": num_selected,
            "num_exploit_clients": int(self.current_num_exploit),
            "num_explore_clients": int(self.current_num_explore),
            "oort_exploration_factor": float(self.current_round_exploration_factor),
            "oort_preferred_duration_T": float(self.preferred_duration_T),
            "avg_oort_utility": float(np.mean(oort_utilities)) if oort_utilities else 0.0,
            "total_oort_stat_utility": float(total_oort_stat_utility),
            "avg_train_loss": float(np.mean(train_losses)) if train_losses else 0.0,
            "avg_accuracy": "",
            "avg_eval_loss": "",
            "round_duration_sec": float(round_duration),
            "avg_fit_duration_sec": float(np.mean(fit_durations)) if fit_durations else 0.0,
            "total_fit_duration_sec": float(np.sum(fit_durations)) if fit_durations else 0.0,
            "avg_samples_per_second": float(np.mean(samples_per_second)) if samples_per_second else 0.0,
            "avg_gpu_memory_mb": float(np.mean(gpu_memory)) if gpu_memory else 0.0,
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
            f"[SERVER][OORT][ROUND {server_round}] "
            f"loss={row['avg_train_loss']:.4f}, "
            f"round_time={round_duration:.2f}s, "
            f"comm={total_transmitted_mb:.2f}MB, "
            f"fairness={fairness:.4f}, "
            f"stat_utility={total_oort_stat_utility:.4f}, "
            f"avg_oort_utility={row['avg_oort_utility']:.4f}"
        )

        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(self, server_round, results, failures):
        loss, metrics = super().aggregate_evaluate(
            server_round,
            results,
            failures,
        )

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
                    row["strategy"],
                    row["selected_clients"],
                    row["num_selected_clients"],
                    row["num_exploit_clients"],
                    row["num_explore_clients"],
                    row["oort_exploration_factor"],
                    row["oort_preferred_duration_T"],
                    row["avg_oort_utility"],
                    row["total_oort_stat_utility"],
                    row["avg_train_loss"],
                    row["avg_accuracy"],
                    row["avg_eval_loss"],
                    row["round_duration_sec"],
                    row["avg_fit_duration_sec"],
                    row["total_fit_duration_sec"],
                    row["avg_samples_per_second"],
                    row["avg_gpu_memory_mb"],
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
