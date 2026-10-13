import os
from typing import Dict, List, Tuple

import flwr as fl
from flwr.common import Scalar

from client_common import client_fn
from fedavg_with_logging import FedAvgWithLogging


# ============================================================
# Experiment configuration
# ============================================================

NUM_CLIENTS = int(
    os.environ.get(
        "NUM_CLIENTS",
        20,
    )
)

SELECTED_CLIENTS = int(
    os.environ.get(
        "SELECTED_CLIENTS",
        5,
    )
)

ROUNDS = int(
    os.environ.get(
        "ROUNDS",
        3,
    )
)

LOCAL_EPOCHS = int(
    os.environ.get(
        "LOCAL_EPOCHS",
        1,
    )
)

BATCH_SIZE = int(
    os.environ.get(
        "BATCH_SIZE",
        16,
    )
)

TARGET_ACCURACY = float(
    os.environ.get(
        "TARGET_ACCURACY",
        0.70,
    )
)

LOG_DIR = os.environ.get(
    "LOG_DIR",
    "results/fedavg_20c_5s_iid_3r",
)


# ============================================================
# Client fit configuration
# ============================================================

def fit_config(
    server_round: int,
):

    return {
        "server_round":
            server_round,

        "local_epochs":
            LOCAL_EPOCHS,
    }


# ============================================================
# Fit metrics aggregation
# ============================================================

def aggregate_fit_metrics(
    results: List[
        Tuple[
            int,
            Dict[str, Scalar],
        ]
    ]
) -> Dict[str, Scalar]:

    if not results:
        return {}

    total_examples = sum(
        n
        for n, _ in results
    )

    train_loss = (
        sum(
            n * float(
                metrics.get(
                    "train_loss",
                    0.0,
                )
            )
            for n, metrics in results
        )
        / max(total_examples, 1)
    )

    fit_durations = [
        float(
            metrics.get(
                "fit_duration",
                0.0,
            )
        )
        for _, metrics in results
    ]

    samples_per_second = [
        float(
            metrics.get(
                "samples_per_second",
                0.0,
            )
        )
        for _, metrics in results
    ]

    gpu_memory = [
        float(
            metrics.get(
                "gpu_memory_mb",
                0.0,
            )
        )
        for _, metrics in results
    ]

    model_bytes = 0

    if results:
        model_bytes = int(
            results[0][1].get(
                "model_bytes",
                0,
            )
        )

    num_successful = len(results)

    download_bytes = (
        model_bytes
        * num_successful
    )

    upload_bytes = (
        model_bytes
        * num_successful
    )

    total_transmitted_bytes = (
        download_bytes
        + upload_bytes
    )

    return {
        "train_loss":
            float(train_loss),

        "avg_fit_duration_sec":
            float(
                sum(fit_durations)
                / max(
                    len(fit_durations),
                    1,
                )
            ),

        "total_fit_duration_sec":
            float(
                sum(fit_durations)
            ),

        "avg_samples_per_second":
            float(
                sum(samples_per_second)
                / max(
                    len(samples_per_second),
                    1,
                )
            ),

        "avg_gpu_memory_mb":
            float(
                sum(gpu_memory)
                / max(
                    len(gpu_memory),
                    1,
                )
            ),

        "model_bytes":
            model_bytes,

        "download_bytes":
            download_bytes,

        "upload_bytes":
            upload_bytes,

        "total_transmitted_bytes":
            total_transmitted_bytes,

        "total_transmitted_mb":
            float(
                total_transmitted_bytes
                / (1024 ** 2)
            ),

        "num_successful_clients":
            num_successful,
    }


# ============================================================
# Evaluation metrics aggregation
# ============================================================

def aggregate_eval_metrics(
    results: List[
        Tuple[
            int,
            Dict[str, Scalar],
        ]
    ]
) -> Dict[str, Scalar]:

    if not results:
        return {}

    total_examples = sum(
        n
        for n, _ in results
    )

    accuracy = (
        sum(
            n * float(
                metrics.get(
                    "accuracy",
                    0.0,
                )
            )
            for n, metrics in results
        )
        / max(total_examples, 1)
    )

    eval_loss = (
        sum(
            n * float(
                metrics.get(
                    "eval_loss",
                    0.0,
                )
            )
            for n, metrics in results
        )
        / max(total_examples, 1)
    )

    eval_durations = [
        float(
            metrics.get(
                "eval_duration",
                0.0,
            )
        )
        for _, metrics in results
    ]

    return {
        "accuracy":
            float(accuracy),

        "eval_loss":
            float(eval_loss),

        "avg_eval_duration_sec":
            float(
                sum(eval_durations)
                / max(
                    len(eval_durations),
                    1,
                )
            ),

        "num_evaluated_clients":
            len(results),
    }


# ============================================================
# Main
# ============================================================

def main():

    print(
        "========== FedAvg Experiment Config =========="
    )

    print(
        f"NUM_CLIENTS      = {NUM_CLIENTS}"
    )

    print(
        f"SELECTED_CLIENTS = {SELECTED_CLIENTS}"
    )

    print(
        f"ROUNDS           = {ROUNDS}"
    )

    print(
        f"LOCAL_EPOCHS     = {LOCAL_EPOCHS}"
    )

    print(
        f"BATCH_SIZE       = {BATCH_SIZE}"
    )

    print(
        f"TARGET_ACCURACY  = {TARGET_ACCURACY}"
    )

    print(
        f"LOG_DIR          = {LOG_DIR}"
    )

    print(
        "=============================================="
    )

    strategy = FedAvgWithLogging(
        num_clients=NUM_CLIENTS,

        log_dir=LOG_DIR,

        target_accuracy=
            TARGET_ACCURACY,

        fraction_fit=
            SELECTED_CLIENTS
            / NUM_CLIENTS,

        min_fit_clients=
            SELECTED_CLIENTS,

        min_available_clients=
            NUM_CLIENTS,

        fraction_evaluate=
            SELECTED_CLIENTS
            / NUM_CLIENTS,

        min_evaluate_clients=
            SELECTED_CLIENTS,

        on_fit_config_fn=
            fit_config,

        fit_metrics_aggregation_fn=
            aggregate_fit_metrics,

        evaluate_metrics_aggregation_fn=
            aggregate_eval_metrics,
    )

    fl.simulation.start_simulation(
        client_fn=client_fn,

        num_clients=
            NUM_CLIENTS,

        config=
            fl.server.ServerConfig(
                num_rounds=ROUNDS
            ),

        strategy=
            strategy,

        client_resources={
            "num_cpus": 4,
            "num_gpus": 1.0,
        },
    )


if __name__ == "__main__":
    main()