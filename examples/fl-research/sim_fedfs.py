import os
from typing import Dict, List, Tuple

import flwr as fl
from flwr.common import Scalar

#File linking:
    #Retrieves clients from client_factory1.py
    #Retrieves strategy from fedfs_with_logging.py
from client_factory1 import client_fn
from fedfs_with_logging import FedFSWithClientLogging

# Reading the experiment settings
NUM_CLIENTS = int(os.environ.get("NUM_CLIENTS", 20))
SELECTED_CLIENTS = int(os.environ.get("SELECTED_CLIENTS", 2))
ROUNDS = int(os.environ.get("ROUNDS", 3))
LOCAL_EPOCHS = int(os.environ.get("LOCAL_EPOCHS", 1))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 16))
TARGET_ACCURACY = float(os.environ.get("TARGET_ACCURACY", 0.70))
LOG_DIR = os.environ.get("LOG_DIR", "results/fedfs_vgg16_cinic10")

#FedFS settings
R_FAST = int(os.environ.get("R_FAST", 1))
R_SLOW = int(os.environ.get("R_SLOW", 1))

# New FedFS settings
DELTA_FAST = float(os.environ.get("DELTA_FAST", 90.0))
DELTA_SLOW = float(os.environ.get("DELTA_SLOW", 180.0))
EPSILON = float(os.environ.get("EPSILON", 0.05))

# This sends settings from the server to each client.
def fit_config(server_round: int):
    return {
        "server_round": server_round,
        "local_epochs": LOCAL_EPOCHS,
    }


def aggregate_train_loss(results: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
    if not results:
        return {}

    total = sum(n for n, _ in results)
    loss = sum(n * float(m.get("train_loss", 0.0)) for n, m in results) / max(total, 1)

    return {"train_loss": loss}


def aggregate_eval_metrics(results: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
    if not results:
        return {}

    total = sum(n for n, _ in results)
    acc = sum(n * float(m.get("accuracy", 0.0)) for n, m in results) / max(total, 1)
    eval_loss = sum(n * float(m.get("eval_loss", 0.0)) for n, m in results) / max(total, 1)
    eval_duration = sum(float(m.get("eval_duration", 0.0)) for _, m in results)

    return {
        "accuracy": acc,
        "eval_loss": eval_loss,
        "eval_duration": eval_duration,
    }


def main():
    print("========== Experiment Config ==========")
    print(f"STRATEGY         = FedFS")
    print(f"NUM_CLIENTS      = {NUM_CLIENTS}")
    print(f"SELECTED_CLIENTS = {SELECTED_CLIENTS}")
    print(f"ROUNDS           = {ROUNDS}")
    print(f"LOCAL_EPOCHS     = {LOCAL_EPOCHS}")
    print(f"BATCH_SIZE       = {BATCH_SIZE}")
    print(f"TARGET_ACCURACY  = {TARGET_ACCURACY}")
    print(f"R_FAST           = {R_FAST}")
    print(f"R_SLOW           = {R_SLOW}")

    # New FedFS parameters
    print(f"DELTA_FAST       = {DELTA_FAST}")
    print(f"DELTA_SLOW       = {DELTA_SLOW}")
    print(f"EPSILON          = {EPSILON}")

    print(f"LOG_DIR          = {LOG_DIR}")
    print("=======================================")

    strategy = FedFSWithClientLogging(
        num_clients=NUM_CLIENTS,
        log_dir=LOG_DIR,
        target_accuracy=TARGET_ACCURACY,

        r_fast=R_FAST,
        r_slow=R_SLOW,

        # New FedFS parameters
        delta_fast=DELTA_FAST,
        delta_slow=DELTA_SLOW,
        epsilon=EPSILON,

        fraction_fit=SELECTED_CLIENTS / NUM_CLIENTS,
        fraction_evaluate=SELECTED_CLIENTS / NUM_CLIENTS,
        min_fit_clients=SELECTED_CLIENTS,
        min_evaluate_clients=SELECTED_CLIENTS,
        min_available_clients=NUM_CLIENTS,

        on_fit_config_fn=fit_config,
        fit_metrics_aggregation_fn=aggregate_train_loss,
        evaluate_metrics_aggregation_fn=aggregate_eval_metrics,
    )

    # running the simulation
    # starting point of Flower
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=ROUNDS),
        strategy=strategy,

        # Sequential execution to reduce CUDA OOM
        client_resources={
            "num_cpus": 4,
            "num_gpus": 1.0,
        },
    )


if __name__ == "__main__":
    main()
