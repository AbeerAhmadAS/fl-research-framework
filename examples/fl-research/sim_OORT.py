import os
from typing import Dict, List, Tuple

import flwr as fl
from flwr.common import Scalar

# File linking:
# Retrieves clients from client_factory1.py
# Retrieves strategy from OORT_with_logging.py
from client_factory1 import client_fn
from OORT_with_logging import OORTWithClientLogging


NUM_CLIENTS = int(os.environ.get("NUM_CLIENTS", 20))
SELECTED_CLIENTS = int(os.environ.get("SELECTED_CLIENTS", 2))
ROUNDS = int(os.environ.get("ROUNDS", 3))
LOCAL_EPOCHS = int(os.environ.get("LOCAL_EPOCHS", 1))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 16))
TARGET_ACCURACY = float(os.environ.get("TARGET_ACCURACY", 0.70))
LOG_DIR = os.environ.get("LOG_DIR", "results/oort_vgg16_cinic10")

# Oort settings
OORT_EXPLORATION_FACTOR = float(os.environ.get("OORT_EXPLORATION_FACTOR", 0.90))
OORT_EXPLORATION_DECAY = float(os.environ.get("OORT_EXPLORATION_DECAY", 0.98))
OORT_MIN_EXPLORATION_FACTOR = float(os.environ.get("OORT_MIN_EXPLORATION_FACTOR", 0.20))
OORT_PACER_WINDOW = int(os.environ.get("OORT_PACER_WINDOW", 20))
OORT_PACER_STEP = float(os.environ.get("OORT_PACER_STEP", 60.0))
OORT_ALPHA = float(os.environ.get("OORT_ALPHA", 2.0))
OORT_CUTOFF = float(os.environ.get("OORT_CUTOFF", 0.95))
OORT_MAX_SELECTIONS = int(os.environ.get("OORT_MAX_SELECTIONS", 10))
OORT_SEED = int(os.environ.get("OORT_SEED", 1234))


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
    print(f"STRATEGY         = Oort")
    print(f"NUM_CLIENTS      = {NUM_CLIENTS}")
    print(f"SELECTED_CLIENTS = {SELECTED_CLIENTS}")
    print(f"ROUNDS           = {ROUNDS}")
    print(f"LOCAL_EPOCHS     = {LOCAL_EPOCHS}")
    print(f"BATCH_SIZE       = {BATCH_SIZE}")
    print(f"TARGET_ACCURACY  = {TARGET_ACCURACY}")
    print(f"LOG_DIR          = {LOG_DIR}")
    print("---------- Oort Settings --------------")
    print(f"OORT_EXPLORATION_FACTOR     = {OORT_EXPLORATION_FACTOR}")
    print(f"OORT_EXPLORATION_DECAY      = {OORT_EXPLORATION_DECAY}")
    print(f"OORT_MIN_EXPLORATION_FACTOR = {OORT_MIN_EXPLORATION_FACTOR}")
    print(f"OORT_PACER_WINDOW           = {OORT_PACER_WINDOW}")
    print(f"OORT_PACER_STEP             = {OORT_PACER_STEP}")
    print(f"OORT_ALPHA                  = {OORT_ALPHA}")
    print(f"OORT_CUTOFF                 = {OORT_CUTOFF}")
    print(f"OORT_MAX_SELECTIONS         = {OORT_MAX_SELECTIONS}")
    print(f"OORT_SEED                   = {OORT_SEED}")
    print("=======================================")

    strategy = OORTWithClientLogging(
        num_clients=NUM_CLIENTS,
        log_dir=LOG_DIR,
        target_accuracy=TARGET_ACCURACY,

        exploration_factor=OORT_EXPLORATION_FACTOR,
        exploration_decay=OORT_EXPLORATION_DECAY,
        min_exploration_factor=OORT_MIN_EXPLORATION_FACTOR,
        pacer_window=OORT_PACER_WINDOW,
        pacer_step=OORT_PACER_STEP,
        straggler_penalty_alpha=OORT_ALPHA,
        cutoff_percentage=OORT_CUTOFF,
        max_selection_per_client=OORT_MAX_SELECTIONS,
        seed=OORT_SEED,

        fraction_fit=SELECTED_CLIENTS / NUM_CLIENTS,
        fraction_evaluate=SELECTED_CLIENTS / NUM_CLIENTS,
        min_fit_clients=SELECTED_CLIENTS,
        min_evaluate_clients=SELECTED_CLIENTS,
        min_available_clients=NUM_CLIENTS,

        on_fit_config_fn=fit_config,
        fit_metrics_aggregation_fn=aggregate_train_loss,
        evaluate_metrics_aggregation_fn=aggregate_eval_metrics,
    )

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=ROUNDS),
        strategy=strategy,
        client_resources={
            "num_cpus": 4,
            "num_gpus": 1.0,
        },
    )


if __name__ == "__main__":
    main()
