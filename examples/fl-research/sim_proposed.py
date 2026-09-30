import os
from typing import Dict, List, Tuple

import flwr as fl
from flwr.common import Scalar

from client_proposed import client_fn
from flwr.server.strategy.proposed import ProposedStrategy


# ============================================================
# Experiment configuration
# ============================================================

NUM_CLIENTS = int(
    os.environ.get(
        "NUM_CLIENTS",
        "20",
    )
)

# Final number of clients that train in each round
SELECTED_CLIENTS = int(
    os.environ.get(
        "SELECTED_CLIENTS",
        "5",
    )
)

# Number of clients retained after Combined-Score ranking
SHORTLIST_SIZE = int(
    os.environ.get(
        "SHORTLIST_SIZE",
        "10",
    )
)

ROUNDS = int(
    os.environ.get(
        "ROUNDS",
        "3",
    )
)

LOCAL_EPOCHS = int(
    os.environ.get(
        "LOCAL_EPOCHS",
        "1",
    )
)

BATCH_SIZE = int(
    os.environ.get(
        "BATCH_SIZE",
        "16",
    )
)

TARGET_ACCURACY = float(
    os.environ.get(
        "TARGET_ACCURACY",
        "0.70",
    )
)


# ============================================================
# Proposed selection configuration
# ============================================================

# score:
#   All profiled and available clients enter scoring.
#
# eligibility_score:
#   Only clients passing the hard eligibility filter
#   enter scoring.
SELECTION_MODE = os.environ.get(
    "SELECTION_MODE",
    "score",
)

# combined_score:
#   Select the highest-scoring clients from the shortlist.
#
# fedavg:
#   Randomly sample the final clients from the shortlist.
FINAL_SELECTOR = os.environ.get(
    "FINAL_SELECTOR",
    "combined_score",
)

# Combined Score =
#   capability_weight * System Capability
#   +
#   (1 - capability_weight) * Novelty
CAPABILITY_WEIGHT = float(
    os.environ.get(
        "CAPABILITY_WEIGHT",
        "0.5",
    )
)

SELECTION_SEED = int(
    os.environ.get(
        "SELECTION_SEED",
        "1234",
    )
)


# ============================================================
# Fit configuration
# ============================================================

def fit_config(server_round: int):

    return {
        "server_round": server_round,
        "local_epochs": LOCAL_EPOCHS,
    }


# ============================================================
# Training metrics aggregation
# ============================================================

def aggregate_train_loss(
    results: List[Tuple[int, Dict[str, Scalar]]]
) -> Dict[str, Scalar]:

    if not results:
        return {}

    total = sum(
        n for n, _ in results
    )

    train_loss = sum(
        n * float(
            metrics.get(
                "train_loss",
                0.0,
            )
        )
        for n, metrics in results
    ) / max(total, 1)

    return {
        "train_loss": train_loss,
    }


# ============================================================
# Evaluation metrics aggregation
# ============================================================

def aggregate_eval_metrics(
    results: List[Tuple[int, Dict[str, Scalar]]]
) -> Dict[str, Scalar]:

    if not results:
        return {}

    total = sum(
        n for n, _ in results
    )

    accuracy = sum(
        n * float(
            metrics.get(
                "accuracy",
                0.0,
            )
        )
        for n, metrics in results
    ) / max(total, 1)

    eval_loss = sum(
        n * float(
            metrics.get(
                "eval_loss",
                0.0,
            )
        )
        for n, metrics in results
    ) / max(total, 1)

    eval_duration = sum(
        float(
            metrics.get(
                "eval_duration",
                0.0,
            )
        )
        for _, metrics in results
    )

    return {
        "accuracy": accuracy,
        "eval_loss": eval_loss,
        "eval_duration": eval_duration,
    }


# ============================================================
# Proposed strategy
# ============================================================

strategy = ProposedStrategy(

    # Final number of training clients
    selected_clients=SELECTED_CLIENTS,

    # Combined-Score shortlist size
    shortlist_size=SHORTLIST_SIZE,

    # Candidate-pool construction
    selection_mode=SELECTION_MODE,

    # Final selector
    final_selector=FINAL_SELECTOR,

    # System Capability / Novelty trade-off
    capability_weight=CAPABILITY_WEIGHT,

    # Reproducibility
    selection_seed=SELECTION_SEED,

    # ProposedStrategy performs its own training-client selection
    fraction_fit=1.0,

    min_fit_clients=SELECTED_CLIENTS,

    min_available_clients=NUM_CLIENTS,

    # --------------------------------------------------------
    # Evaluation
    # Same evaluation configuration used in previous
    # baseline experiments.
    # --------------------------------------------------------

    fraction_evaluate=(
        SELECTED_CLIENTS
        / NUM_CLIENTS
    ),

    min_evaluate_clients=SELECTED_CLIENTS,

    # --------------------------------------------------------
    # Configuration and metric aggregation
    # --------------------------------------------------------

    on_fit_config_fn=fit_config,

    fit_metrics_aggregation_fn=aggregate_train_loss,

    evaluate_metrics_aggregation_fn=aggregate_eval_metrics,
)


# ============================================================
# Run simulation
# ============================================================

if __name__ == "__main__":

    print(
        "\n"
        "========================================\n"
        "PROPOSED CLIENT SELECTION EXPERIMENT\n"
        "========================================"
    )

    print(
        f"Number of clients: "
        f"{NUM_CLIENTS}"
    )

    print(
        f"Shortlist size: "
        f"{SHORTLIST_SIZE}"
    )

    print(
        f"Final selected clients: "
        f"{SELECTED_CLIENTS}"
    )

    print(
        f"Rounds: "
        f"{ROUNDS}"
    )

    print(
        f"Local epochs: "
        f"{LOCAL_EPOCHS}"
    )

    print(
        f"Batch size: "
        f"{BATCH_SIZE}"
    )

    print(
        f"Target accuracy: "
        f"{TARGET_ACCURACY}"
    )

    print(
        f"Selection mode: "
        f"{SELECTION_MODE}"
    )

    print(
        f"Final selector: "
        f"{FINAL_SELECTOR}"
    )

    print(
        f"Capability weight: "
        f"{CAPABILITY_WEIGHT:.2f}"
    )

    print(
        f"Novelty weight: "
        f"{1.0 - CAPABILITY_WEIGHT:.2f}"
    )

    print(
        f"Selection seed: "
        f"{SELECTION_SEED}"
    )

    print(
        "Execution mode: Sequential"
    )

    print(
        "Client resources: "
        "4 CPUs, 1 GPU"
    )

    print(
        "========================================\n"
    )

    fl.simulation.start_simulation(

        client_fn=client_fn,

        num_clients=NUM_CLIENTS,

        config=fl.server.ServerConfig(
            num_rounds=ROUNDS
        ),

        strategy=strategy,

        # Sequential execution to reduce CUDA OOM
        client_resources={
            "num_cpus": 4,
            "num_gpus": 1.0,
        },
    )