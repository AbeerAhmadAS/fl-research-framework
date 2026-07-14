# FAVOR experiment runner.
#
# FAVOR_MODE=train:
# - runs multiple federated-learning episodes
# - uses one selected client per agent-training step
# - keeps one shared DDQN agent and replay memory across episodes
# - fits PCA only once, then reloads the same PCA checkpoint in later episodes
#
# FAVOR_MODE=eval:
# - loads the trained DDQN checkpoint
# - selects Top-K clients in each round
# - does not update the agent
# - requires and reuses the PCA checkpoint created during training

from __future__ import annotations

import os
import timeit
from logging import INFO
from pathlib import Path
from typing import Dict, List, Tuple

import flwr as fl
from flwr.common import (
    Scalar,
    ndarrays_to_parameters,
)
from flwr.common.logger import log
from flwr.server import Server
from flwr.server.client_manager import (
    SimpleClientManager,
)
from flwr.server.history import History

from client_factory2 import (
    FavorTaskConfig,
    prepare_favor_task,
)
from favor_with_logging import (
    FavorWithLogging,
)
from flwr.server.strategy.favor import (
    DoubleDQNAgent,
)


MODE = os.environ.get(
    "FAVOR_MODE",
    "train",
).strip().lower()

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

LOCAL_EPOCHS = int(
    os.environ.get(
        "LOCAL_EPOCHS",
        5,
    )
)

PCA_COMPONENTS = int(
    os.environ.get(
        "PCA_COMPONENTS",
        NUM_CLIENTS,
    )
)

TARGET_ACCURACY = float(
    os.environ.get(
        "TARGET_ACCURACY",
        0.99,
    )
)

EPISODES = int(
    os.environ.get(
        "EPISODES",
        20,
    )
)

MAX_AGENT_STEPS = int(
    os.environ.get(
        "MAX_AGENT_STEPS",
        100,
    )
)

EVAL_ROUNDS = int(
    os.environ.get(
        "EVAL_ROUNDS",
        100,
    )
)

BASE_SEED = int(
    os.environ.get(
        "BASE_SEED",
        1234,
    )
)

LOG_DIR = os.environ.get(
    "LOG_DIR",
    "results/favor_mnist",
)

AGENT_CHECKPOINT = os.environ.get(
    "FAVOR_CHECKPOINT",
    os.path.join(
        LOG_DIR,
        "favor_agent.pt",
    ),
)

PCA_CHECKPOINT = os.environ.get(
    "FAVOR_PCA_PATH",
    os.path.join(
        LOG_DIR,
        "favor_pca.pkl",
    ),
)

HIDDEN_DIM = int(
    os.environ.get(
        "DQN_HIDDEN_DIM",
        512,
    )
)

DQN_LR = float(
    os.environ.get(
        "DQN_LR",
        0.0001,
    )
)

GAMMA = float(
    os.environ.get(
        "DQN_GAMMA",
        0.99,
    )
)

REPLAY_CAPACITY = int(
    os.environ.get(
        "DQN_REPLAY_CAPACITY",
        10000,
    )
)

DQN_BATCH_SIZE = int(
    os.environ.get(
        "DQN_BATCH_SIZE",
        32,
    )
)

DQN_WARMUP = int(
    os.environ.get(
        "DQN_WARMUP",
        32,
    )
)

TARGET_UPDATE_INTERVAL = int(
    os.environ.get(
        "TARGET_UPDATE_INTERVAL",
        20,
    )
)

EPSILON_START = float(
    os.environ.get(
        "EPSILON_START",
        1.0,
    )
)

EPSILON_END = float(
    os.environ.get(
        "EPSILON_END",
        0.05,
    )
)

EPSILON_DECAY_STEPS = int(
    os.environ.get(
        "EPSILON_DECAY_STEPS",
        2000,
    )
)

CLIENT_NUM_CPUS = float(
    os.environ.get(
        "CLIENT_NUM_CPUS",
        1,
    )
)

CLIENT_NUM_GPUS = float(
    os.environ.get(
        "CLIENT_NUM_GPUS",
        0.1,
    )
)


class EarlyStoppingServer(Server):
    """Flower server that ends a FAVOR episode at target accuracy."""

    def fit(
        self,
        num_rounds: int,
        timeout,
    ):
        history = History()

        log(
            INFO,
            "[INIT]",
        )

        self.parameters = (
            self._get_initial_parameters(
                server_round=0,
                timeout=timeout,
            )
        )

        initial_eval = (
            self.strategy.evaluate(
                0,
                parameters=self.parameters,
            )
        )

        if initial_eval is not None:
            history.add_loss_centralized(
                server_round=0,
                loss=initial_eval[0],
            )

            history.add_metrics_centralized(
                server_round=0,
                metrics=initial_eval[1],
            )

        start_time = (
            timeit.default_timer()
        )

        for current_round in range(
            1,
            num_rounds + 1,
        ):
            log(
                INFO,
                "[ROUND %s]",
                current_round,
            )

            fit_result = self.fit_round(
                server_round=current_round,
                timeout=timeout,
            )

            if fit_result is not None:
                (
                    parameters_prime,
                    fit_metrics,
                    _,
                ) = fit_result

                if (
                    parameters_prime
                    is not None
                ):
                    self.parameters = (
                        parameters_prime
                    )

                history.add_metrics_distributed_fit(
                    server_round=current_round,
                    metrics=fit_metrics,
                )

            centralized = (
                self.strategy.evaluate(
                    current_round,
                    parameters=self.parameters,
                )
            )

            if centralized is not None:
                loss, metrics = (
                    centralized
                )

                history.add_loss_centralized(
                    server_round=current_round,
                    loss=loss,
                )

                history.add_metrics_centralized(
                    server_round=current_round,
                    metrics=metrics,
                )

            if getattr(
                self.strategy,
                "episode_done",
                False,
            ):
                log(
                    INFO,
                    "Target accuracy reached "
                    "in round %s",
                    current_round,
                )

                break

        return (
            history,
            timeit.default_timer()
            - start_time,
        )


def fit_config(
    server_round: int,
) -> Dict[str, Scalar]:
    return {
        "server_round": int(
            server_round
        ),
        "local_epochs": int(
            LOCAL_EPOCHS
        ),
    }


def aggregate_fit_metrics(
    results: List[
        Tuple[
            int,
            Dict[str, Scalar],
        ]
    ],
) -> Dict[str, Scalar]:
    if not results:
        return {}

    total = sum(
        num_examples
        for num_examples, _
        in results
    )

    weighted_loss = sum(
        num_examples
        * float(
            metrics.get(
                "train_loss",
                0.0,
            )
        )
        for num_examples, metrics
        in results
    ) / max(total, 1)

    return {
        "train_loss": float(
            weighted_loss
        )
    }


def build_agent(
) -> DoubleDQNAgent:
    state_dim = (
        NUM_CLIENTS + 1
    ) * PCA_COMPONENTS

    return DoubleDQNAgent(
        state_dim=state_dim,
        num_actions=NUM_CLIENTS,
        hidden_dim=HIDDEN_DIM,
        learning_rate=DQN_LR,
        gamma=GAMMA,
        replay_capacity=(
            REPLAY_CAPACITY
        ),
        batch_size=DQN_BATCH_SIZE,
        warmup_steps=DQN_WARMUP,
        target_update_interval=(
            TARGET_UPDATE_INTERVAL
        ),
        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        epsilon_decay_steps=(
            EPSILON_DECAY_STEPS
        ),
        seed=BASE_SEED,
        device=os.environ.get(
            "AGENT_DEVICE",
            "cpu",
        ),
    )


def run_one_simulation(
    agent: DoubleDQNAgent,
    task_config: FavorTaskConfig,
    mode: str,
    episode: int,
    rounds: int,
):
    (
        client_fn,
        evaluate_fn,
        initial_ndarrays,
    ) = prepare_favor_task(
        task_config
    )

    strategy = FavorWithLogging(
        log_dir=LOG_DIR,

        num_clients=NUM_CLIENTS,

        selected_clients=(
            SELECTED_CLIENTS
        ),

        mode=mode,

        pca_components=(
            PCA_COMPONENTS
        ),

        target_accuracy=(
            TARGET_ACCURACY
        ),

        agent=agent,

        agent_checkpoint=(
            AGENT_CHECKPOINT
            if mode == "train"
            else None
        ),

        pca_checkpoint=(
            PCA_CHECKPOINT
        ),

        episode=episode,

        seed=BASE_SEED,

        initial_parameters=(
            ndarrays_to_parameters(
                initial_ndarrays
            )
        ),

        # FAVOR performs its own client selection.
        fraction_fit=1.0,

        fraction_evaluate=0.0,

        min_fit_clients=1,

        min_evaluate_clients=0,

        min_available_clients=(
            NUM_CLIENTS
        ),

        evaluate_fn=evaluate_fn,

        on_fit_config_fn=fit_config,

        fit_metrics_aggregation_fn=(
            aggregate_fit_metrics
        ),
    )

    client_manager = (
        SimpleClientManager()
    )

    server = EarlyStoppingServer(
        client_manager=client_manager,
        strategy=strategy,
    )

    return fl.simulation.start_simulation(
        client_fn=client_fn,

        num_clients=NUM_CLIENTS,

        server=server,

        config=fl.server.ServerConfig(
            num_rounds=rounds
        ),

        client_resources={
            "num_cpus": (
                CLIENT_NUM_CPUS
            ),
            "num_gpus": (
                CLIENT_NUM_GPUS
            ),
        },
    )


def validate_configuration(
) -> None:
    if MODE not in {
        "train",
        "eval",
    }:
        raise ValueError(
            "FAVOR_MODE must be train or eval."
        )

    if NUM_CLIENTS <= 0:
        raise ValueError(
            "NUM_CLIENTS must be positive."
        )

    if not (
        1
        <= SELECTED_CLIENTS
        <= NUM_CLIENTS
    ):
        raise ValueError(
            "SELECTED_CLIENTS must be between "
            "1 and NUM_CLIENTS."
        )

    if PCA_COMPONENTS <= 0:
        raise ValueError(
            "PCA_COMPONENTS must be positive."
        )

    if (
        PCA_COMPONENTS
        > NUM_CLIENTS
    ):
        raise ValueError(
            "PCA_COMPONENTS cannot exceed "
            "NUM_CLIENTS."
        )

    if MODE == "eval":
        if not os.path.exists(
            AGENT_CHECKPOINT
        ):
            raise FileNotFoundError(
                "Trained FAVOR agent not found: "
                f"{AGENT_CHECKPOINT}"
            )

        if not os.path.exists(
            PCA_CHECKPOINT
        ):
            raise FileNotFoundError(
                "Trained FAVOR PCA checkpoint "
                f"not found: {PCA_CHECKPOINT}"
            )


def main(
) -> None:
    validate_configuration()

    Path(LOG_DIR).mkdir(
        parents=True,
        exist_ok=True,
    )

    task_config = (
        FavorTaskConfig()
    )

    print(
        "========== FAVOR Configuration =========="
    )

    print(
        f"MODE               = {MODE}"
    )

    print(
        f"NUM_CLIENTS        = {NUM_CLIENTS}"
    )

    print(
        f"SELECTED_CLIENTS   = {SELECTED_CLIENTS}"
    )

    print(
        f"LOCAL_EPOCHS       = {LOCAL_EPOCHS}"
    )

    print(
        f"PCA_COMPONENTS     = {PCA_COMPONENTS}"
    )

    print(
        f"TARGET_ACCURACY    = {TARGET_ACCURACY}"
    )

    print(
        f"SAMPLES_PER_CLIENT = "
        f"{task_config.samples_per_client}"
    )

    print(
        f"DOMINANT_FRACTION  = "
        f"{task_config.dominant_fraction}"
    )

    print(
        f"LOG_DIR            = {LOG_DIR}"
    )

    print(
        f"AGENT_CHECKPOINT   = "
        f"{AGENT_CHECKPOINT}"
    )

    print(
        f"PCA_CHECKPOINT     = "
        f"{PCA_CHECKPOINT}"
    )

    print(
        "========================================="
    )

    agent = build_agent()

    if MODE == "train":
        if os.path.exists(
            AGENT_CHECKPOINT
        ):
            print(
                "Loading existing training "
                f"checkpoint: "
                f"{AGENT_CHECKPOINT}"
            )

            agent.load(
                AGENT_CHECKPOINT,
                load_replay=True,
            )

        # One extra Flower round is used
        # for client profiling.
        rounds_per_episode = (
            MAX_AGENT_STEPS + 1
        )

        for episode in range(
            1,
            EPISODES + 1,
        ):
            print(
                "\n========== "
                f"TRAIN EPISODE "
                f"{episode}/{EPISODES} "
                "=========="
            )

            run_one_simulation(
                agent=agent,
                task_config=task_config,
                mode="train",
                episode=episode,
                rounds=rounds_per_episode,
            )

            agent.save(
                AGENT_CHECKPOINT
            )

    else:
        agent.load(
            AGENT_CHECKPOINT,
            load_replay=False,
        )

        agent.online.eval()
        agent.target.eval()

        # One extra Flower round is used
        # for client profiling.
        run_one_simulation(
            agent=agent,
            task_config=task_config,
            mode="eval",
            episode=1,
            rounds=EVAL_ROUNDS + 1,
        )


if __name__ == "__main__":
    main()
