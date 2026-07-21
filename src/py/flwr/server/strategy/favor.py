# FAVOR core strategy for Flower.
#
# This file contains algorithm logic only:
# - full local/global model-weight state
# - PCA compression fitted once from the first profiling models
# - reuse of the same PCA basis across all episodes and evaluation
# - stable logical client identities (logical_cid)
# - Double DQN agent
# - single-client action during agent training
# - Top-K client selection during inference
# - FAVOR reward
# - FedAvg model aggregation
#

from __future__ import annotations

import random
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import torch
from sklearn.decomposition import PCA
from torch import nn
from torch.optim import Adam

from flwr.common import (
    FitIns,
    FitRes,
    NDArrays,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from .fedavg import FedAvg


def flatten_ndarrays(arrays: NDArrays) -> np.ndarray:
    """Flatten all model tensors into one float32 vector."""
    if not arrays:
        raise ValueError("Model parameters cannot be empty.")

    return np.concatenate(
        [
            np.asarray(array, dtype=np.float32).reshape(-1)
            for array in arrays
        ]
    ).astype(np.float32, copy=False)


def favor_reward(
    accuracy: float,
    target_accuracy: float,
    reward_base: float = 64.0,
) -> float:
    """Return the FAVOR reward: base ** (accuracy - target) - 1."""
    if reward_base <= 1.0:
        raise ValueError("reward_base must be greater than one.")

    return float(
        reward_base
        ** (
            float(accuracy)
            - float(target_accuracy)
        )
        - 1.0
    )


class _QNetwork(nn.Module):
    """Two-hidden-layer Q network used by the FAVOR DDQN agent."""

    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        hidden_dim: int = 512,
    ):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(state)


class _ReplayBuffer:
    """Replay memory for DDQN transitions."""

    def __init__(
        self,
        capacity: int,
    ):
        if capacity <= 0:
            raise ValueError(
                "Replay capacity must be positive."
            )

        self._items: Deque[
            Tuple[
                np.ndarray,
                int,
                float,
                np.ndarray,
                bool,
            ]
        ] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._items)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self._items.append(
            (
                np.asarray(
                    state,
                    dtype=np.float32,
                ).copy(),
                int(action),
                float(reward),
                np.asarray(
                    next_state,
                    dtype=np.float32,
                ).copy(),
                bool(done),
            )
        )

    def sample(
        self,
        batch_size: int,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        if batch_size > len(self._items):
            raise ValueError(
                "Cannot sample more transitions than stored."
            )

        batch = random.sample(
            self._items,
            batch_size,
        )

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
        ) = zip(*batch)

        return (
            np.stack(states),
            np.asarray(
                actions,
                dtype=np.int64,
            ),
            np.asarray(
                rewards,
                dtype=np.float32,
            ),
            np.stack(next_states),
            np.asarray(
                dones,
                dtype=np.float32,
            ),
        )

    def state_dict(self) -> Dict:
        return {
            "items": list(self._items),
            "capacity": self._items.maxlen,
        }

    def load_state_dict(
        self,
        state: Dict,
    ) -> None:
        capacity = int(
            state.get(
                "capacity",
                self._items.maxlen,
            )
        )

        self._items = deque(
            state.get("items", []),
            maxlen=capacity,
        )


class DoubleDQNAgent:
    """Double DQN agent used by FAVOR."""

    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        hidden_dim: int = 512,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        replay_capacity: int = 10_000,
        batch_size: int = 32,
        warmup_steps: int = 32,
        target_update_interval: int = 20,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 2_000,
        seed: int = 1234,
        device: Optional[str] = None,
    ):
        if state_dim <= 0 or num_actions <= 0:
            raise ValueError(
                "state_dim and num_actions must be positive."
            )

        if not 0.0 <= epsilon_end <= epsilon_start <= 1.0:
            raise ValueError(
                "Require 0 <= epsilon_end <= epsilon_start <= 1."
            )

        if target_update_interval <= 0:
            raise ValueError(
                "target_update_interval must be positive."
            )

        self.state_dim = int(state_dim)
        self.num_actions = int(num_actions)
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.warmup_steps = int(warmup_steps)

        self.target_update_interval = int(
            target_update_interval
        )

        self.epsilon_start = float(
            epsilon_start
        )

        self.epsilon_end = float(
            epsilon_end
        )

        self.epsilon_decay_steps = max(
            1,
            int(epsilon_decay_steps),
        )

        self.seed = int(seed)

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.device = torch.device(
            device
            if device is not None
            else (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )

        self.online = _QNetwork(
            self.state_dim,
            self.num_actions,
            hidden_dim,
        ).to(self.device)

        self.target = _QNetwork(
            self.state_dim,
            self.num_actions,
            hidden_dim,
        ).to(self.device)

        self.target.load_state_dict(
            self.online.state_dict()
        )

        self.target.eval()

        self.optimizer = Adam(
            self.online.parameters(),
            lr=learning_rate,
        )

        self.replay = _ReplayBuffer(
            replay_capacity
        )

        self.environment_steps = 0
        self.gradient_steps = 0

        self.last_loss: Optional[
            float
        ] = None

    @property
    def epsilon(self) -> float:
        progress = min(
            1.0,
            self.environment_steps
            / self.epsilon_decay_steps,
        )

        return float(
            self.epsilon_start
            + progress
            * (
                self.epsilon_end
                - self.epsilon_start
            )
        )

    def q_values(
        self,
        state: np.ndarray,
    ) -> np.ndarray:
        state_array = np.asarray(
            state,
            dtype=np.float32,
        )

        if state_array.shape != (
            self.state_dim,
        ):
            raise ValueError(
                f"Expected state shape "
                f"{(self.state_dim,)}, "
                f"received {state_array.shape}."
            )

        state_tensor = (
            torch.from_numpy(state_array)
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.no_grad():
            values = self.online(
                state_tensor
            ).squeeze(0)

        return (
            values.detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )

    def select_training_action(
        self,
        state: np.ndarray,
    ) -> int:
        """Select exactly one logical client action while training."""

        if random.random() < self.epsilon:
            action = random.randrange(
                self.num_actions
            )
        else:
            action = int(
                np.argmax(
                    self.q_values(state)
                )
            )

        self.environment_steps += 1

        return action

    def select_top_k(
        self,
        state: np.ndarray,
        k: int,
    ) -> List[int]:
        """Select the Top-K logical clients during FAVOR inference."""

        if not 1 <= k <= self.num_actions:
            raise ValueError(
                "k must be between 1 and num_actions."
            )

        values = self.q_values(state)

        return [
            int(index)
            for index in np.argsort(
                values
            )[-k:][::-1]
        ]

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.replay.add(
            state,
            action,
            reward,
            next_state,
            done,
        )

    def learn(
        self,
    ) -> Optional[float]:
        """Perform one Double DQN update."""

        minimum = max(
            self.batch_size,
            self.warmup_steps,
        )

        if len(self.replay) < minimum:
            return None

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
        ) = self.replay.sample(
            self.batch_size
        )

        states_t = torch.from_numpy(
            states
        ).to(self.device)

        actions_t = torch.from_numpy(
            actions
        ).to(self.device)

        rewards_t = torch.from_numpy(
            rewards
        ).to(self.device)

        next_states_t = torch.from_numpy(
            next_states
        ).to(self.device)

        dones_t = torch.from_numpy(
            dones
        ).to(self.device)

        predicted_q = self.online(
            states_t
        ).gather(
            1,
            actions_t.unsqueeze(1),
        ).squeeze(1)

        with torch.no_grad():
            # Double DQN:
            # online network selects the next action,
            # target network evaluates the selected action.
            next_actions = self.online(
                next_states_t
            ).argmax(dim=1)

            next_q = self.target(
                next_states_t
            ).gather(
                1,
                next_actions.unsqueeze(1),
            ).squeeze(1)

            target_q = (
                rewards_t
                + self.gamma
                * (1.0 - dones_t)
                * next_q
            )

        loss = nn.functional.mse_loss(
            predicted_q,
            target_q,
        )

        self.optimizer.zero_grad()

        loss.backward()

        nn.utils.clip_grad_norm_(
            self.online.parameters(),
            max_norm=10.0,
        )

        self.optimizer.step()

        self.gradient_steps += 1

        self.last_loss = float(
            loss.detach().cpu()
        )

        if (
            self.gradient_steps
            % self.target_update_interval
            == 0
        ):
            self.target.load_state_dict(
                self.online.state_dict()
            )

        return self.last_loss

    def save(
        self,
        path: str,
    ) -> None:
        destination = Path(path)

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        torch.save(
            {
                "state_dim": self.state_dim,
                "num_actions": self.num_actions,
                "online": (
                    self.online.state_dict()
                ),
                "target": (
                    self.target.state_dict()
                ),
                "optimizer": (
                    self.optimizer.state_dict()
                ),
                "replay": (
                    self.replay.state_dict()
                ),
                "environment_steps": (
                    self.environment_steps
                ),
                "gradient_steps": (
                    self.gradient_steps
                ),
                "last_loss": self.last_loss,
            },
            destination,
        )

    def load(
        self,
        path: str,
        load_replay: bool = True,
    ) -> None:
        checkpoint = torch.load(
            path,
            map_location=self.device,
        )

        if (
            int(checkpoint["state_dim"])
            != self.state_dim
        ):
            raise ValueError(
                "Agent checkpoint state dimension "
                "does not match."
            )

        if (
            int(checkpoint["num_actions"])
            != self.num_actions
        ):
            raise ValueError(
                "Agent checkpoint action dimension "
                "does not match."
            )

        self.online.load_state_dict(
            checkpoint["online"]
        )

        self.target.load_state_dict(
            checkpoint["target"]
        )

        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(
                checkpoint["optimizer"]
            )

        if (
            load_replay
            and "replay" in checkpoint
        ):
            self.replay.load_state_dict(
                checkpoint["replay"]
            )

        self.environment_steps = int(
            checkpoint.get(
                "environment_steps",
                0,
            )
        )

        self.gradient_steps = int(
            checkpoint.get(
                "gradient_steps",
                0,
            )
        )

        self.last_loss = checkpoint.get(
            "last_loss"
        )


class _FavorState:
    """PCA encoder for the global model and latest local client models."""

    def __init__(
        self,
        num_clients: int,
        n_components: int,
        seed: int = 1234,
    ):
        if not 1 <= n_components <= num_clients:
            raise ValueError(
                "PCA components must be between "
                "1 and num_clients."
            )

        self.num_clients = int(
            num_clients
        )

        self.n_components = int(
            n_components
        )

        self.seed = int(seed)

        self.pca = PCA(
            n_components=self.n_components,
            svd_solver="full",
            random_state=self.seed,
        )

        self.pca_is_fitted = False
        self.state_is_initialized = False

        self.global_embedding = np.zeros(
            self.n_components,
            dtype=np.float32,
        )

        self.local_embeddings = np.zeros(
            (
                self.num_clients,
                self.n_components,
            ),
            dtype=np.float32,
        )

        self.local_initialized = np.zeros(
            self.num_clients,
            dtype=bool,
        )

    @property
    def state_dim(self) -> int:
        return (
            self.num_clients + 1
        ) * self.n_components

    def fit_pca(
        self,
        local_model_vectors: Sequence[
            np.ndarray
        ],
    ) -> None:
        """Fit PCA once from one profiled model per logical client."""

        if self.pca_is_fitted:
            raise RuntimeError(
                "PCA is already fitted "
                "and must not be refitted."
            )

        if (
            len(local_model_vectors)
            != self.num_clients
        ):
            raise ValueError(
                "PCA fitting requires one profiled "
                "model for every client."
            )

        matrix = np.stack(
            [
                np.asarray(
                    vector,
                    dtype=np.float32,
                )
                for vector
                in local_model_vectors
            ]
        )

        self.pca.fit(matrix)

        self.pca_is_fitted = True

    def initialize_from_profiled_models(
        self,
        local_model_vectors: Sequence[
            np.ndarray
        ],
        initial_global_vector: np.ndarray,
        fit_pca_if_needed: bool,
    ) -> None:
        """Initialize one episode's state using the fixed PCA basis."""

        if (
            len(local_model_vectors)
            != self.num_clients
        ):
            raise ValueError(
                "Profiling must provide one local "
                "model for every client."
            )

        if not self.pca_is_fitted:
            if not fit_pca_if_needed:
                raise RuntimeError(
                    "A fitted PCA checkpoint is required "
                    "but was not loaded."
                )

            self.fit_pca(
                local_model_vectors
            )

        matrix = np.stack(
            [
                np.asarray(
                    vector,
                    dtype=np.float32,
                )
                for vector
                in local_model_vectors
            ]
        )

        self.local_embeddings = (
            self.pca.transform(matrix)
            .astype(np.float32)
        )

        self.global_embedding = (
            self.pca.transform(
                np.asarray(
                    initial_global_vector,
                    dtype=np.float32,
                ).reshape(1, -1)
            )[0]
            .astype(np.float32)
        )

        self.local_initialized[:] = True
        self.state_is_initialized = True

    def transform(
        self,
        model_vector: np.ndarray,
    ) -> np.ndarray:
        if not self.pca_is_fitted:
            raise RuntimeError(
                "PCA has not been fitted or loaded."
            )

        return (
            self.pca.transform(
                np.asarray(
                    model_vector,
                    dtype=np.float32,
                ).reshape(1, -1)
            )[0]
            .astype(np.float32)
        )

    def update_local(
        self,
        client_index: int,
        model_vector: np.ndarray,
    ) -> None:
        if not (
            0
            <= client_index
            < self.num_clients
        ):
            raise ValueError(
                f"Invalid logical client index: "
                f"{client_index}"
            )

        self.local_embeddings[
            client_index
        ] = self.transform(
            model_vector
        )

        self.local_initialized[
            client_index
        ] = True

    def update_global(
        self,
        model_vector: np.ndarray,
    ) -> None:
        self.global_embedding = (
            self.transform(model_vector)
        )

    def build(
        self,
    ) -> np.ndarray:
        if (
            not self.state_is_initialized
            or not bool(
                np.all(
                    self.local_initialized
                )
            )
        ):
            raise RuntimeError(
                "FAVOR state is not fully initialized."
            )

        state = np.concatenate(
            [
                self.global_embedding,
                self.local_embeddings.reshape(
                    -1
                ),
            ]
        ).astype(np.float32)

        if state.shape != (
            self.state_dim,
        ):
            raise RuntimeError(
                f"Invalid FAVOR state shape: "
                f"{state.shape}; expected "
                f"{(self.state_dim,)}."
            )

        if not np.isfinite(
            state
        ).all():
            raise RuntimeError(
                "FAVOR state contains NaN "
                "or infinity."
            )

        return state

    def save_pca(
        self,
        path: str,
    ) -> None:
        if not self.pca_is_fitted:
            raise RuntimeError(
                "Cannot save PCA before fitting it."
            )

        destination = Path(path)

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(
            {
                "pca": self.pca,
                "num_clients": (
                    self.num_clients
                ),
                "n_components": (
                    self.n_components
                ),
            },
            destination,
        )

    def load_pca(
        self,
        path: str,
    ) -> None:
        source = Path(path)

        if not source.exists():
            raise FileNotFoundError(
                f"PCA checkpoint not found: "
                f"{source}"
            )

        payload = joblib.load(
            source
        )

        if isinstance(
            payload,
            PCA,
        ):
            loaded_pca = payload
            loaded_clients = (
                self.num_clients
            )
            loaded_components = (
                loaded_pca.n_components_
            )
        else:
            loaded_pca = payload["pca"]

            loaded_clients = int(
                payload["num_clients"]
            )

            loaded_components = int(
                payload["n_components"]
            )

        if (
            loaded_clients
            != self.num_clients
        ):
            raise ValueError(
                "PCA checkpoint client count "
                "does not match the experiment."
            )

        if (
            loaded_components
            != self.n_components
        ):
            raise ValueError(
                "PCA checkpoint component count "
                "does not match the experiment."
            )

        self.pca = loaded_pca
        self.pca_is_fitted = True
        self.state_is_initialized = False
        self.local_initialized[:] = False


class Favor(FedAvg):
    """FAVOR client-selection strategy integrated with Flower."""

    def __init__(
        self,
        num_clients: int,
        selected_clients: int,
        mode: str,
        pca_components: int,
        target_accuracy: float,
        agent: Optional[
            DoubleDQNAgent
        ] = None,
        agent_checkpoint: Optional[
            str
        ] = None,
        pca_checkpoint: Optional[
            str
        ] = None,
        episode: int = 1,
        reward_base: float = 64.0,
        seed: int = 1234,
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
        )

        normalized_mode = (
            mode.strip().lower()
        )

        if normalized_mode not in {
            "train",
            "eval",
        }:
            raise ValueError(
                "mode must be 'train' or 'eval'."
            )

        if not (
            1
            <= selected_clients
            <= num_clients
        ):
            raise ValueError(
                "selected_clients must be between "
                "1 and num_clients."
            )

        if reward_base <= 1.0:
            raise ValueError(
                "reward_base must be greater than one."
            )

        self.num_clients = int(
            num_clients
        )

        self.selected_clients = int(
            selected_clients
        )

        self.mode = normalized_mode

        self.target_accuracy = float(
            target_accuracy
        )

        self.reward_base = float(
            reward_base
        )

        self.episode = int(
            episode
        )

        self.seed = int(seed)

        self.agent_checkpoint = (
            agent_checkpoint
        )

        self.pca_checkpoint = (
            pca_checkpoint
        )

        self.state_encoder = _FavorState(
            num_clients=self.num_clients,
            n_components=pca_components,
            seed=self.seed,
        )

        if (
            self.pca_checkpoint
            and Path(
                self.pca_checkpoint
            ).exists()
        ):
            self.state_encoder.load_pca(
                self.pca_checkpoint
            )

        elif self.mode == "eval":
            raise FileNotFoundError(
                "Evaluation requires the PCA "
                "checkpoint created during training."
            )

        expected_state_dim = (
            self.state_encoder.state_dim
        )

        if agent is None:
            raise ValueError(
                "A shared Double DQN agent "
                "must be provided."
            )

        if (
            agent.state_dim
            != expected_state_dim
        ):
            raise ValueError(
                f"Agent state_dim="
                f"{agent.state_dim}, but FAVOR "
                f"requires {expected_state_dim}."
            )

        if (
            agent.num_actions
            != self.num_clients
        ):
            raise ValueError(
                "Agent action count must equal "
                "the number of logical clients."
            )

        self.agent = agent

        # Flower/Ray proxy IDs can change between episodes.
        # DDQN actions always use stable logical IDs.
        self.proxy_to_logical: Dict[
            str,
            int,
        ] = {}

        self.logical_to_proxy: Dict[
            int,
            str,
        ] = {}

        self.current_global_parameters: Optional[
            Parameters
        ] = None

        self.pending_state: Optional[
            np.ndarray
        ] = None

        self.pending_action: Optional[
            int
        ] = None

        self.last_selected_cids: List[
            str
        ] = []

        self.last_selected_proxy_cids: List[
            str
        ] = []

        self.last_q_values: Dict[
            str,
            float,
        ] = {}

        self.last_reward: Optional[
            float
        ] = None

        self.last_accuracy: Optional[
            float
        ] = None

        self.last_dqn_loss: Optional[
            float
        ] = None

        self.last_phase = (
            "initialization"
        )

        self.episode_done = False

        self.target_round: Optional[
            int
        ] = None

    @property
    def is_profiling_complete(
        self,
    ) -> bool:
        return (
            self.state_encoder
            .state_is_initialized
        )

    def _base_fit_config(
        self,
        server_round: int,
    ) -> Dict[str, Scalar]:
        config: Dict[
            str,
            Scalar,
        ] = {}

        if (
            self.on_fit_config_fn
            is not None
        ):
            config.update(
                self.on_fit_config_fn(
                    server_round
                )
            )

        config["favor_mode"] = (
            self.mode
        )

        config["favor_episode"] = (
            self.episode
        )

        return config

    def _logical_cid_from_result(
        self,
        fit_res: FitRes,
    ) -> int:
        if (
            "logical_cid"
            not in fit_res.metrics
        ):
            raise RuntimeError(
                "Client fit metrics must include "
                "the stable logical_cid."
            )

        logical_cid = int(
            fit_res.metrics[
                "logical_cid"
            ]
        )

        if not (
            0
            <= logical_cid
            < self.num_clients
        ):
            raise RuntimeError(
                f"logical_cid={logical_cid} "
                f"is outside "
                f"[0, {self.num_clients - 1}]."
            )

        return logical_cid

    def _build_logical_client_mapping(
        self,
        results: List[
            Tuple[
                ClientProxy,
                FitRes,
            ]
        ],
    ) -> None:
        """Build the episode-specific proxy mapping from stable logical IDs."""

        proxy_to_logical: Dict[
            str,
            int,
        ] = {}

        logical_to_proxy: Dict[
            int,
            str,
        ] = {}

        for client, fit_res in results:
            proxy_cid = str(
                client.cid
            )

            logical_cid = (
                self._logical_cid_from_result(
                    fit_res
                )
            )

            if (
                logical_cid
                in logical_to_proxy
            ):
                raise RuntimeError(
                    "Duplicate logical_cid returned "
                    "during profiling: "
                    f"{logical_cid}."
                )

            if (
                proxy_cid
                in proxy_to_logical
            ):
                raise RuntimeError(
                    "Duplicate Flower proxy ID "
                    "during profiling: "
                    f"{proxy_cid}."
                )

            proxy_to_logical[
                proxy_cid
            ] = logical_cid

            logical_to_proxy[
                logical_cid
            ] = proxy_cid

        expected = set(
            range(self.num_clients)
        )

        observed = set(
            logical_to_proxy
        )

        if observed != expected:
            missing = sorted(
                expected - observed
            )

            extra = sorted(
                observed - expected
            )

            raise RuntimeError(
                "Invalid logical client mapping. "
                f"Missing={missing}, "
                f"extra={extra}."
            )

        self.proxy_to_logical = (
            proxy_to_logical
        )

        self.logical_to_proxy = (
            logical_to_proxy
        )

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[
        Tuple[
            ClientProxy,
            FitIns,
        ]
    ]:
        available_clients = list(
            client_manager
            .all()
            .values()
        )

        if (
            len(available_clients)
            != self.num_clients
        ):
            raise RuntimeError(
                f"Expected {self.num_clients} "
                f"available clients, found "
                f"{len(available_clients)}."
            )

        available_by_proxy = {
            str(client.cid): client
            for client
            in available_clients
        }

        self.current_global_parameters = (
            parameters
        )

        config = self._base_fit_config(
            server_round
        )

        if not self.is_profiling_complete:
            # Every episode profiles all clients.
            # PCA is fitted only once in the first episode.
            self.last_phase = "profiling"

            config["favor_phase"] = (
                "profiling"
            )

            self.pending_state = None
            self.pending_action = None

            selected_clients = (
                available_clients
            )

            self.last_selected_cids = [
                str(index)
                for index
                in range(
                    self.num_clients
                )
            ]

            self.last_selected_proxy_cids = [
                str(client.cid)
                for client
                in selected_clients
            ]

        else:
            if (
                len(self.logical_to_proxy)
                != self.num_clients
            ):
                raise RuntimeError(
                    "Logical-to-proxy mapping "
                    "is incomplete after profiling."
                )

            state = (
                self.state_encoder.build()
            )

            q_values = (
                self.agent.q_values(state)
            )

            self.last_q_values = {
                str(logical_cid): float(
                    q_values[logical_cid]
                )
                for logical_cid
                in range(
                    self.num_clients
                )
            }

            if self.mode == "train":
                self.last_phase = (
                    "agent_training"
                )

                selected_logical_ids = [
                    self.agent
                    .select_training_action(
                        state
                    )
                ]

                self.pending_state = state

                self.pending_action = (
                    selected_logical_ids[0]
                )

            else:
                self.last_phase = (
                    "top_k_inference"
                )

                selected_logical_ids = (
                    self.agent.select_top_k(
                        state,
                        self.selected_clients,
                    )
                )

                self.pending_state = None
                self.pending_action = None

            selected_proxy_ids = [
                self.logical_to_proxy[
                    logical_cid
                ]
                for logical_cid
                in selected_logical_ids
            ]

            missing_proxies = [
                proxy_id
                for proxy_id
                in selected_proxy_ids
                if proxy_id
                not in available_by_proxy
            ]

            if missing_proxies:
                raise RuntimeError(
                    "Flower proxy IDs changed inside "
                    "the same episode: "
                    f"{missing_proxies}."
                )

            selected_clients = [
                available_by_proxy[
                    proxy_id
                ]
                for proxy_id
                in selected_proxy_ids
            ]

            config["favor_phase"] = (
                self.last_phase
            )

            self.last_selected_cids = [
                str(logical_cid)
                for logical_cid
                in selected_logical_ids
            ]

            self.last_selected_proxy_cids = (
                selected_proxy_ids
            )

        fit_instruction = FitIns(
            parameters,
            config,
        )

        return [
            (
                client,
                fit_instruction,
            )
            for client
            in selected_clients
        ]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[
            Tuple[
                ClientProxy,
                FitRes,
            ]
        ],
        failures,
    ) -> Tuple[
        Optional[Parameters],
        Dict[str, Scalar],
    ]:
        if (
            failures
            and not self.accept_failures
        ):
            return None, {}

        if not results:
            return None, {}

        if not self.is_profiling_complete:
            if (
                self.current_global_parameters
                is None
            ):
                raise RuntimeError(
                    "Initial global parameters "
                    "are unavailable."
                )

            if (
                len(results)
                != self.num_clients
            ):
                raise RuntimeError(
                    "Profiling must return one "
                    "successful result from "
                    "every client."
                )

            self._build_logical_client_mapping(
                results
            )

            vectors_by_logical_id: Dict[
                int,
                np.ndarray,
            ] = {}

            for client, fit_res in results:
                logical_cid = (
                    self._logical_cid_from_result(
                        fit_res
                    )
                )

                proxy_cid = str(
                    client.cid
                )

                if (
                    self.proxy_to_logical.get(
                        proxy_cid
                    )
                    != logical_cid
                ):
                    raise RuntimeError(
                        "Inconsistent logical "
                        "client mapping."
                    )

                vectors_by_logical_id[
                    logical_cid
                ] = flatten_ndarrays(
                    parameters_to_ndarrays(
                        fit_res.parameters
                    )
                )

            local_vectors = [
                vectors_by_logical_id[
                    logical_cid
                ]
                for logical_cid
                in range(
                    self.num_clients
                )
            ]

            initial_global_vector = (
                flatten_ndarrays(
                    parameters_to_ndarrays(
                        self.current_global_parameters
                    )
                )
            )

            pca_was_already_fitted = (
                self.state_encoder
                .pca_is_fitted
            )

            self.state_encoder.initialize_from_profiled_models(
                local_model_vectors=(
                    local_vectors
                ),
                initial_global_vector=(
                    initial_global_vector
                ),
                fit_pca_if_needed=(
                    self.mode == "train"
                ),
            )

            if not pca_was_already_fitted:
                if not self.pca_checkpoint:
                    raise RuntimeError(
                        "A PCA checkpoint path "
                        "is required for "
                        "FAVOR training."
                    )

                self.state_encoder.save_pca(
                    self.pca_checkpoint
                )

            # Profiling initializes the state.
            # It does not update the global model.
            return (
                self.current_global_parameters,
                {
                    "profiling_clients": (
                        self.num_clients
                    ),
                    "pca_fitted_this_episode": (
                        not pca_was_already_fitted
                    ),
                },
            )

        for client, fit_res in results:
            proxy_cid = str(
                client.cid
            )

            if (
                proxy_cid
                not in self.proxy_to_logical
            ):
                raise RuntimeError(
                    "Unknown Flower proxy ID "
                    f"after profiling: "
                    f"{proxy_cid}."
                )

            mapped_logical_cid = (
                self.proxy_to_logical[
                    proxy_cid
                ]
            )

            reported_logical_cid = (
                self._logical_cid_from_result(
                    fit_res
                )
            )

            if (
                mapped_logical_cid
                != reported_logical_cid
            ):
                raise RuntimeError(
                    "Client identity mismatch: "
                    "proxy mapping and "
                    "logical_cid differ."
                )

            local_vector = flatten_ndarrays(
                parameters_to_ndarrays(
                    fit_res.parameters
                )
            )

            self.state_encoder.update_local(
                mapped_logical_cid,
                local_vector,
            )

        (
            aggregated_parameters,
            aggregated_metrics,
        ) = super().aggregate_fit(
            server_round,
            results,
            failures,
        )

        if (
            aggregated_parameters
            is not None
        ):
            global_vector = (
                flatten_ndarrays(
                    parameters_to_ndarrays(
                        aggregated_parameters
                    )
                )
            )

            self.state_encoder.update_global(
                global_vector
            )

        return (
            aggregated_parameters,
            aggregated_metrics,
        )

    def evaluate(
        self,
        server_round: int,
        parameters: Parameters,
    ):
        result = super().evaluate(
            server_round,
            parameters,
        )

        if result is None:
            return None

        loss, metrics = result

        accuracy = float(
            metrics.get(
                "accuracy",
                0.0,
            )
        )

        self.last_accuracy = accuracy

        # Profiling is not an RL action.
        # It must not create a replay transition.
        if (
            server_round > 0
            and self.is_profiling_complete
            and self.last_phase
            != "profiling"
        ):
            reward = favor_reward(
                accuracy=accuracy,
                target_accuracy=(
                    self.target_accuracy
                ),
                reward_base=(
                    self.reward_base
                ),
            )

            self.last_reward = reward

            done = (
                accuracy
                >= self.target_accuracy
            )

            if (
                self.mode == "train"
                and self.pending_state
                is not None
                and self.pending_action
                is not None
            ):
                next_state = (
                    self.state_encoder.build()
                )

                self.agent.remember(
                    state=self.pending_state,
                    action=self.pending_action,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                )

                self.last_dqn_loss = (
                    self.agent.learn()
                )

                if self.agent_checkpoint:
                    self.agent.save(
                        self.agent_checkpoint
                    )

            if (
                done
                and not self.episode_done
            ):
                self.episode_done = True
                self.target_round = (
                    server_round
                )

        else:
            self.last_reward = None
            self.last_dqn_loss = None

        return loss, metrics
