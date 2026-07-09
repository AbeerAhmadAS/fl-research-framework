import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
from flwr.common import FitIns, FitRes, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy


class OORT(fl.server.strategy.FedAvg):
    """
    Core Oort participant selection strategy.

    This file contains only the algorithm logic:
    - statistical utility
    - system utility penalty
    - exploration/exploitation
    - temporal uncertainty
    - pacer
    - client history updates

    It does not contain logging, CSV files, hardware reporting,
    communication-cost calculation, fairness calculation, or evaluation analysis.
    """

    def __init__(
        self,
        num_clients: int,
        exploration_factor: float = 0.90,
        exploration_decay: float = 0.98,
        min_exploration_factor: float = 0.20,
        pacer_window: int = 20,
        pacer_step: float = 60.0,
        straggler_penalty_alpha: float = 2.0,
        cutoff_percentage: float = 0.95,
        max_selection_per_client: int = 10,
        seed: int = 1234,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.num_clients = num_clients

        # Oort parameters
        self.exploration_factor = exploration_factor
        self.exploration_decay = exploration_decay
        self.min_exploration_factor = min_exploration_factor
        self.pacer_window = pacer_window
        self.pacer_step = pacer_step
        self.preferred_duration_T = pacer_step
        self.straggler_penalty_alpha = straggler_penalty_alpha
        self.cutoff_percentage = cutoff_percentage
        self.max_selection_per_client = max_selection_per_client
        self.seed = seed

        random.seed(seed)
        np.random.seed(seed)

        # Internal Oort state
        self.participation = defaultdict(int)
        self.explored_clients = set()
        self.client_stat_utility = defaultdict(float)
        self.client_fit_duration = defaultdict(float)
        self.client_last_round = defaultdict(int)
        self.client_oort_utility = defaultdict(float)
        self.round_stat_utility_history: List[float] = []

        # Current-round algorithm metadata
        self.current_num_exploit = 0
        self.current_num_explore = 0
        self.current_selected_clients: List[str] = []
        self.current_round_exploration_factor = exploration_factor

    def _num_fit_clients(self, num_available_clients: int) -> int:
        sample_size, _ = self.num_fit_clients(num_available_clients)
        return sample_size

    def _update_pacer(self) -> None:
        W = self.pacer_window

        if len(self.round_stat_utility_history) < 2 * W:
            return

        previous_window = self.round_stat_utility_history[-2 * W : -W]
        recent_window = self.round_stat_utility_history[-W:]

        # Pacer:
        # If the achieved statistical utility decreases, Oort relaxes T
        # to allow slower but potentially more useful clients.
        if sum(previous_window) > sum(recent_window):
            self.preferred_duration_T += self.pacer_step

    def _calculate_oort_utility(self, cid: str, server_round: int) -> float:
        stat_utility = float(self.client_stat_utility.get(cid, 0.0))
        fit_duration = float(self.client_fit_duration.get(cid, 0.0))

        # Temporal uncertainty bonus.
        # This encourages revisiting clients that may have become useful again.
        last_round = int(self.client_last_round.get(cid, 0))
        staleness = max(1, server_round - last_round)
        temporal_bonus = np.sqrt(0.1 * np.log(max(server_round, 2)) * staleness)

        utility = stat_utility + temporal_bonus

        # System utility penalty.
        # If the client is slower than preferred duration T, penalize it.
        if fit_duration > self.preferred_duration_T and fit_duration > 0:
            penalty = (self.preferred_duration_T / fit_duration) ** self.straggler_penalty_alpha
            utility *= penalty

        return float(max(utility, 1e-12))

    def _weighted_sample_without_replacement(
        self,
        clients: List[ClientProxy],
        utilities: Dict[str, float],
        k: int,
    ) -> List[ClientProxy]:
        if k <= 0 or not clients:
            return []

        k = min(k, len(clients))
        weights = np.array(
            [max(float(utilities.get(client.cid, 0.0)), 1e-12) for client in clients],
            dtype=np.float64,
        )

        if weights.sum() <= 0:
            probs = np.ones(len(clients), dtype=np.float64) / len(clients)
        else:
            probs = weights / weights.sum()

        selected_indices = np.random.choice(
            len(clients),
            size=k,
            replace=False,
            p=probs,
        )

        return [clients[i] for i in selected_indices]

    def _oort_select_clients(
        self,
        server_round: int,
        available_clients: List[ClientProxy],
        sample_size: int,
    ) -> Tuple[List[ClientProxy], int, int]:
        self._update_pacer()

        explored_candidates = [
            client
            for client in available_clients
            if client.cid in self.explored_clients
            and self.participation[client.cid] < self.max_selection_per_client
        ]

        unexplored_candidates = [
            client
            for client in available_clients
            if client.cid not in self.explored_clients
        ]

        self.current_round_exploration_factor = self.exploration_factor

        explore_k = int(round(self.exploration_factor * sample_size))
        explore_k = min(explore_k, sample_size)
        exploit_k = sample_size - explore_k

        # If there are no explored clients yet, force exploration.
        if not explored_candidates:
            explore_k = sample_size
            exploit_k = 0

        utilities = {}
        for client in explored_candidates:
            utilities[client.cid] = self._calculate_oort_utility(
                cid=client.cid,
                server_round=server_round,
            )
            self.client_oort_utility[client.cid] = utilities[client.cid]

        exploit_selected = []

        if exploit_k > 0 and explored_candidates:
            sorted_clients = sorted(
                explored_candidates,
                key=lambda c: utilities.get(c.cid, 0.0),
                reverse=True,
            )

            cutoff_index = min(exploit_k - 1, len(sorted_clients) - 1)
            cutoff_utility = utilities.get(sorted_clients[cutoff_index].cid, 0.0)
            threshold = self.cutoff_percentage * cutoff_utility

            high_utility_pool = [
                c for c in sorted_clients
                if utilities.get(c.cid, 0.0) >= threshold
            ]

            exploit_selected = self._weighted_sample_without_replacement(
                clients=high_utility_pool,
                utilities=utilities,
                k=exploit_k,
            )

        # -------------------------------------------------------------------------
        # Difference from the original Oort implementation:
        # Oort can use prior device profiling information to guide exploration
        # toward faster unexplored clients. Since such information is not
        # available in Flower before a client's first participation, unexplored
        # clients are selected randomly. After their first participation,
        # actual performance statistics are collected and used by Oort in all
        # subsequent rounds.
        # -------------------------------------------------------------------------
        remaining_needed = sample_size - len(exploit_selected)

        already_selected = set(c.cid for c in exploit_selected)
        explore_pool = [
            c for c in unexplored_candidates
            if c.cid not in already_selected
        ]

        if len(explore_pool) < remaining_needed:
            fallback_pool = [
                c for c in available_clients
                if c.cid not in already_selected
                and self.participation[c.cid] < self.max_selection_per_client
            ]
            explore_pool = fallback_pool

        explore_selected = random.sample(
            explore_pool,
            k=min(remaining_needed, len(explore_pool)),
        )

        selected = exploit_selected + explore_selected

        # If still short, fill randomly from available clients.
        if len(selected) < sample_size:
            selected_cids = set(c.cid for c in selected)
            fill_pool = [
                c for c in available_clients
                if c.cid not in selected_cids
            ]
            selected += random.sample(
                fill_pool,
                k=min(sample_size - len(selected), len(fill_pool)),
            )

        # Decay exploration factor after each round.
        if self.exploration_factor > self.min_exploration_factor:
            self.exploration_factor = max(
                self.min_exploration_factor,
                self.exploration_factor * self.exploration_decay,
            )

        return selected, len(exploit_selected), len(explore_selected)

    def configure_fit(self, server_round, parameters, client_manager):
        available_clients = list(client_manager.all().values())
        sample_size = self._num_fit_clients(len(available_clients))

        config = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)

        config["strategy"] = "Oort"
        config["server_round"] = server_round

        fit_ins = FitIns(parameters, config)

        sampled_clients, num_exploit, num_explore = self._oort_select_clients(
            server_round=server_round,
            available_clients=available_clients,
            sample_size=sample_size,
        )

        selected = [(client, fit_ins) for client in sampled_clients]
        selected_cids = [client.cid for client in sampled_clients]

        for cid in selected_cids:
            self.participation[cid] += 1

        self.current_num_exploit = num_exploit
        self.current_num_explore = num_explore
        self.current_selected_clients = selected_cids

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

        round_stat_utility = 0.0

        for client, fit_res in results:
            cid = client.cid

            train_loss = float(fit_res.metrics.get("train_loss", 0.0))
            num_examples = int(fit_res.metrics.get("num_examples", 0))
            fit_duration = float(fit_res.metrics.get("fit_duration", 0.0))

            # Oort statistical utility:
            # Prefer the explicit oort_stat_utility computed by the client.
            # Fallback to num_examples * train_loss if unavailable.
            stat_utility = float(
                fit_res.metrics.get(
                    "oort_stat_utility",
                    num_examples * max(train_loss, 0.0),
                )
            )

            self.explored_clients.add(cid)
            self.client_stat_utility[cid] = stat_utility
            self.client_fit_duration[cid] = fit_duration
            self.client_last_round[cid] = server_round

            round_stat_utility += stat_utility

        self.round_stat_utility_history.append(round_stat_utility)

        return aggregated_parameters, aggregated_metrics
