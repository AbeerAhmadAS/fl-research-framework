import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from .fedavg import FedAvg
import numpy as np
from flwr.common import FitIns, FitRes, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy


class OORT(FedAvg):
    """
    Core Oort participant selection strategy.

    This file contains only the algorithm logic:
    - statistical utility
    - system utility penalty
    - exploration/exploitation
    - temporal uncertainty
    - pacer
    - client history updates

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
        utility_clip_percentile: float = 95.0,
        initial_pacer_percentile: float = 20.0,
        pacer_delta: float = 5.0,
        pacer_tolerance: float = 0.10,
        blacklist_max_fraction: float = 0.50,
        client_profiles: Optional[Dict[str, Dict[str, float]]] = None,
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

        # pacer_step is kept as the initial/fallback preferred duration in seconds.
        # The Oort pacer then adjusts a duration percentile instead of adding a
        # fixed number of seconds directly to preferred_duration_T.
        self.pacer_step = pacer_step
        self.preferred_duration_T = pacer_step
        self.initial_pacer_percentile = float(
            np.clip(initial_pacer_percentile, 0.0, 100.0)
        )
        self.round_threshold_percentile = self.initial_pacer_percentile
        self.pacer_delta = max(float(pacer_delta), 0.0)
        self.pacer_tolerance = max(float(pacer_tolerance), 0.0)

        self.straggler_penalty_alpha = straggler_penalty_alpha
        self.cutoff_percentage = cutoff_percentage
        self.max_selection_per_client = max_selection_per_client
        self.utility_clip_percentile = float(
            np.clip(utility_clip_percentile, 0.0, 100.0)
        )
        self.blacklist_max_fraction = float(
            np.clip(blacklist_max_fraction, 0.0, 1.0)
        )
        self.seed = seed

        # Optional client profiles used only to guide exploration before a
        # client's first successful participation. The runner can provide
        # simulated profiles, while a real deployment can provide measured
        # device profiles without changing the Oort selection logic.
        self.client_profiles = client_profiles or {}
        self.client_estimated_duration = defaultdict(float)
        self.client_initial_reward = defaultdict(lambda: 1.0)

        for cid, profile in self.client_profiles.items():
            estimated_duration = float(profile.get("estimated_duration", 0.0))
            initial_reward = float(profile.get("initial_reward", 1.0))

            if np.isfinite(estimated_duration) and estimated_duration > 0.0:
                self.client_estimated_duration[str(cid)] = estimated_duration

            if np.isfinite(initial_reward) and initial_reward > 0.0:
                self.client_initial_reward[str(cid)] = initial_reward

        random.seed(seed)
        np.random.seed(seed)

        # Internal Oort state
        self.participation = defaultdict(int)
        self.explored_clients = set()
        self.blacklisted_clients = set()
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

    def _refresh_blacklist(self) -> None:
        """Blacklist repeatedly selected clients without blocking all clients."""
        if self.max_selection_per_client <= 0:
            return

        candidates = [
            cid
            for cid, count in self.participation.items()
            if count >= self.max_selection_per_client
        ]

        max_blacklisted = int(self.num_clients * self.blacklist_max_fraction)
        if self.blacklist_max_fraction > 0.0:
            max_blacklisted = max(1, max_blacklisted)

        candidates.sort(
            key=lambda cid: self.participation[cid],
            reverse=True,
        )

        self.blacklisted_clients = set(candidates[:max_blacklisted])

    def _is_client_eligible(self, cid: str) -> bool:
        return cid not in self.blacklisted_clients

    def _refresh_preferred_duration(self) -> None:
        """Set T from the current percentile of observed client durations."""
        observed_durations = [
            float(duration)
            for duration in self.client_fit_duration.values()
            if np.isfinite(duration) and duration > 0.0
        ]

        if not observed_durations:
            return

        self.preferred_duration_T = float(
            np.percentile(
                observed_durations,
                self.round_threshold_percentile,
            )
        )

    def _update_pacer(self) -> None:
        W = self.pacer_window

        # Always refresh T from the currently selected duration percentile.
        self._refresh_preferred_duration()

        if W <= 0 or len(self.round_stat_utility_history) < 2 * W:
            return

        previous_window = self.round_stat_utility_history[-2 * W : -W]
        recent_window = self.round_stat_utility_history[-W:]

        previous_sum = float(sum(previous_window))
        recent_sum = float(sum(recent_window))
        denominator = max(abs(previous_sum), 1e-12)
        relative_change = abs(recent_sum - previous_sum) / denominator

        # Oort pacer:
        # When utility decreases or becomes nearly flat, increase the duration
        # percentile so slower but potentially useful clients can participate.
        if recent_sum < previous_sum or relative_change <= self.pacer_tolerance:
            self.round_threshold_percentile = min(
                100.0,
                self.round_threshold_percentile + self.pacer_delta,
            )
            self._refresh_preferred_duration()

    def _get_normalized_stat_utilities(
        self,
        client_ids: List[str],
    ) -> Dict[str, float]:
        """Clip statistical rewards, then normalize them to the [0, 1] range."""
        if not client_ids:
            return {}

        raw_utilities = np.array(
            [
                max(float(self.client_stat_utility.get(cid, 0.0)), 0.0)
                for cid in client_ids
            ],
            dtype=np.float64,
        )

        finite_utilities = raw_utilities[np.isfinite(raw_utilities)]
        if finite_utilities.size == 0:
            return {cid: 0.0 for cid in client_ids}

        # Utility clipping limits the influence of extreme statistical rewards.
        clip_value = float(
            np.percentile(finite_utilities, self.utility_clip_percentile)
        )
        clipped = np.minimum(raw_utilities, clip_value)
        clipped = np.where(np.isfinite(clipped), clipped, 0.0)

        min_reward = float(np.min(clipped))
        max_reward = float(np.max(clipped))
        reward_range = max_reward - min_reward

        if reward_range <= 1e-12:
            normalized = np.ones_like(clipped, dtype=np.float64)
        else:
            normalized = (clipped - min_reward) / reward_range

        return {
            cid: float(normalized[index])
            for index, cid in enumerate(client_ids)
        }

    def _calculate_oort_utility(
        self,
        cid: str,
        server_round: int,
        normalized_stat_utility: Optional[float] = None,
    ) -> float:
        if normalized_stat_utility is None:
            normalized_stat_utility = float(
                self._get_normalized_stat_utilities([cid]).get(cid, 0.0)
            )

        fit_duration = float(self.client_fit_duration.get(cid, 0.0))

        # Temporal uncertainty from Oort Algorithm 1.
        # Clients with older feedback receive a larger uncertainty bonus.
        last_round = max(int(self.client_last_round.get(cid, 0)), 1)
        temporal_bonus = np.sqrt(
            0.1 * np.log(max(server_round, 2)) / last_round
        )

        utility = float(normalized_stat_utility) + float(temporal_bonus)

        # System utility penalty.
        # If the client is slower than preferred duration T, penalize it.
        if fit_duration > self.preferred_duration_T and fit_duration > 0:
            penalty = (
                self.preferred_duration_T / fit_duration
            ) ** self.straggler_penalty_alpha
            utility *= penalty

        return float(max(utility, 1e-12))

    def _calculate_exploration_utility(self, cid: str) -> float:
        """Calculate speed-guided utility for a not-yet-explored client."""
        initial_reward = float(self.client_initial_reward.get(cid, 1.0))
        estimated_duration = float(
            self.client_estimated_duration.get(
                cid,
                self.client_fit_duration.get(cid, 0.0),
            )
        )

        # A missing profile receives a neutral, very small utility rather than
        # silently being treated as a known fast client.
        if not np.isfinite(estimated_duration) or estimated_duration <= 0.0:
            return 1e-12

        utility = max(initial_reward, 1e-12)

        # Apply the same system penalty used during exploitation. This makes
        # exploration prefer clients expected to finish within the current T.
        if estimated_duration > self.preferred_duration_T:
            penalty = (
                self.preferred_duration_T / estimated_duration
            ) ** self.straggler_penalty_alpha
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
        self._refresh_blacklist()
        self._update_pacer()

        eligible_clients = [
            client
            for client in available_clients
            if self._is_client_eligible(client.cid)
        ]
        sample_size = min(sample_size, len(eligible_clients))

        explored_candidates = [
            client
            for client in eligible_clients
            if client.cid in self.explored_clients
        ]

        unexplored_candidates = [
            client
            for client in eligible_clients
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

        explored_client_ids = [client.cid for client in explored_candidates]
        normalized_stat_utilities = self._get_normalized_stat_utilities(
            explored_client_ids
        )

        utilities = {}
        for client in explored_candidates:
            utilities[client.cid] = self._calculate_oort_utility(
                cid=client.cid,
                server_round=server_round,
                normalized_stat_utility=normalized_stat_utilities.get(
                    client.cid,
                    0.0,
                ),
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

        # Speed-guided exploration:
        # Unexplored clients are sampled according to their initial profile,
        # especially their estimated training duration. Faster clients receive
        # higher exploration utility, while slow clients are penalized using T.
        remaining_needed = sample_size - len(exploit_selected)

        already_selected = set(c.cid for c in exploit_selected)
        explore_pool = [
            c for c in unexplored_candidates
            if c.cid not in already_selected
        ]

        if len(explore_pool) < remaining_needed:
            fallback_pool = [
                c for c in eligible_clients
                if c.cid not in already_selected
            ]
            explore_pool = fallback_pool

        exploration_utilities = {
            client.cid: self._calculate_exploration_utility(client.cid)
            for client in explore_pool
        }

        explore_selected = self._weighted_sample_without_replacement(
            clients=explore_pool,
            utilities=exploration_utilities,
            k=remaining_needed,
        )

        selected = exploit_selected + explore_selected

        # If still short, fill randomly from eligible clients only.
        if len(selected) < sample_size:
            selected_cids = set(c.cid for c in selected)
            fill_pool = [
                c for c in eligible_clients
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

            fit_duration = float(fit_res.metrics.get("fit_duration", 0.0))

            # Oort requires the exact statistical utility computed by the client.
            # No approximation based on average train loss is used.
            if "oort_stat_utility" not in fit_res.metrics:
                raise ValueError(
                    f"Client {cid} did not return the required "
                    "'oort_stat_utility' metric."
                )

            stat_utility = float(fit_res.metrics["oort_stat_utility"])

            if not np.isfinite(stat_utility) or stat_utility < 0.0:
                raise ValueError(
                    f"Client {cid} returned an invalid oort_stat_utility: "
                    f"{stat_utility}."
                )

            if not np.isfinite(fit_duration) or fit_duration < 0.0:
                raise ValueError(
                    f"Client {cid} returned an invalid fit_duration: "
                    f"{fit_duration}."
                )

            self.explored_clients.add(cid)
            self.client_stat_utility[cid] = stat_utility
            self.client_fit_duration[cid] = fit_duration
            self.client_estimated_duration[cid] = fit_duration
            self.client_last_round[cid] = server_round

            # Count only successful client feedback for blacklisting.
            self.participation[cid] += 1

            round_stat_utility += stat_utility

        self.round_stat_utility_history.append(round_stat_utility)
        self._refresh_blacklist()

        return aggregated_parameters, aggregated_metrics
