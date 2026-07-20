#This file represents the core FedFS strategy.
#FedFS = Federating Fast and Slow.
#
#FedFS core logic:
#1) deadline-based local training
#2) partial work return from clients
#3) work contribution ratio wk
#4) importance sampling using probability proportional to 1 - wk + epsilon
#5) alternating fast and slow timeouts

from collections import defaultdict
from typing import List, Optional, Tuple
from .fedavg import FedAvg
import numpy as np
from flwr.common import FitIns, FitRes, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy


class FedFS(FedAvg):
    def __init__(
        self,
        r_fast: int = 1,
        r_slow: int = 1,
        delta_fast: float = 90.0,
        delta_slow: float = 180.0,
        epsilon: float = 0.05,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # Number of consecutive fast and slow rounds.
        self.r_fast = r_fast
        self.r_slow = r_slow

        # FedFS alternating timeout values.
        # Fast rounds use a smaller deadline.
        # Slow rounds use a larger deadline.
        self.delta_fast = delta_fast
        self.delta_slow = delta_slow

        # Minimum client selection probability factor.
        # This prevents high-contribution clients from being fully excluded.
        self.epsilon = epsilon

        # FedFS contribution history.
        # actual_work = cumulative processed samples.
        # max_work = cumulative maximum possible samples.
        # wk = actual_work / max_work.
        self.client_actual_work = defaultdict(float)
        self.client_max_work = defaultdict(float)
        self.client_work_ratio = defaultdict(float)

    def _fedfs_round_type(self, server_round: int) -> str:
        cycle = self.r_fast + self.r_slow
        pos = (server_round - 1) % cycle

        if pos < self.r_fast:
            return "fast"
        return "slow"

    def _fedfs_deadline(self, server_round: int) -> float:
        round_type = self._fedfs_round_type(server_round)

        if round_type == "fast":
            return self.delta_fast
        return self.delta_slow

    def _num_fit_clients(self, num_available_clients: int) -> int:
        sample_size, _ = self.num_fit_clients(num_available_clients)
        return sample_size

    def _get_wk(self, cid: str) -> float:
        max_work = self.client_max_work.get(cid, 0.0)

        if max_work <= 0.0:
            return 0.0

        return float(self.client_actual_work.get(cid, 0.0) / max_work)

    def _importance_sample_clients(
        self,
        available_clients: List[ClientProxy],
        sample_size: int,
    ) -> List[ClientProxy]:
        # FedFS probability:
        # Pk proportional to 1 - wk + epsilon
        weights = []

        for client in available_clients:
            cid = client.cid
            wk = self._get_wk(cid)
            weight = max(0.0, 1.0 - wk + self.epsilon)
            weights.append(weight)

        weights = np.array(weights, dtype=np.float64)

        if weights.sum() <= 0.0:
            probs = np.ones(len(available_clients), dtype=np.float64) / len(available_clients)
        else:
            probs = weights / weights.sum()

        sample_size = min(sample_size, len(available_clients))

        selected_indices = np.random.choice(
            len(available_clients),
            size=sample_size,
            replace=False,
            p=probs,
        )

        return [available_clients[i] for i in selected_indices]

    # This works before the start of each training round.
    def configure_fit(self, server_round, parameters, client_manager):
        available_clients = list(client_manager.all().values())
        num_available_clients = len(available_clients)
        sample_size = self._num_fit_clients(num_available_clients)

        config = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)

        round_type = self._fedfs_round_type(server_round)
        fit_deadline_sec = self._fedfs_deadline(server_round)

        # Send FedFS deadline to the selected clients.
        config["fedfs_round_type"] = round_type
        config["fit_deadline_sec"] = fit_deadline_sec

        fit_ins = FitIns(parameters, config)

        # FedFS uses importance sampling based on previous contribution wk.
        # Clients with no history have wk = 0, so they naturally receive higher probability.
        sampled_clients = self._importance_sample_clients(
            available_clients=available_clients,
            sample_size=sample_size,
        )

        selected = [(client, fit_ins) for client in sampled_clients]

        return selected

    # This works after clients have completed local training.
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures,
    ) -> Tuple[Optional[Parameters], dict[str, Scalar]]:

        # Update FedFS work contribution history before or after aggregation.
        
        for client, fit_res in results:
            cid = client.cid

            actual_work = float(fit_res.metrics.get("num_examples", 0.0))
            max_possible_work = float(fit_res.metrics.get("max_possible_examples", 0.0))

            self.client_actual_work[cid] += actual_work
            self.client_max_work[cid] += max_possible_work
            self.client_work_ratio[cid] = self._get_wk(cid)

        
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round,
            results,
            failures,
        )

        return aggregated_parameters, aggregated_metrics
