"""
Proposed client selection strategy.
"""
import math
import random
from typing import Dict, List, Tuple,Union
from flwr.common import (
    FitIns,
    FitRes,
    GetPropertiesIns,
    Parameters,
    Scalar,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.fedavg import FedAvg
class ProposedStrategy(FedAvg):
    """Proposed data-aware and system-aware client selection strategy."""
    def __init__(
        self,
        *,
        selected_clients: int,
        selection_seed: int = 1234,
        capability_weight: float = 0.5,
        shortlist_size: int = 10,
        final_selector: str = "combined_score",
        selection_mode: str = "score",
        **kwargs,
        
    ):
        super().__init__(**kwargs)
        self.selected_clients = selected_clients
        self.selection_seed = selection_seed
        # ----------------------------------------------------
        # Storage for future stages
        # ----------------------------------------------------
        self.client_profiles: Dict[
            str,
            Dict[str, Scalar],
        ] = {}
        self.eligible_clients: Dict[
            str,
            ClientProxy,
        ] = {}
        # ----------------------------------------------------
        # Reference Bank
        # ----------------------------------------------------
        # Stores the data profiles of clients that have
        # successfully contributed to the global model.
        #
        # Key   : logical client CID
        # Value : statistical data profile
        self.reference_bank: Dict[
            str,
            Dict[str, Scalar],
        ] = {}
        # ----------------------------------------------------
        # Novelty scores for the current round
        # ----------------------------------------------------
        self.novelty_scores: Dict[
            str,
            Dict[str, Scalar],
        ] = {}
         # ----------------------------------------------------
        # System capability scores for the current round
        # ----------------------------------------------------

        self.system_capability_scores: Dict[
            str,
            Dict[str, object],
        ] = {}

        # ----------------------------------------------------
        # Combined selection score configuration
        # ----------------------------------------------------

        self.capability_weight = float(
            capability_weight
        )

        if not 0.0 <= self.capability_weight <= 1.0:
            raise ValueError(
                "capability_weight must be between "
                "0.0 and 1.0"
            )

        self.combined_scores: Dict[
            str,
            Dict[str, object],
        ] = {}

        # ----------------------------------------------------
        # Selection mode
        # ----------------------------------------------------

        valid_selection_modes = {
            "score",
            "eligibility_score",
        }

        if selection_mode not in valid_selection_modes:
            raise ValueError(
                "selection_mode must be either "
                "'score' or 'eligibility_score'"
            )

        self.selection_mode = selection_mode
         # ----------------------------------------------------
        # Final selector
        # ----------------------------------------------------

        valid_final_selectors = {
            "combined_score",
            "fedavg",
        }

        if final_selector not in valid_final_selectors:
            raise ValueError(
                "final_selector must be either "
                "'combined_score' or 'fedavg'"
            )

        self.final_selector = final_selector
        if shortlist_size <= 0:
            raise ValueError(
                "shortlist_size must be greater than 0"
            )

        if selected_clients <= 0:
            raise ValueError(
                "selected_clients must be greater than 0"
            )

        if selected_clients > shortlist_size:
            raise ValueError(
                "selected_clients cannot be greater "
                "than shortlist_size"
            )

        self.shortlist_size = int(
            shortlist_size
        )

        # ----------------------------------------------------
        # Client system history
        # ----------------------------------------------------
        # Stores historical system-performance information
        # for each logical client. History is built online
        # from actual participation. New clients start without
        # historical performance information.
        self.client_history: Dict[
            str,
            Dict[str, object],
        ] = {}
        # ----------------------------------------------------
        # Round participation history
        # ----------------------------------------------------
        # Maps Flower/Ray proxy CID -> logical client CID
        self.proxy_to_logical_cid: Dict[str, str] = {}
        # Clients selected for the current training round
        self.selected_proxy_cids: List[str] = []
        # Clients that successfully returned FitRes
        self.successful_proxy_cids: List[str] = []
        # Logical CIDs of successful clients
        self.successful_logical_cids: List[str] = []
    # ========================================================
    # Client system history helpers
    # ========================================================
    def _get_or_create_client_history(
        self,
        logical_cid: str,
    ) -> Dict[str, object]:
        """
        Return the history record for a logical client.

        If the client has never been selected before, create an
        empty cold-start history record.
        """
        if logical_cid not in self.client_history:
            self.client_history[logical_cid] = {
                "selected_count": 0,
                "successful_count": 0,
                "throughput_history": [],
            }
        return self.client_history[logical_cid]

    def _record_selected_clients(
        self,
        selected_clients: List[ClientProxy],
    ) -> None:
        """
        Record that clients were selected for local training.

        This happens before training starts so that a selected
        client is counted even if it later fails.
        """
        for client in selected_clients:
            proxy_cid = str(client.cid)
            logical_cid = self.proxy_to_logical_cid.get(
                proxy_cid,
                proxy_cid,
            )
            history = self._get_or_create_client_history(
                logical_cid
            )
            history["selected_count"] = (
                int(history["selected_count"]) + 1
            )

    def _update_client_history_after_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
    ) -> None:
        """
        Update client history using successful FitRes results.

        Only clients that successfully return FitRes appear in
        results. Successful participation and training throughput
        are therefore updated here.
        """
        print(
            f"\n"
            f"========================================\n"
            f"[ROUND {server_round}] "
            f"CLIENT HISTORY UPDATE\n"
            f"========================================"
        )
        for client_proxy, fit_res in results:
            proxy_cid = str(client_proxy.cid)
            logical_cid = self.proxy_to_logical_cid.get(
                proxy_cid,
                proxy_cid,
            )
            history = self._get_or_create_client_history(
                logical_cid
            )
            # --------------------------------------------
            # Successful participation
            # --------------------------------------------
            history["successful_count"] = (
                int(history["successful_count"]) + 1
            )
            # --------------------------------------------
            # Training throughput
            # --------------------------------------------
            throughput = fit_res.metrics.get(
                "samples_per_second"
            )
            if throughput is not None:
                throughput_history = history[
                    "throughput_history"
                ]
                if isinstance(throughput_history, list):
                    throughput_history.append(
                        float(throughput)
                    )
            # --------------------------------------------
            # Derived historical metrics
            # --------------------------------------------
            selected_count = int(
                history["selected_count"]
            )
            successful_count = int(
                history["successful_count"]
            )
            completion_rate = (
                successful_count / selected_count
                if selected_count > 0
                else None
            )
            throughput_values = history[
                "throughput_history"
            ]
            if (
                isinstance(throughput_values, list)
                and throughput_values
            ):
                average_throughput = (
                    sum(throughput_values)
                    / len(throughput_values)
                )
            else:
                average_throughput = None
            # --------------------------------------------
            # Print history
            # --------------------------------------------
            print(f"Logical CID={logical_cid}")
            print(
                f"  Selected count: "
                f"{selected_count}"
            )
            print(
                f"  Successful count: "
                f"{successful_count}"
            )
            if completion_rate is not None:
                print(
                    f"  Completion rate: "
                    f"{completion_rate:.4f}"
                )
            else:
                print("  Completion rate: N/A")
            if average_throughput is not None:
                print(
                    f"  Historical throughput: "
                    f"{average_throughput:.4f} "
                    f"samples/s"
                )
            else:
                print("  Historical throughput: N/A")

    # ========================================================
    # Collect pre-selection properties
    # ========================================================
    def _collect_client_properties(
        self,
        server_round: int,
        client_manager: ClientManager,
    ) -> Dict[str, Dict[str, Scalar]]:
        """
        Request pre-selection properties from all currently
        available clients.
        No local model training is performed in this step.
        """
        available_clients = client_manager.all()
        print(
            f"\n"
            f"========================================\n"
            f"[ROUND {server_round}] "
            f"PRE-SELECTION PROFILING\n"
            f"========================================"
        )
        print(
            f"Available clients: "
            f"{len(available_clients)}"
        )
        profiles: Dict[
            str,
            Dict[str, Scalar],
        ] = {}
        for cid, client in available_clients.items():
            try:
                request = GetPropertiesIns(
                    config={
                        "server_round":
                            int(server_round),
                    }
                )
                response = client.get_properties(
                    request,
                    timeout=None,
                    group_id=server_round,
                )
                properties = response.properties
                profiles[cid] = properties
                logical_cid = str(
                    properties.get(
                       "cid",
                        cid,
                    )
                )
                self.proxy_to_logical_cid[cid] = logical_cid
                print(
                    f"[ROUND {server_round}] "
                    f"Proxy CID={cid}, "
                    f"Logical CID={logical_cid}, "
                    f"eligible="
                    f"{properties.get('eligible')}"
                )
            except Exception as exc:
                print(
                    f"[ROUND {server_round}] "
                    f"Client {cid}: "
                    f"property request failed: "
                    f"{exc}"
                )
        print(
            f"Profiles received: "
            f"{len(profiles)}"
        )
        return profiles
    # ========================================================
    # Eligibility filtering
    # ========================================================
    def _filter_eligible_clients(
        self,
        server_round: int,
        profiles: Dict[
            str,
            Dict[str, Scalar],
        ],
        client_manager: ClientManager,
    ) -> Dict[str, ClientProxy]:
        """
        Build the pool of clients that passed the local
        system eligibility check.
        """
        available_clients = client_manager.all()
        eligible_clients: Dict[
            str,
            ClientProxy,
        ] = {}
        deferred_clients: List[str] = []
        for cid, properties in profiles.items():
            is_eligible = bool(
                properties.get(
                    "eligible",
                    False,
                )
            )
            if (
                is_eligible
                and cid in available_clients
            ):
                eligible_clients[cid] = (
                    available_clients[cid]
                )
            else:
                deferred_clients.append(cid)
        print(
            f"[ROUND {server_round}] "
            f"Eligible clients: "
            f"{len(eligible_clients)}"
        )
        print(
            f"[ROUND {server_round}] "
            f"Deferred clients: "
            f"{len(deferred_clients)}"
        )
        if deferred_clients:
            print(
                f"[ROUND {server_round}] "
                f"Deferred CIDs: "
                f"{deferred_clients}"
            )
        return eligible_clients

    # ========================================================
    # Candidate pool construction
    # ========================================================

    def _build_candidate_pool(
        self,
        server_round: int,
        profiles: Dict[
            str,
            Dict[str, Scalar],
        ],
        client_manager: ClientManager,
        eligible_clients: Dict[
            str,
            ClientProxy,
        ],
    ) -> Dict[str, ClientProxy]:
        """
        Build the candidate pool according to the configured
        experimental selection mode.

        score:
            All available clients with successfully collected
            profiles remain candidates. Eligibility is recorded
            for analysis but is not used as a hard filter.

        eligibility_score:
            Only clients that passed the local eligibility check
            remain candidates.
        """

        available_clients = client_manager.all()

        # ----------------------------------------------------
        # Experiment A:
        # Score-Based Selection
        # ----------------------------------------------------

        if self.selection_mode == "score":

            candidate_clients = {
                proxy_cid: available_clients[proxy_cid]
                for proxy_cid in profiles
                if proxy_cid in available_clients
            }

        # ----------------------------------------------------
        # Experiment B:
        # Eligibility + Score Selection
        # ----------------------------------------------------

        else:

            candidate_clients = dict(
                eligible_clients
            )

        print(
            f"\n"
            f"========================================\n"
            f"[ROUND {server_round}] "
            f"CANDIDATE POOL\n"
            f"========================================"
        )

        print(
            f"Selection mode: "
            f"{self.selection_mode}"
        )

        print(
            f"Profiled clients: "
            f"{len(profiles)}"
        )

        print(
            f"Eligible clients: "
            f"{len(eligible_clients)}"
        )

        print(
            f"Candidate clients: "
            f"{len(candidate_clients)}"
        )

        candidate_logical_cids = [
            self.proxy_to_logical_cid.get(
                proxy_cid,
                proxy_cid,
            )
            for proxy_cid in candidate_clients
        ]

        print(
            f"Candidate logical CIDs: "
            f"{candidate_logical_cids}"
        )

        return candidate_clients

    # ========================================================
    # System capability normalization
    # ========================================================

    @staticmethod
    def _normalize_higher_is_better(
        value: float,
        minimum: float,
        maximum: float,
    ) -> float:
        """
        Round-wise Min-Max normalization for metrics where
        a larger value represents better system capability.

        Returns a value in [0, 1].

        If all candidates have the same value, the metric
        cannot distinguish between them, so all receive 0.5.
        """

        if math.isclose(maximum, minimum):
            return 0.5

        normalized = (
            (value - minimum)
            / (maximum - minimum)
        )

        return max(
            0.0,
            min(1.0, normalized),
        )


    @staticmethod
    def _normalize_lower_is_better(
        value: float,
        minimum: float,
        maximum: float,
    ) -> float:
        """
        Round-wise Min-Max normalization for metrics where
        a smaller value represents better system capability.

        Used for latency.

        Returns a value in [0, 1].

        If all candidates have the same value, all receive 0.5.
        """

        if math.isclose(maximum, minimum):
            return 0.5

        normalized = (
            (maximum - value)
            / (maximum - minimum)
        )

        return max(
            0.0,
            min(1.0, normalized),
        )


    def _compute_system_capability_scores(
        self,
        server_round: int,
        profiles: Dict[
            str,
            Dict[str, Scalar],
        ],
        eligible_clients: Dict[
            str,
            ClientProxy,
        ],
    ) -> Dict[str, Dict[str, object]]:
        """
        Compute round-wise normalized system capability scores
        for all eligible clients.

        Dimensions:

        Compute:
            Historical training throughput.

        Memory:
            Available RAM + free GPU memory.

        Network:
            Upload bandwidth + download bandwidth + latency.

        Reliability:
            Historical completion rate.

        Historical dimensions are excluded for cold-start
        clients until real history becomes available.
        """

        capability_scores: Dict[
            str,
            Dict[str, object],
        ] = {}

        print(
            f"\n"
            f"========================================\n"
            f"[ROUND {server_round}] "
            f"SYSTEM CAPABILITY ANALYSIS\n"
            f"========================================"
        )

        if not eligible_clients:
            print("No eligible clients.")
            return capability_scores

        # ----------------------------------------------------
        # Collect current non-historical metrics
        # ----------------------------------------------------

        ram_values: List[float] = []
        gpu_values: List[float] = []
        upload_values: List[float] = []
        download_values: List[float] = []
        latency_values: List[float] = []

        for proxy_cid in eligible_clients:

            properties = profiles.get(proxy_cid)

            if properties is None:
                continue

            ram_values.append(
                float(
                    properties.get(
                        "available_ram_mb",
                        0.0,
                    )
                )
            )

            gpu_values.append(
                float(
                    properties.get(
                        "gpu_free_memory_mb",
                        0.0,
                    )
                )
            )

            upload_values.append(
                float(
                    properties.get(
                        "upload_bandwidth_mbps",
                        0.0,
                    )
                )
            )

            download_values.append(
                float(
                    properties.get(
                        "download_bandwidth_mbps",
                        0.0,
                    )
                )
            )

            latency_values.append(
                float(
                    properties.get(
                        "latency_ms",
                        0.0,
                    )
                )
            )

        if not ram_values:
            print(
                "No valid system capability profiles."
            )
            return capability_scores

        # ----------------------------------------------------
        # Round-wise Min-Max ranges
        # ----------------------------------------------------

        ram_min = min(ram_values)
        ram_max = max(ram_values)

        gpu_min = min(gpu_values)
        gpu_max = max(gpu_values)

        upload_min = min(upload_values)
        upload_max = max(upload_values)

        download_min = min(download_values)
        download_max = max(download_values)

        latency_min = min(latency_values)
        latency_max = max(latency_values)

        # ----------------------------------------------------
        # Historical throughput range
        #
        # Only clients with actual throughput history are
        # included in this range.
        # ----------------------------------------------------

        throughput_by_client: Dict[
            str,
            float,
        ] = {}

        for proxy_cid in eligible_clients:

            logical_cid = (
                self.proxy_to_logical_cid.get(
                    proxy_cid,
                    proxy_cid,
                )
            )

            history = self.client_history.get(
                logical_cid
            )

            if history is None:
                continue

            throughput_history = history.get(
                "throughput_history",
                [],
            )

            if (
                isinstance(throughput_history, list)
                and throughput_history
            ):
                average_throughput = (
                    sum(
                        float(value)
                        for value in throughput_history
                    )
                    / len(throughput_history)
                )

                throughput_by_client[
                    proxy_cid
                ] = average_throughput

        if throughput_by_client:
            throughput_min = min(
                throughput_by_client.values()
            )

            throughput_max = max(
                throughput_by_client.values()
            )

        else:
            throughput_min = None
            throughput_max = None

        # ----------------------------------------------------
        # Compute each client's capability dimensions
        # ----------------------------------------------------

        for proxy_cid in eligible_clients:

            properties = profiles.get(
                proxy_cid
            )

            if properties is None:
                continue

            logical_cid = (
                self.proxy_to_logical_cid.get(
                    proxy_cid,
                    proxy_cid,
                )
            )

            # ------------------------------------------------
            # Memory
            # ------------------------------------------------

            ram = float(
                properties.get(
                    "available_ram_mb",
                    0.0,
                )
            )

            gpu_memory = float(
                properties.get(
                    "gpu_free_memory_mb",
                    0.0,
                )
            )

            ram_score = (
                self._normalize_higher_is_better(
                    ram,
                    ram_min,
                    ram_max,
                )
            )

            gpu_score = (
                self._normalize_higher_is_better(
                    gpu_memory,
                    gpu_min,
                    gpu_max,
                )
            )

            memory_score = (
                ram_score
                + gpu_score
            ) / 2.0

            # ------------------------------------------------
            # Network
            # ------------------------------------------------

            upload = float(
                properties.get(
                    "upload_bandwidth_mbps",
                    0.0,
                )
            )

            download = float(
                properties.get(
                    "download_bandwidth_mbps",
                    0.0,
                )
            )

            latency = float(
                properties.get(
                    "latency_ms",
                    0.0,
                )
            )

            upload_score = (
                self._normalize_higher_is_better(
                    upload,
                    upload_min,
                    upload_max,
                )
            )

            download_score = (
                self._normalize_higher_is_better(
                    download,
                    download_min,
                    download_max,
                )
            )

            latency_score = (
                self._normalize_lower_is_better(
                    latency,
                    latency_min,
                    latency_max,
                )
            )

            network_score = (
                upload_score
                + download_score
                + latency_score
            ) / 3.0

            # ------------------------------------------------
            # Compute
            # ------------------------------------------------

            compute_score = None
            historical_throughput = None

            if proxy_cid in throughput_by_client:

                historical_throughput = (
                    throughput_by_client[
                        proxy_cid
                    ]
                )

                if (
                    throughput_min is not None
                    and throughput_max is not None
                ):
                    compute_score = (
                        self._normalize_higher_is_better(
                            historical_throughput,
                            throughput_min,
                            throughput_max,
                        )
                    )

            # ------------------------------------------------
            # Reliability
            # ------------------------------------------------

            reliability_score = None
            completion_rate = None

            history = self.client_history.get(
                logical_cid
            )

            if history is not None:

                selected_count = int(
                    history.get(
                        "selected_count",
                        0,
                    )
                )

                successful_count = int(
                    history.get(
                        "successful_count",
                        0,
                    )
                )

                if selected_count > 0:

                    completion_rate = (
                        successful_count
                        / selected_count
                    )

                    reliability_score = max(
                        0.0,
                        min(
                            1.0,
                            completion_rate,
                        ),
                    )

            # ------------------------------------------------
            # Available-dimension averaging
            # ------------------------------------------------

            available_dimension_scores = [
                memory_score,
                network_score,
            ]

            if compute_score is not None:
                available_dimension_scores.append(
                    compute_score
                )

            if reliability_score is not None:
                available_dimension_scores.append(
                    reliability_score
                )

            system_capability_score = (
                sum(available_dimension_scores)
                / len(available_dimension_scores)
            )

            # ------------------------------------------------
            # Store results
            # ------------------------------------------------

            capability_scores[proxy_cid] = {

                "logical_cid":
                    logical_cid,

                "ram_score":
                    float(ram_score),

                "gpu_score":
                    float(gpu_score),

                "memory_score":
                    float(memory_score),

                "upload_score":
                    float(upload_score),

                "download_score":
                    float(download_score),

                "latency_score":
                    float(latency_score),

                "network_score":
                    float(network_score),

                "historical_throughput":
                    (
                        float(historical_throughput)
                        if historical_throughput
                        is not None
                        else None
                    ),

                "compute_score":
                    (
                        float(compute_score)
                        if compute_score
                        is not None
                        else None
                    ),

                "completion_rate":
                    (
                        float(completion_rate)
                        if completion_rate
                        is not None
                        else None
                    ),

                "reliability_score":
                    (
                        float(reliability_score)
                        if reliability_score
                        is not None
                        else None
                    ),

                "system_capability_score":
                    float(
                        system_capability_score
                    ),
            }

            # ------------------------------------------------
            # Print results
            # ------------------------------------------------

            print(
                f"Logical CID={logical_cid}"
            )

            print(
                f"  Memory score: "
                f"{memory_score:.4f}"
            )

            print(
                f"  Network score: "
                f"{network_score:.4f}"
            )

            if compute_score is None:
                print(
                    "  Compute score: N/A "
                    "(cold start)"
                )
            else:
                print(
                    f"  Compute score: "
                    f"{compute_score:.4f}"
                )

            if reliability_score is None:
                print(
                    "  Reliability score: N/A "
                    "(cold start)"
                )
            else:
                print(
                    f"  Reliability score: "
                    f"{reliability_score:.4f}"
                )

            print(
                f"  System capability score: "
                f"{system_capability_score:.4f}"
            )

        return capability_scores



    
    # ========================================================
    # Initialization selection
    # ========================================================
    def _initial_selection(
        self,
        server_round: int,
        eligible_clients: Dict[
            str,
            ClientProxy,
        ],
    ) -> List[ClientProxy]:
        """
        Select clients during the initialization stage.
        V1 uses deterministic random sampling among eligible
        clients because no training-history reference exists
        yet.
        This is NOT the final proposed selection mechanism.
        """
        candidates = list(
            eligible_clients.values()
        )
        if len(candidates) < self.selected_clients:
            print(
                f"[ROUND {server_round}] "
                f"Not enough eligible clients. "
                f"Required={self.selected_clients}, "
                f"Eligible={len(candidates)}"
            )
            return []
        rng = random.Random(
            self.selection_seed
            + server_round
        )
        selected = rng.sample(
            candidates,
            self.selected_clients,
        )
        selected_cids = [
            client.cid
            for client in selected
        ]
        print(
            f"[ROUND {server_round}] "
            f"Initialization selection: "
            f"{selected_cids}"
        )
        return selected
    # ========================================================
    # Flower configure_fit
    # ========================================================
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
        """
        Configure clients selected for local training.
        """
        # ----------------------------------------------------
        # 1. Collect pre-selection information
        # ----------------------------------------------------
        profiles = (
            self._collect_client_properties(
                server_round,
                client_manager,
            )
        )
        self.client_profiles = profiles
        # ----------------------------------------------------
        # 2. Eligibility filtering
        # ----------------------------------------------------
        eligible_clients = (
            self._filter_eligible_clients(
                server_round,
                profiles,
                client_manager,
            )
        )
        self.eligible_clients = (
            eligible_clients
        )

        # ----------------------------------------------------
        # 3. Build experimental candidate pool
        # ----------------------------------------------------

        candidate_clients = (
            self._build_candidate_pool(
                server_round,
                profiles,
                client_manager,
                eligible_clients,
            )
        )

        # ----------------------------------------------------
        # 3. System capability analysis
        # ----------------------------------------------------

        self.system_capability_scores = (
            self._compute_system_capability_scores(
                server_round,
                profiles,
                candidate_clients,
            )
        )

        # ----------------------------------------------------
        # 4. Novelty analysis
        # ----------------------------------------------------
        self.novelty_scores = (
            self._compute_novelty_scores(
                server_round,
                profiles,
                candidate_clients,
            )
        )
        # ----------------------------------------------------
        # 5. Initialization selection
        # ----------------------------------------------------
        selected_clients = (
            self._initial_selection(
                server_round,
                eligible_clients,
            )
        )
        self.selected_proxy_cids = [
            client.cid
            for client in selected_clients
        ]
        selected_logical_cids = [
            self.proxy_to_logical_cid.get(
                client.cid,
                client.cid,
            )
            for client in selected_clients
        ]
        print(
            f"[ROUND {server_round}] "
            f"Selected logical clients: "
            f"{selected_logical_cids}"
        )
        if not selected_clients:
            print(
                f"[ROUND {server_round}] "
                f"No clients selected."
            )
            return []
        # ----------------------------------------------------
        # 5. Record selected clients in system history
        # ----------------------------------------------------
        self._record_selected_clients(
            selected_clients
        )
        # ----------------------------------------------------
        # 6. Build Flower FitIns
        # ----------------------------------------------------
        config = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(
                server_round
            )
        fit_ins = FitIns(
            parameters,
            config,
        )
        # ----------------------------------------------------
        # 7. Only selected clients receive fit instructions
        # ----------------------------------------------------
        return [
            (
                client,
                fit_ins,
            )
            for client in selected_clients
        ]
    def _extract_data_profile(
        self,
        properties: Dict[str, Scalar],
    ) -> Dict[str, Scalar]:
        """
        Extract only the statistical data descriptors needed
        for the Reference Bank.
        System capability information is intentionally excluded.
        """
        data_profile: Dict[str, Scalar] = {}
    # ----------------------------------------------------
    # Number of local samples
    # Stored as metadata, not currently used in novelty.
    # ----------------------------------------------------
        if "num_samples" in properties:
            data_profile["num_samples"] = properties["num_samples"]
    # ----------------------------------------------------
    # Class distribution
    # ----------------------------------------------------
        for class_id in range(10):
            key = f"class_{class_id}_ratio"
            if key in properties:
                data_profile[key] = properties[key]
    # ----------------------------------------------------
    # RGB mean
    # ----------------------------------------------------
        for channel in ["r", "g", "b"]:
            key = f"rgb_mean_{channel}"
            if key in properties:
                data_profile[key] = properties[key]
    # ----------------------------------------------------
    # RGB standard deviation
    # ----------------------------------------------------
        for channel in ["r", "g", "b"]:
            key = f"rgb_std_{channel}"
            if key in properties:
                data_profile[key] = properties[key]
        return data_profile
    def _update_reference_bank(
        self,
        server_round: int,
    ) -> None:
        """
        Update the Reference Bank using only clients that
        successfully completed local training in this round.
        """
        print(
            f"\n"
            f"========================================\n"
            f"[ROUND {server_round}] "
            f"REFERENCE BANK UPDATE\n"
            f"========================================"
        )
        added_clients: List[str] = []
        for proxy_cid in self.successful_proxy_cids:
            logical_cid = self.proxy_to_logical_cid.get(
                proxy_cid,
                proxy_cid,
            )
        # ------------------------------------------------
        # Do not add the same logical client twice
        # ------------------------------------------------
            if logical_cid in self.reference_bank:
                print(
                    f"Logical CID={logical_cid}: "
                    f"already represented in Reference Bank."
                )
                continue
        # ------------------------------------------------
        # Retrieve the properties collected before training
        # ------------------------------------------------
            properties = self.client_profiles.get(proxy_cid)
            if properties is None:
                print(
                    f"Logical CID={logical_cid}: "
                    f"profile not found; not added."
                )
                continue
        # ------------------------------------------------
        # Extract data descriptors only
        # ------------------------------------------------
            data_profile = self._extract_data_profile(
                properties
            )
        # ------------------------------------------------
        # Store the profile
        # ------------------------------------------------
            self.reference_bank[logical_cid] = (
                data_profile
            )
            added_clients.append(logical_cid)
            print(
                f"Logical CID={logical_cid}: "
                f"added to Reference Bank."
            )
        print(
            f"New references added: "
            f"{len(added_clients)}"
        )
        print(
            f"Reference Bank size: "
            f"{len(self.reference_bank)}"
        )
        print(
            f"Reference logical CIDs: "
            f"{list(self.reference_bank.keys())}"
        )
    def _compute_label_distance(
        self,
        candidate_profile: Dict[str, Scalar],
        reference_profile: Dict[str, Scalar],
    ) -> float:
        """
        Compute Jensen-Shannon distance between the class
        distributions of a candidate and a reference profile.
        """
        candidate_distribution = [
            float(
                candidate_profile.get(
                    f"class_{class_id}_ratio",
                    0.0,
                 )
            )
            for class_id in range(10)
        ]
        reference_distribution = [
            float(
                reference_profile.get(
                    f"class_{class_id}_ratio",
                    0.0,
                )
            )
            for class_id in range(10)
        ]
        midpoint_distribution = [
            (candidate_value + reference_value) / 2.0
            for candidate_value, reference_value in zip(
                candidate_distribution,
                reference_distribution,
            )
        ]
        def _kl_divergence(
            distribution_p: List[float],
            distribution_q: List[float],
        ) -> float:
            divergence = 0.0
            for p_value, q_value in zip(
                distribution_p,
                distribution_q,
            ):
                if p_value > 0.0:
                    divergence += (
                        p_value
                        * math.log2(
                            p_value / q_value
                        )
                    )
            return divergence
        js_divergence = 0.5 * (
            _kl_divergence(
                candidate_distribution,
                midpoint_distribution,
            )
            + _kl_divergence(
                reference_distribution,
                midpoint_distribution,
            )
        )
        js_divergence = max(
            js_divergence,
            0.0,
       )
        return math.sqrt(
            js_divergence
        )
    def _compute_visual_distance(
        self,
        candidate_profile: Dict[str, Scalar],
        reference_profile: Dict[str, Scalar],
    ) -> float:
        """
        Compute visual statistical distance using RGB mean
        and RGB standard deviation.
        """
        channels = [
            "r",
            "g",
            "b",
        ]
        candidate_mean = [
            float(
                candidate_profile.get(
                    f"rgb_mean_{channel}",
                     0.0,
                )
            )
            for channel in channels
        ]
        reference_mean = [
            float(
                reference_profile.get(
                    f"rgb_mean_{channel}",
                    0.0,
                )
            )
            for channel in channels
        ]
        candidate_std = [
            float(
                candidate_profile.get(
                    f"rgb_std_{channel}",
                    0.0,
                )
            )
            for channel in channels
        ]
        reference_std = [
            float(
                reference_profile.get(
                    f"rgb_std_{channel}",
                    0.0,
                )
            )
            for channel in channels
       ]
        mean_distance = math.sqrt(
            sum(
                (
                    candidate_value
                    - reference_value
                 ) ** 2
                for candidate_value, reference_value
                in zip(
                    candidate_mean,
                    reference_mean,
                )
            )
        )
        std_distance = math.sqrt(
            sum(
                (
                    candidate_value
                    - reference_value
                ) ** 2
                for candidate_value, reference_value
                in zip(
                    candidate_std,
                    reference_std,
                )
            )
        )
        visual_distance = (
            mean_distance
            + std_distance
        ) / 2.0
        return visual_distance
    def _compute_novelty_scores(
        self,
        server_round: int,
        profiles: Dict[
            str,
            Dict[str, Scalar],
        ],
        eligible_clients: Dict[
            str,
            ClientProxy,
        ],
    ) -> Dict[str, Dict[str, Scalar]]:
        """
        Compute novelty information for eligible clients
        relative to the current Reference Bank.
        Novelty is calculated for observation only in this
        stage and is not yet used for client selection.
        """
        novelty_scores: Dict[
            str,
            Dict[str, Scalar],
        ] = {}
        print(
            f"\n"
            f"========================================\n"
            f"[ROUND {server_round}] "
            f"NOVELTY ANALYSIS\n"
            f"========================================"
        )
        if not self.reference_bank:
            print(
                "Reference Bank is empty. "
                "Novelty cannot be computed yet."
            )
            return novelty_scores
        for proxy_cid in eligible_clients:
            properties = profiles.get(
                proxy_cid
            )
            if properties is None:
                continue
            logical_cid = (
                self.proxy_to_logical_cid.get(
                    proxy_cid,
                    proxy_cid,
                )
            )
            candidate_profile = (
                self._extract_data_profile(
                    properties
                )
            )
            minimum_label_distance = float(
                "inf"
            )
            minimum_visual_distance = float(
                "inf"
            )
            closest_label_reference = None
            closest_visual_reference = None
            for (
                reference_cid,
                reference_profile,
            ) in self.reference_bank.items():
                label_distance = (
                    self._compute_label_distance(
                        candidate_profile,
                        reference_profile,
                    )
                )
                visual_distance = (
                    self._compute_visual_distance(
                        candidate_profile,
                        reference_profile,
                    )
                )
                if (
                    label_distance
                    < minimum_label_distance
                ):
                    minimum_label_distance = (
                        label_distance
                    )
                    closest_label_reference = (
                        reference_cid
                    )
                if (
                    visual_distance
                    < minimum_visual_distance
                ):
                    minimum_visual_distance = (
                        visual_distance
                    )
                    closest_visual_reference = (
                        reference_cid
                    )
            novelty_scores[proxy_cid] = {
                "logical_cid": logical_cid,
                "label_novelty": (
                    minimum_label_distance
                ),
                "visual_novelty": (
                    minimum_visual_distance
                ),
                "closest_label_reference": (
                    str(closest_label_reference)
                ),
                "closest_visual_reference": (
                    str(closest_visual_reference)
                ),
            }
            print(
                f"Logical CID={logical_cid}"
            )
            print(
                f"  Label novelty: "
                f"{minimum_label_distance:.6f}"
            )
            print(
                f"  Closest label reference: "
                f"{closest_label_reference}"
            )
            print(
                f"  Visual novelty: "
                f"{minimum_visual_distance:.6f}"
            )
            print(
                f"  Closest visual reference: "
                f"{closest_visual_reference}"
            )
        return novelty_scores  

    # ========================================================
    # Final data novelty score
    # ========================================================

    def _compute_final_novelty_scores(
        self,
        server_round: int,
        novelty_scores: Dict[
            str,
            Dict[str, float],
        ],
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute one final novelty score for each client.

        The two novelty dimensions are normalized separately
        across the current candidate pool:

            1. Label-distribution novelty
            2. Visual-statistics novelty

        The final novelty score is the equally weighted
        average of the two normalized dimensions.

        FinalNovelty =
            0.5 * NormalizedLabelNovelty
            + 0.5 * NormalizedVisualNovelty
        """

        final_scores: Dict[
            str,
            Dict[str, float],
        ] = {}

        print(
            f"\n"
            f"========================================\n"
            f"[ROUND {server_round}] "
            f"FINAL NOVELTY ANALYSIS\n"
            f"========================================"
        )

        if not novelty_scores:
            print(
                "No novelty scores available."
            )
            return final_scores

        # ----------------------------------------------------
        # Collect raw novelty dimensions
        # ----------------------------------------------------

        label_values: Dict[str, float] = {}
        visual_values: Dict[str, float] = {}

        for proxy_cid, info in novelty_scores.items():

            if "label_novelty" not in info:
                continue

            if "visual_novelty" not in info:
                continue

            label_values[proxy_cid] = float(
                info["label_novelty"]
            )

            visual_values[proxy_cid] = float(
                info["visual_novelty"]
            )

        if not label_values:
            print(
                "No complete novelty information available."
            )
            return final_scores

        # ----------------------------------------------------
        # Round-wise ranges
        # ----------------------------------------------------

        label_min = min(
            label_values.values()
        )
        label_max = max(
            label_values.values()
        )

        visual_min = min(
            visual_values.values()
        )
        visual_max = max(
            visual_values.values()
        )

        # ----------------------------------------------------
        # Normalize each dimension separately
        # ----------------------------------------------------

        for proxy_cid in label_values:

            raw_label = label_values[
                proxy_cid
            ]

            raw_visual = visual_values[
                proxy_cid
            ]

            normalized_label = (
                self._normalize_higher_is_better(
                    raw_label,
                    label_min,
                    label_max,
                )
            )

            normalized_visual = (
                self._normalize_higher_is_better(
                    raw_visual,
                    visual_min,
                    visual_max,
                )
            )

            # ------------------------------------------------
            # Equal contribution from both novelty dimensions
            # ------------------------------------------------

            final_novelty = (
                0.5 * normalized_label
                + 0.5 * normalized_visual
            )

            final_scores[proxy_cid] = {
                "raw_label_novelty":
                    raw_label,

                "raw_visual_novelty":
                    raw_visual,

                "normalized_label_novelty":
                    normalized_label,

                "normalized_visual_novelty":
                    normalized_visual,

                "novelty_score":
                    float(final_novelty),
            }

            print(
                f"Client {proxy_cid}"
            )

            print(
                f"  Raw label novelty: "
                f"{raw_label:.6f}"
            )

            print(
                f"  Raw visual novelty: "
                f"{raw_visual:.6f}"
            )

            print(
                f"  Normalized label novelty: "
                f"{normalized_label:.4f}"
            )

            print(
                f"  Normalized visual novelty: "
                f"{normalized_visual:.4f}"
            )

            print(
                f"  Final novelty score: "
                f"{final_novelty:.4f}"
            )

        return final_scores

    # ========================================================
    # Combined capability + novelty score
    # ========================================================

    def _compute_combined_scores(
        self,
        server_round: int,
        system_capability_scores: Dict[
            str,
            Dict[str, object],
        ],
        final_novelty_scores: Dict[
            str,
            Dict[str, float],
        ],
    ) -> Dict[str, Dict[str, object]]:
        """
        Combine System Capability and final Data Novelty.

        When final novelty is unavailable, for example when
        the Reference Bank is empty during cold start,
        selection is based only on System Capability.

        Otherwise:

            CombinedScore =
                alpha * SystemCapability
                + (1 - alpha) * Novelty
        """

        combined_scores: Dict[
            str,
            Dict[str, object],
        ] = {}

        print(
            f"\n"
            f"========================================\n"
            f"[ROUND {server_round}] "
            f"COMBINED SELECTION SCORE\n"
            f"========================================"
        )

        if not system_capability_scores:
            print(
                "No System Capability scores available."
            )
            return combined_scores

        alpha = self.capability_weight
        novelty_weight = 1.0 - alpha

        for (
            proxy_cid,
            capability_info,
        ) in system_capability_scores.items():

            logical_cid = str(
                capability_info.get(
                    "logical_cid",
                    proxy_cid,
                )
            )

            capability_score = float(
                capability_info[
                    "system_capability_score"
                ]
            )

            # ------------------------------------------------
            # Novelty is available
            # ------------------------------------------------

            novelty_info = final_novelty_scores.get(
                proxy_cid
            )

            if (
                novelty_info is not None
                and "novelty_score" in novelty_info
            ):

                novelty_score = float(
                    novelty_info[
                        "novelty_score"
                    ]
                )

                combined_score = (
                    alpha * capability_score
                    + novelty_weight * novelty_score
                )

                novelty_available = True

            # ------------------------------------------------
            # Cold start / no Reference Bank
            # ------------------------------------------------

            else:

                novelty_score = None

                combined_score = (
                    capability_score
                )

                novelty_available = False

            # ------------------------------------------------
            # Store result
            # ------------------------------------------------

            combined_scores[proxy_cid] = {
                "logical_cid":
                    logical_cid,

                "system_capability_score":
                    capability_score,

                "novelty_score":
                    novelty_score,

                "capability_weight":
                    alpha,

                "novelty_weight":
                    novelty_weight,

                "novelty_available":
                    novelty_available,

                "combined_score":
                    float(combined_score),
            }

            # ------------------------------------------------
            # Logging
            # ------------------------------------------------

            print(
                f"Logical CID={logical_cid}"
            )

            print(
                f"  System capability: "
                f"{capability_score:.4f}"
            )

            if novelty_score is None:

                print(
                    "  Novelty score: N/A"
                )

                print(
                    "  Selection basis: "
                    "System Capability only"
                )

            else:

                print(
                    f"  Novelty score: "
                    f"{novelty_score:.4f}"
                )

                print(
                    f"  Weights: "
                    f"Capability={alpha:.2f}, "
                    f"Novelty={novelty_weight:.2f}"
                )

            print(
                f"  Combined score: "
                f"{combined_score:.4f}"
            )

        return combined_scores

    # ========================================================
    # Combined-score shortlist
    # ========================================================

    def _select_top_clients_by_combined_score(
        self,
        server_round: int,
        candidate_clients: Dict[
            str,
            ClientProxy,
        ],
        combined_scores: Dict[
            str,
            Dict[str, object],
        ],
    ) -> Dict[str, ClientProxy]:
        """
        Rank candidate clients by Combined Score and return
        the highest-scoring clients as a shortlist.

        This function performs pre-selection only.

        It does NOT determine which clients will necessarily
        train the model. A later final-selection stage can use
        the shortlist with:

            1. Combined-score Top-K
            2. FedAvg-style random sampling
            3. Oort
        """

        print(
            f"\n"
            f"========================================\n"
            f"[ROUND {server_round}] "
            f"COMBINED-SCORE SHORTLIST\n"
            f"========================================"
        )

        if not candidate_clients:
            print(
                "No candidate clients available."
            )
            return {}

        if not combined_scores:
            print(
                "No Combined Scores available."
            )
            return {}

        # ----------------------------------------------------
        # Keep only clients for which both the proxy and
        # Combined Score are available.
        # ----------------------------------------------------

        scored_candidates = [
            proxy_cid
            for proxy_cid in candidate_clients
            if proxy_cid in combined_scores
        ]

        if not scored_candidates:
            print(
                "No scored candidate clients available."
            )
            return {}

        # ----------------------------------------------------
        # Deterministic ranking
        #
        # Primary criterion:
        #   Higher Combined Score is better.
        #
        # Tie-break:
        #   proxy CID, to make repeated tests deterministic.
        # ----------------------------------------------------

        ranked_proxy_cids = sorted(
            scored_candidates,
            key=lambda proxy_cid: (
                -float(
                    combined_scores[
                        proxy_cid
                    ]["combined_score"]
                ),
                str(proxy_cid),
            ),
        )

        # ----------------------------------------------------
        # The shortlist cannot exceed the number of available
        # scored candidates.
        # ----------------------------------------------------

        actual_shortlist_size = min(
            self.shortlist_size,
            len(ranked_proxy_cids),
        )

        shortlisted_proxy_cids = (
            ranked_proxy_cids[
                :actual_shortlist_size
            ]
        )

        shortlist = {
            proxy_cid:
                candidate_clients[proxy_cid]
            for proxy_cid
            in shortlisted_proxy_cids
        }

        # ----------------------------------------------------
        # Logging
        # ----------------------------------------------------

        print(
            f"Candidate clients: "
            f"{len(candidate_clients)}"
        )

        print(
            f"Scored candidates: "
            f"{len(scored_candidates)}"
        )

        print(
            f"Configured shortlist size: "
            f"{self.shortlist_size}"
        )

        print(
            f"Actual shortlist size: "
            f"{len(shortlist)}"
        )

        print(
            "\nRanking:"
        )

        for rank, proxy_cid in enumerate(
            ranked_proxy_cids,
            start=1,
        ):

            info = combined_scores[
                proxy_cid
            ]

            logical_cid = str(
                info.get(
                    "logical_cid",
                    proxy_cid,
                )
            )

            score = float(
                info["combined_score"]
            )

            selected_for_shortlist = (
                proxy_cid
                in shortlist
            )

            status = (
                "SHORTLIST"
                if selected_for_shortlist
                else "OUT"
            )

            print(
                f"  Rank {rank}: "
                f"Logical CID={logical_cid}, "
                f"Combined={score:.4f}, "
                f"{status}"
            )

        return shortlist

    # ========================================================
    # Final client selection
    # ========================================================

    def _select_final_clients(
        self,
        server_round: int,
        shortlist: Dict[
            str,
            ClientProxy,
        ],
        combined_scores: Dict[
            str,
            Dict[str, object],
        ],
    ) -> Dict[str, ClientProxy]:
        """
        Select the final clients that will train the model.

        Supported final selectors:

        combined_score:
            Select the highest-scoring K clients from
            the Combined Score shortlist.

        fedavg:
            Randomly sample K clients from the
            Combined Score shortlist.
        """

        print(
            f"\n"
            f"========================================\n"
            f"[ROUND {server_round}] "
            f"FINAL CLIENT SELECTION\n"
            f"========================================"
        )

        if not shortlist:
            print(
                "No clients available in the shortlist."
            )
            return {}

        # ----------------------------------------------------
        # Number of clients that can actually be selected
        # ----------------------------------------------------

        final_count = min(
            self.selected_clients,
            len(shortlist),
        )

        print(
            f"Final selector: "
            f"{self.final_selector}"
        )

        print(
            f"Shortlist size: "
            f"{len(shortlist)}"
        )

        print(
            f"Requested final clients: "
            f"{self.selected_clients}"
        )

        print(
            f"Actual final clients: "
            f"{final_count}"
        )

        # ----------------------------------------------------
        # 1. Combined Score final selection
        # ----------------------------------------------------

        if self.final_selector == "combined_score":

            scored_shortlist = [
                proxy_cid
                for proxy_cid in shortlist
                if proxy_cid in combined_scores
            ]

            ranked_proxy_cids = sorted(
                scored_shortlist,
                key=lambda proxy_cid: (
                    -float(
                        combined_scores[
                            proxy_cid
                        ]["combined_score"]
                    ),
                    str(proxy_cid),
                ),
            )

            selected_proxy_cids = (
                ranked_proxy_cids[
                    :final_count
                ]
            )

        # ----------------------------------------------------
        # 2. FedAvg-style random selection
        # ----------------------------------------------------

        elif self.final_selector == "fedavg":

            candidates = list(
                shortlist.keys()
            )

            rng = random.Random(
                self.selection_seed
                + server_round
            )

            selected_proxy_cids = rng.sample(
                candidates,
                final_count,
            )

        # ----------------------------------------------------
        # Safety check
        # ----------------------------------------------------

        else:
            raise ValueError(
                f"Unsupported final selector: "
                f"{self.final_selector}"
            )

        # ----------------------------------------------------
        # Build final client dictionary
        # ----------------------------------------------------

        final_clients = {
            proxy_cid:
                shortlist[proxy_cid]
            for proxy_cid
            in selected_proxy_cids
        }

        # ----------------------------------------------------
        # Logging
        # ----------------------------------------------------

        print(
            "\nFinal selected clients:"
        )

        for proxy_cid in selected_proxy_cids:

            score_info = combined_scores.get(
                proxy_cid,
                {}
            )

            logical_cid = str(
                score_info.get(
                    "logical_cid",
                    self.proxy_to_logical_cid.get(
                        proxy_cid,
                        proxy_cid,
                    ),
                )
            )

            combined_score = (
                score_info.get(
                    "combined_score"
                )
            )

            if combined_score is None:

                print(
                    f"  Logical CID="
                    f"{logical_cid}"
                )

            else:

                print(
                    f"  Logical CID="
                    f"{logical_cid}, "
                    f"Combined="
                    f"{float(combined_score):.4f}"
                )

        return final_clients
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[
            Tuple[
                ClientProxy,
                FitRes,
            ]
        ],
        failures: List[
            Union[
                Tuple[ClientProxy, FitRes],
                BaseException,
             ]
        ],
     ):
        """
        Aggregate successful client updates using FedAvg and
        identify which selected clients actually completed
        local training successfully.
        """
    # ----------------------------------------------------
    # 1. Identify successful clients
    # ----------------------------------------------------
        self.successful_proxy_cids = [
            client.cid
            for client, _ in results
        ]
        self.successful_logical_cids = [
            self.proxy_to_logical_cid.get(
                client.cid,
                client.cid,
            )
            for client, _ in results
        ]
        print(
            f"\n"
            f"========================================\n"
            f"[ROUND {server_round}] "
            f"PARTICIPATION RESULT\n"
            f"========================================"
        )
        print(
            f"Selected clients: "
            f"{len(self.selected_proxy_cids)}"
        )
        print(
            f"Successful clients: "
            f"{len(self.successful_proxy_cids)}"
        )
        print(
            f"Failures: "
            f"{len(failures)}"
        )
        print(
            f"Successful logical CIDs: "
            f"{self.successful_logical_cids}"
        )
        # ----------------------------------------------------
        # 2. Update client system history
        # ----------------------------------------------------
        self._update_client_history_after_fit(
            server_round,
            results,
        )
    # ----------------------------------------------------
    # 3. Use the original FedAvg aggregation
    # ----------------------------------------------------
        aggregated_parameters, aggregated_metrics = (
            super().aggregate_fit(
                server_round,
                results,
                failures,
            )
        )
        # ----------------------------------------------------
        # 4. Update Reference Bank
        # ----------------------------------------------------
        self._update_reference_bank(
            server_round
        )
        return (
            aggregated_parameters,
            aggregated_metrics,
        )
