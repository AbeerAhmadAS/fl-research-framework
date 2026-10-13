from typing import Dict, List, Tuple

import flwr as fl
from flwr.common import (
    FitRes,
    GetPropertiesIns,
    Parameters,
    Scalar,
)
from flwr.server.client_proxy import ClientProxy

from experiment_logger import ExperimentLogger


class FedAvgWithLogging(fl.server.strategy.FedAvg):

    def __init__(
        self,
        num_clients: int,
        log_dir: str = "results",
        target_accuracy: float = 0.70,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.num_clients = num_clients

        self.logger = ExperimentLogger(
            num_clients=num_clients,
            log_dir=log_dir,
            strategy_name="FedAvg",
            target_accuracy=target_accuracy,
        )

        # Flower/Ray proxy CID -> stable logical client CID.
        self.proxy_to_logical_cid: Dict[str, str] = {}

        self.selected_proxy_cids: List[str] = []
        self.selected_logical_cids: List[str] = []

        # Mapping is static because logical CIDs do not change.
        self._cid_mapping_initialized = False

    # ========================================================
    # Lightweight CID mapping
    # ========================================================

    def _initialize_cid_mapping(
        self,
        client_manager,
    ) -> None:
        """
        Build Flower proxy CID -> logical CID mapping.

        This requests ONLY the client's logical CID.
        It does not perform data profiling, system profiling,
        eligibility checking, model initialization, or training.

        The mapping is created once and then reused.
        """

        if self._cid_mapping_initialized:
            return

        available_clients = client_manager.all()

        print(
            "[FedAvg] Initializing proxy-to-logical "
            "CID mapping..."
        )

        for proxy_cid, client in available_clients.items():

            try:
                request = GetPropertiesIns(
                    config={}
                )

                response = client.get_properties(
                    request,
                    timeout=None,
                    group_id=0,
                )

                logical_cid = str(
                    response.properties.get(
                        "cid",
                        proxy_cid,
                    )
                )

                self.proxy_to_logical_cid[
                    str(proxy_cid)
                ] = logical_cid

            except Exception as exc:

                raise RuntimeError(
                    "Failed to resolve logical CID for "
                    f"proxy CID {proxy_cid}: {exc}"
                ) from exc

        if len(self.proxy_to_logical_cid) != self.num_clients:
            raise RuntimeError(
                "Incomplete proxy-to-logical CID mapping: "
                f"expected {self.num_clients}, "
                f"received "
                f"{len(self.proxy_to_logical_cid)}."
            )

        self._cid_mapping_initialized = True

        print(
            "[FedAvg] CID mapping initialized for "
            f"{len(self.proxy_to_logical_cid)} clients."
        )

    # ========================================================
    # Selection
    # ========================================================

    def configure_fit(
        self,
        server_round,
        parameters,
        client_manager,
    ):

        # ----------------------------------------------------
        # Resolve stable logical IDs once.
        #
        # This is done BEFORE start_round so that this
        # bookkeeping step is not counted as FedAvg round
        # duration.
        # ----------------------------------------------------

        self._initialize_cid_mapping(
            client_manager
        )

        # ----------------------------------------------------
        # Start the common round timer.
        # ----------------------------------------------------

        self.logger.start_round(
            server_round
        )

        # ----------------------------------------------------
        # Standard Flower FedAvg random sampling.
        # ----------------------------------------------------

        selected = super().configure_fit(
            server_round,
            parameters,
            client_manager,
        )

        self.selected_proxy_cids = [
            str(client.cid)
            for client, _ in selected
        ]

        # ----------------------------------------------------
        # Convert the actually selected Flower proxy IDs
        # to stable logical client IDs BEFORE training.
        # ----------------------------------------------------

        self.selected_logical_cids = [
            self.proxy_to_logical_cid[
                proxy_cid
            ]
            for proxy_cid
            in self.selected_proxy_cids
        ]

        # ----------------------------------------------------
        # Record selection immediately.
        #
        # This remains correct even if a selected client
        # later fails during fit.
        # ----------------------------------------------------

        self.logger.record_selection(
            server_round=server_round,
            selected_client_ids=
                self.selected_logical_cids,
        )

        print(
            f"[ROUND {server_round}] "
            f"FedAvg sampled "
            f"{len(selected)} clients: "
            f"{self.selected_logical_cids}"
        )

        return selected

    # ========================================================
    # Fit aggregation + logging
    # ========================================================

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
        Parameters | None,
        Dict[str, Scalar],
    ]:

        aggregated_parameters, aggregated_metrics = (
            super().aggregate_fit(
                server_round,
                results,
                failures,
            )
        )

        successful_logical_cids: List[str] = []

        train_losses: List[float] = []
        num_examples: List[int] = []
        fit_durations: List[float] = []
        samples_per_second: List[float] = []
        gpu_memory: List[float] = []

        model_bytes = 0

        for client, fit_res in results:

            proxy_cid = str(
                client.cid
            )

            # Prefer the mapping established before training.
            # The returned metric is used as a consistency
            # check/fallback.
            logical_cid = (
                self.proxy_to_logical_cid.get(
                    proxy_cid
                )
            )

            returned_cid = str(
                fit_res.metrics.get(
                    "cid",
                    logical_cid
                    if logical_cid is not None
                    else proxy_cid,
                )
            )

            if logical_cid is None:
                logical_cid = returned_cid
                self.proxy_to_logical_cid[
                    proxy_cid
                ] = logical_cid

            elif returned_cid != logical_cid:
                raise RuntimeError(
                    "Logical CID mismatch for proxy "
                    f"{proxy_cid}: mapping={logical_cid}, "
                    f"fit_result={returned_cid}"
                )

            successful_logical_cids.append(
                logical_cid
            )

            train_losses.append(
                float(
                    fit_res.metrics.get(
                        "train_loss",
                        0.0,
                    )
                )
            )

            num_examples.append(
                int(
                    fit_res.num_examples
                )
            )

            fit_durations.append(
                float(
                    fit_res.metrics.get(
                        "fit_duration",
                        0.0,
                    )
                )
            )

            samples_per_second.append(
                float(
                    fit_res.metrics.get(
                        "samples_per_second",
                        0.0,
                    )
                )
            )

            gpu_memory.append(
                float(
                    fit_res.metrics.get(
                        "gpu_memory_mb",
                        0.0,
                    )
                )
            )

            if model_bytes == 0:
                model_bytes = int(
                    fit_res.metrics.get(
                        "model_bytes",
                        0,
                    )
                )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Do NOT overwrite selected_logical_cids here.
        #
        # selected clients were already recorded BEFORE fit.
        # successful clients are the subset that returned
        # successful FitRes objects.
        # ----------------------------------------------------

        self.logger.record_fit_results(
            server_round=server_round,
            successful_client_ids=
                successful_logical_cids,
            train_losses=train_losses,
            num_examples=num_examples,
            fit_durations=fit_durations,
            samples_per_second=
                samples_per_second,
            gpu_memory_mb=gpu_memory,
            model_bytes=model_bytes,
        )

        print(
            f"[ROUND {server_round}] "
            f"Selected logical clients: "
            f"{self.selected_logical_cids}"
        )

        print(
            f"[ROUND {server_round}] "
            f"Successful logical clients: "
            f"{successful_logical_cids}"
        )

        print(
            f"[ROUND {server_round}] "
            f"Failures: {len(failures)}"
        )

        return (
            aggregated_parameters,
            aggregated_metrics,
        )

    # ========================================================
    # Evaluation aggregation + logging
    # ========================================================

    def aggregate_evaluate(
        self,
        server_round,
        results,
        failures,
    ):

        loss, metrics = (
            super().aggregate_evaluate(
                server_round,
                results,
                failures,
            )
        )

        accuracy = 0.0

        if metrics is not None:
            accuracy = float(
                metrics.get(
                    "accuracy",
                    0.0,
                )
            )

        eval_loss = (
            float(loss)
            if loss is not None
            else 0.0
        )

        self.logger.record_evaluation(
            server_round=server_round,
            accuracy=accuracy,
            eval_loss=eval_loss,
        )

        return loss, metrics