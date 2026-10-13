from typing import Dict, List, Tuple

from flwr.common import FitRes, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy

from flwr.server.strategy.proposed import ProposedStrategy

from experiment_logger import ExperimentLogger


class ProposedWithLogging(ProposedStrategy):
    """
    Experimental wrapper around ProposedStrategy.

    ProposedStrategy:
        Implements the client-selection algorithm.

    ExperimentLogger:
        Measures and stores experiment metrics.

    This wrapper only connects the two.
    """

    def __init__(
        self,
        num_clients: int,
        log_dir: str,
        target_accuracy: float = 0.70,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.experiment_logger = ExperimentLogger(
            num_clients=num_clients,
            log_dir=log_dir,
            strategy_name="Proposed",
            target_accuracy=target_accuracy,
        )

    def configure_fit(
        self,
        server_round,
        parameters,
        client_manager,
    ):
        # Start timing BEFORE Proposed client profiling,
        # scoring, shortlisting, and final selection.
        self.experiment_logger.start_round(
            server_round
        )

        selected = super().configure_fit(
            server_round,
            parameters,
            client_manager,
        )

        
        # Convert selected Flower/Ray proxy CIDs
        # to logical client CIDs.
        selected_logical_cids = [
            str(
                self.proxy_to_logical_cid.get(
                    proxy_cid,
                    proxy_cid,
                )
            )
            for proxy_cid in self.selected_proxy_cids
        ]

        self.experiment_logger.record_selection(
            server_round=server_round,
            selected_client_ids=selected_logical_cids,
        )

        return selected

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures,
    ) -> Tuple[Parameters, Dict[str, Scalar]]:

        aggregated_parameters, aggregated_metrics = (
            super().aggregate_fit(
                server_round,
                results,
                failures,
            )
        )

        # -----------------------------------------------------
        # Extract common fit metrics from successful clients
        # -----------------------------------------------------

        successful_logical_cids = [
            str(cid)
            for cid in self.successful_logical_cids
        ]

        train_losses = [
            float(
                fit_res.metrics.get(
                    "train_loss",
                    0.0,
                )
            )
            for _, fit_res in results
        ]
        num_examples = [
            int(fit_res.num_examples)
            for _, fit_res in results
        ]
        num_examples = [
            int(fit_res.num_examples)
            for _, fit_res in results
        ]

        fit_durations = [
            float(
                fit_res.metrics.get(
                    "fit_duration",
                    0.0,
                )
            )
            for _, fit_res in results
        ]

        samples_per_second = [
            float(
                fit_res.metrics.get(
                    "samples_per_second",
                    0.0,
                )
            )
            for _, fit_res in results
        ]

        gpu_memory_mb = [
            float(
                fit_res.metrics.get(
                    "gpu_memory_mb",
                    0.0,
                )
            )
            for _, fit_res in results
        ]

        model_bytes = 0

        if results:
            model_bytes = int(
                results[0][1].metrics.get(
                    "model_bytes",
                    0,
                )
            )

        self.experiment_logger.record_fit_results(
            server_round=server_round,
            successful_client_ids=successful_logical_cids,
            train_losses=train_losses,
            num_examples=num_examples,
            fit_durations=fit_durations,
            samples_per_second=samples_per_second,
            gpu_memory_mb=gpu_memory_mb,
            model_bytes=model_bytes,
        )

        return (
            aggregated_parameters,
            aggregated_metrics,
        )

    def aggregate_evaluate(
        self,
        server_round,
        results,
        failures,
    ):
        loss, metrics = super().aggregate_evaluate(
            server_round,
            results,
            failures,
        )

        accuracy = (
            float(metrics.get("accuracy", 0.0))
            if metrics
            else 0.0
        )

        # Flower's aggregated distributed loss is the
        # authoritative evaluation loss.
        eval_loss = (
            float(loss)
            if loss is not None
            else 0.0
        )

        self.experiment_logger.record_evaluation(
            server_round=server_round,
            accuracy=accuracy,
            eval_loss=eval_loss,
        )

        return loss, metrics