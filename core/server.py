import timeit
import os

from logging import INFO, WARNING
from typing import Optional

import flwr as fl

from flwr.server.history import History
from flwr.common.logger import log

from . import history as hist


class SavingServer(fl.server.Server):
    def __init__(self, starting_round=0, history=None, compute_bound_only=False, dryrun=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.starting_round = starting_round
        log(INFO, "Starting round: %s", self.starting_round)
        self.history = history if history is not None else hist.CustomHistory()
        self.compute_bound_only = compute_bound_only
        self.dryrun = dryrun

    def fit(self, num_rounds: int, timeout: Optional[float]) -> History:
        """Run federated learning for a number of rounds."""
        history = self.history

        # Initialize parameters
        log(INFO, "Initializing global parameters")
        self.parameters = self._get_initial_parameters(timeout=timeout)
        if self.starting_round == 0:
            log(INFO, "Evaluating initial parameters")
            res = self.strategy.evaluate(0, parameters=self.parameters)
            if res is not None:
                loss_cen, metrics_cen = res
                log(
                    INFO,
                    "initial progress | Round number: 0 "
                    f"| Loss Centralized {loss_cen} "
                    f"| Metrics Centralized: {metrics_cen}",
                )
                history.add_loss_centralized(server_round=self.starting_round, loss=res[0])
                history.add_metrics_centralized(server_round=self.starting_round, metrics=res[1])

        log(INFO, "Federated Learning starts (start_time)")
        start_time = timeit.default_timer()

        if self.compute_bound_only:  # Does not work for personalized FL
            history = self._compute_bound_only(num_rounds=num_rounds, timeout=timeout, history=history)
        else:
            history = self._fit(num_rounds=num_rounds, timeout=timeout, history=history)

        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        log(INFO, f"Federated Learning finished in {elapsed}")
        return history

    def _compute_bound_only(self, num_rounds: int, timeout: Optional[float], history) -> History:
        log(INFO, "Compute Bound Only starts (start_time)")
        start_time = timeit.default_timer()

        # Evaluate model using strategy implementation (Server-side)
        res_cen = self.strategy.evaluate(self.starting_round + 1, parameters=self.parameters)
        if res_cen is not None:
            loss_cen, metrics_cen = res_cen
            log(
                INFO,
                f"fit progress | Round number: {self.starting_round + 1} "
                f"| Loss Centralized: {loss_cen} "
                f"| Metrics Centralized: {metrics_cen} "
                f"| Time spent: {timeit.default_timer() - start_time}",
            )
            history.add_loss_centralized(server_round=self.starting_round + 1, loss=loss_cen)
            history.add_metrics_centralized(server_round=self.starting_round + 1, metrics=metrics_cen)

        # Evaluate model on a sample of available clients (Client-side)
        # To evaluate multiple times the MC Sampling, we make a LOOP
        log(
            WARNING,
            "MC Sampling begins (it can take a while, i.e. multiple days if mc_samples higher than 100_000)",
        )
        for current_round in range(self.starting_round + 1, self.starting_round + num_rounds + 1):
            res_fed = self.evaluate_round(server_round=current_round, timeout=timeout)
            if res_fed is not None:
                loss_fed, evaluate_metrics_fed, _ = res_fed
                if loss_fed is not None:
                    history.add_loss_distributed(server_round=current_round, loss=loss_fed)
                    history.add_metrics_distributed(server_round=current_round, metrics=evaluate_metrics_fed)

            # Save Metrics
            cwd = os.getcwd()  # Path has been automatically changed by Hydra
            file_suffix = "_tmp"
            hist.plot_all_metrics_from_history(history, cwd, suffix=file_suffix)
        return history

    def _fit(self, num_rounds: int, timeout: Optional[float], history) -> History:
        log(INFO, "Federated Learning starts (start_time)")
        start_time = timeit.default_timer()

        # Run federated learning for num_rounds
        for current_round in range(self.starting_round + 1, self.starting_round + num_rounds + 1):
            # Train model and replace previous global model
            res_fit = self.fit_round(
                server_round=current_round,
                timeout=timeout,
            )
            if res_fit is not None:
                parameters_prime, fit_metrics, _ = res_fit  # fit_metrics_aggregated
                if parameters_prime:
                    self.parameters = parameters_prime
                history.add_metrics_distributed_fit(server_round=current_round, metrics=fit_metrics)

            # Evaluate model using strategy implementation
            res_cen = self.strategy.evaluate(current_round, parameters=self.parameters)
            if res_cen is not None:
                loss_cen, metrics_cen = res_cen
                log(
                    INFO,
                    f"fit progress | Round number: {current_round} "
                    f"| Loss Centralized: {loss_cen} "
                    f"| Metrics Centralized: {metrics_cen} "
                    f"| Time spent: {timeit.default_timer() - start_time}",
                )
                history.add_loss_centralized(server_round=current_round, loss=loss_cen)
                history.add_metrics_centralized(server_round=current_round, metrics=metrics_cen)

            # Evaluate model on a sample of available clients
            res_fed = self.evaluate_round(server_round=current_round, timeout=timeout)
            if res_fed is not None:
                loss_fed, evaluate_metrics_fed, _ = res_fed
                if loss_fed is not None:
                    history.add_loss_distributed(server_round=current_round, loss=loss_fed)
                    history.add_metrics_distributed(server_round=current_round, metrics=evaluate_metrics_fed)

            # Save Metrics
            cwd = os.getcwd()  # Path has been automatically changed by Hydra
            file_suffix = "_tmp"
            hist.plot_all_metrics_from_history(history, cwd, suffix=file_suffix)
            if self.dryrun:
                break
        return history
