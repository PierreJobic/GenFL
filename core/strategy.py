import os
import math
import copy
import numpy as np
import flwr as fl

from logging import INFO
from collections import OrderedDict
from typing import Dict, List, Tuple, Union
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
)
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager
from omegaconf import open_dict

import torch

from . import train, bounds

FitResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, FitRes]],
    List[Union[Tuple[ClientProxy, FitRes], BaseException]],
]

DEVICE = torch.device("cpu")


def get_universal_config_fn(
    optimizer_name: str,
    learning_rate: float,
    momentum: float,
    num_epochs: int,
    device: torch.device,
    dryrun: bool,
    toolarge: bool = False,
    is_local_pbobj_cfg: bool = False,
    pbobj_cfg: Dict[str, Union[int, float, str]] = None,
    compute_risk_certificates_every_n_rounds: int = math.inf,
    compute_client_risk_certificates_every_n_rounds: int = math.inf,
    train_size=None,
):
    def config_fn(server_round: int):
        """Return config to use for next round of training."""
        config = {
            "round": server_round,
            "learning_rate": learning_rate,
            "momentum": momentum,
            "optimizer_name": optimizer_name,
            "num_epochs": num_epochs,
            "device": device,
            "dryrun": dryrun,
            # Optional (depends on context)
            "toolarge": toolarge,
            "is_local_pbobj_cfg": is_local_pbobj_cfg,
            "pbobj_cfg": pbobj_cfg,
            "compute_risk_certificates": (server_round % compute_risk_certificates_every_n_rounds) == 0,
            "compute_client_risk_certificates": (server_round % compute_client_risk_certificates_every_n_rounds) == 0,
            # Lambda var
            "train_size": train_size,
        }
        return config

    return config_fn


class MonitoringStrategy(fl.server.strategy.FedAvg):
    def __init__(
        self,
        net,
        pbobj_cfg: Dict[str, Union[int, float, str]] = None,
        is_probabilistic_net: bool = False,
        save_every_n_rounds: int = math.inf,
        evaluate_client_every_n_rounds: int = math.inf,
        evaluate_server_every_n_rounds: int = math.inf,
        compute_risk_certificates_every_n_rounds: int = math.inf,
        compute_client_risk_certificates_every_n_rounds: int = math.inf,
        dryrun: bool = False,
        device: torch.device = DEVICE,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.net = net
        self.pbobj_cfg = pbobj_cfg  # Not used in this class
        self.is_probabilistic_net = is_probabilistic_net
        self.save_every_n_rounds = save_every_n_rounds
        self.evaluate_client_every_n_rounds = evaluate_client_every_n_rounds
        self.evaluate_server_every_n_rounds = evaluate_server_every_n_rounds
        self.compute_risk_certificates_every_n_rounds = compute_risk_certificates_every_n_rounds
        self.compute_client_risk_certificates_every_n_rounds = compute_client_risk_certificates_every_n_rounds
        self.dryrun = dryrun
        self.device = device

    def initialize_parameters(self, client_manager: ClientManager) -> Parameters | None:
        initial_parameters = super().initialize_parameters(
            client_manager
        )  # should be given by __init__ (not taken randomly from a client)
        cwd = os.getcwd()  # Path has been automatically changed by Hydra
        dict_to_save = {
            "model": self.net.state_dict(),
            "round": 0,
        }
        torch.save(dict_to_save, cwd + "/checkpoint_round_0.pth")
        self.last_round_saved = 0
        return initial_parameters

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """This function can be used to configure client-side training of model parameters."""
        client_instructions = super().configure_fit(
            server_round, parameters, client_manager
        )  # Given by on_fit_config_fn
        log(
            INFO,
            f"Configure Fit: Server round {server_round} | Client Instructions: {client_instructions[0][1].config}",
        )
        return client_instructions

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Tuple[ClientProxy, FitRes] | BaseException],
    ) -> Tuple[Parameters | None, Dict[str, Scalar]]:
        """This function can be used to aggregate model parameters and metrics."""
        # Aggregate parameters and metrics with FedAvg aggregate function
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        log(INFO, f"Aggregate Fit: Server round {server_round}: aggregated_metrics: {aggregated_metrics}")

        if self.dryrun or (aggregated_parameters is not None):
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

            if self.dryrun or (server_round % self.save_every_n_rounds == 0):
                log(INFO, f"Saving round {server_round} aggregated_ndarrays, aggregated_metrics")

                # Convert `List[np.ndarray]` to PyTorch`state_dict`
                params_dict = zip(self.net.state_dict().keys(), aggregated_ndarrays)
                state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
                self.net.load_state_dict(state_dict, strict=True)

                # Save the model
                path = os.getcwd()
                dict_to_save = {
                    "model": self.net.state_dict(),
                    "round": server_round,
                    "aggregated_metrics": aggregated_metrics,
                }
                torch.save(dict_to_save, path + f"/checkpoint_round_{server_round}.pth")
                self.last_round_saved = server_round

        return aggregated_parameters, aggregated_metrics

    def evaluate(self, server_round: int, parameters: Parameters) -> Tuple[float, Dict[str, Scalar]] | None:
        """This function can be used to perform centralized (i.e., server-side) evaluation of model parameters."""
        if self.dryrun or (server_round % self.evaluate_server_every_n_rounds == 0):
            res = super().evaluate(server_round, parameters)  # Given by on_evaluate_fn
            if res is not None:
                _, metrics = res
                log(INFO, f"Server-side Evaluate: Server round {server_round}: metrics: {metrics}")
            return res
        else:
            log(INFO, f"Server-side Evaluate: Server round {server_round}: NO EVALUATION")
            return None

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]] | None:
        """This function can be used to configure client-side evaluation of model parameters."""
        if server_round % self.evaluate_client_every_n_rounds == 0:
            client_instructions = super().configure_evaluate(
                server_round, parameters, client_manager
            )  # Given by on_evaluate_config_fn
            log(
                INFO,
                f"Configure Evaluate: Server round {server_round} | "
                f"Client Instructions: {client_instructions[0][1].config}",
            )
            return client_instructions
        else:
            log(INFO, f"Client-side Evaluate: Server round {server_round}: NO EVALUATION")
            return None  # No evaluation

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Tuple[ClientProxy, EvaluateRes] | BaseException],
    ) -> Tuple[float | None, Dict[str, Scalar]]:
        """This function can be used to aggregate client-side evaluation results."""
        aggregated_parameters, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)
        log(INFO, f"Aggregate Evaluate: Server round {server_round}: aggregated_metrics: {aggregated_metrics}")
        return aggregated_parameters, aggregated_metrics


class GenFLDziugaite(MonitoringStrategy):
    """Need:
    attr
    ----
        self.net: torch.nn.Module"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def get_evaluate_fn(net, test_loader, pbobj_cfg, samples_ensemble, loss_fn, device, dryrun):
        def evaluate_fn(server_round: int, parameters: Parameters, config: Dict[str, Union[int, float, str]]):
            # Load parameters into the model
            params_dict = zip(net.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)

            # Evaluate the model on the testset
            log(INFO, f"Server-side Evaluate: Server round {server_round}")
            with torch.no_grad():
                metrics_stch = train.testStochastic(
                    net=net,
                    test_loader=test_loader,
                    pbobj_cfg=pbobj_cfg,
                    loss_fn=loss_fn,
                    device=device,
                    dryrun=dryrun,
                )
                log(INFO, f"Server-side Evaluate: Server round {server_round} | test stochastic finished")
                metrics_mean = train.testPosteriorMean(
                    net=net,
                    test_loader=test_loader,
                    pbobj_cfg=pbobj_cfg,
                    loss_fn=loss_fn,
                    device=device,
                    dryrun=dryrun,
                )
                log(INFO, f"Server-side Evaluate: Server round {server_round} | test mean finished")
                metrics_ens = train.testEnsemble(
                    net=net,
                    test_loader=test_loader,
                    pbobj_cfg=pbobj_cfg,
                    loss_fn=loss_fn,
                    device=device,
                    samples=samples_ensemble,
                    dryrun=dryrun,
                )
                log(INFO, f"Server-side Evaluate: Server round {server_round} | test ensemble finished")
                metrics = metrics_stch | metrics_mean | metrics_ens
                metrics["KL_n"] = net.compute_kl() / pbobj_cfg.n_posterior
            log(INFO, f"Server-side Evaluate: Server round {server_round}: metrics: {metrics}")
            return metrics_mean["loss_mean"], metrics  # Emphasize on the deterministic loss

        return evaluate_fn

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Tuple[ClientProxy, EvaluateRes] | BaseException],
    ) -> Tuple[float | None, Dict[str, Scalar]]:
        # aggregated results
        aggregated_losses, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)

        if self.dryrun or (server_round % self.compute_risk_certificates_every_n_rounds == 0):
            with torch.no_grad():
                # from checkpoint
                path = os.getcwd()
                checkpoint = torch.load(path + f"/checkpoint_round_{self.last_round_saved}.pth")
                self.net.load_state_dict(checkpoint["model"])
                log_lambda = torch.log(self.net.sigma_prior).to(self.device)
                log_lambda_down, log_lambda_up = bounds.discretize_log_lambda(
                    log_lambda=log_lambda,
                    log_prior_std_base=self.pbobj_cfg.log_prior_std_base,
                    log_prior_std_precision=self.pbobj_cfg.log_prior_std_precision,
                )  # We have to discretize lambda for the Union Bound argument

                m = self.pbobj_cfg.n_bound

                # compute B
                log(
                    INFO,
                    f"Aggregate Evaluate: Server round {server_round}: computing PAC-Bayes Bounds with "
                    f"train_size m={m}",
                )
                self.net.eval()
                kl = self.net.compute_kl()
                second_cfg = copy.deepcopy(self.pbobj_cfg)
                with open_dict(second_cfg):
                    second_cfg.kl_penalty = 1.0
                B_up = bounds.bound(
                    cfg=second_cfg,
                    empirical_risk=0.0,
                    kl=kl,
                    train_size=m,
                    lamba_disc=torch.exp(2.0 * torch.tensor(log_lambda_up)),
                )
                B_down = bounds.bound(
                    cfg=second_cfg,
                    empirical_risk=0.0,
                    kl=kl,
                    train_size=m,
                    lamba_disc=torch.exp(2.0 * torch.tensor(log_lambda_down)),
                )

                # first kl inversion
                mean_train_accuracy = aggregated_metrics["error_01"]
                mean_train_error_kl_inv = bounds.inv_kl(
                    1.0 - mean_train_accuracy, np.log(2 / self.pbobj_cfg.delta_test) / self.pbobj_cfg.mc_samples
                )
                # second kl inversion : using the best B i.e. sqrt[1./2. * B_{RE}]
                if B_up < B_down:
                    pacb_bounds = bounds.approximate_BPAC_bound(1.0 - mean_train_error_kl_inv, B_up)
                else:
                    pacb_bounds = bounds.approximate_BPAC_bound(1.0 - mean_train_error_kl_inv, B_down)
                bounds_metrics = {
                    "B_up": B_up,
                    "B_down": B_down,
                    "empirical_risk_01": mean_train_error_kl_inv,
                    "risk_01": pacb_bounds,
                    "KL": kl,
                }
                aggregated_metrics["pacb_bounds"] = pacb_bounds
                aggregated_metrics["KL"] = kl
                log(INFO, f"Aggregate Evaluate: Server round {server_round}: bounds_metrics: {bounds_metrics}")
        else:
            bounds_metrics = {}
        final_metrics = aggregated_metrics | bounds_metrics
        return aggregated_losses, final_metrics


class GenFLPosterior(MonitoringStrategy):
    """Need:
    attr
    ----
        self.net: torch.nn.Module"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def get_evaluate_fn(net, test_loader, pbobj_cfg, samples_ensemble, loss_fn, device, dryrun):
        def evaluate_fn(server_round: int, parameters: Parameters, config: Dict[str, Union[int, float, str]]):
            # Load parameters into the model
            params_dict = zip(net.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)

            # Evaluate the model on the testset
            log(INFO, f"Server-side Evaluate: Server round {server_round}")
            with torch.no_grad():
                metrics_stch = train.testStochastic(
                    net=net,
                    test_loader=test_loader,
                    pbobj_cfg=pbobj_cfg,
                    loss_fn=loss_fn,
                    device=device,
                    dryrun=dryrun,
                )
                log(INFO, f"Server-side Evaluate: Server round {server_round} | test stochastic finished")
                metrics_mean = train.testPosteriorMean(
                    net=net,
                    test_loader=test_loader,
                    pbobj_cfg=pbobj_cfg,
                    loss_fn=loss_fn,
                    device=device,
                    dryrun=dryrun,
                )
                log(INFO, f"Server-side Evaluate: Server round {server_round} | test mean finished")
                metrics_ens = train.testEnsemble(
                    net=net,
                    test_loader=test_loader,
                    pbobj_cfg=pbobj_cfg,
                    loss_fn=loss_fn,
                    device=device,
                    samples=samples_ensemble,
                    dryrun=dryrun,
                )
                log(INFO, f"Server-side Evaluate: Server round {server_round} | test ensemble finished")
                metrics = metrics_stch | metrics_mean | metrics_ens
                metrics["KL_n"] = net.compute_kl() / pbobj_cfg.n_posterior
            log(INFO, f"Server-side Evaluate: Server round {server_round}: metrics: {metrics}")
            return metrics_mean["loss_mean"], metrics  # Emphasize on the deterministic loss

        return evaluate_fn

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Tuple[ClientProxy, EvaluateRes] | BaseException],
    ) -> Tuple[float | None, Dict[str, Scalar]]:
        # aggregated results
        """This function can be used to aggregate client-side evaluation results."""
        aggregated_losses, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)
        log(INFO, f"Aggregate Evaluate: Server round {server_round}: aggregated_metrics: {aggregated_metrics}")

        if self.dryrun or (server_round % self.compute_risk_certificates_every_n_rounds) == 0:
            with torch.no_grad():
                # from checkpoint
                path = os.getcwd()
                checkpoint = torch.load(path + f"/checkpoint_round_{self.last_round_saved}.pth")
                self.net.load_state_dict(checkpoint["model"])

                kl = self.net.compute_kl()
                log(
                    INFO,
                    f"Aggregate Evaluate: Server round {server_round}: computing PBB with config={self.pbobj_cfg}",
                )
                (
                    train_obj,
                    kl_n,
                    empirical_risk_ce,
                    empirical_risk_01,
                    risk_ce,
                    risk_01,
                ) = bounds.compute_final_stats_risk_server(
                    error_ce=aggregated_metrics["error_ce"],
                    error_01=aggregated_metrics["error_01"],
                    cfg=self.pbobj_cfg,
                    kl=kl,
                    # lambda_var=config["lambda_var"], TODO
                    lambda_disc=getattr(self.net, "sigma_prior", None),
                    dryrun=self.dryrun,
                )
                metrics_rc = {
                    "train_obj": train_obj,
                    "risk_ce": risk_ce,
                    "risk_01": risk_01,
                    "KL_n": kl_n,
                    "empirical_risk_ce": empirical_risk_ce,
                    "empirical_risk_01": empirical_risk_01,
                    "error_ce": aggregated_metrics["error_ce"],
                    "error_01": aggregated_metrics["error_01"],
                }

                log(INFO, f"Aggregate Evaluate: Server round {server_round}: bounds_metrics: {metrics_rc}")
        else:
            metrics_rc = {}
        final_metrics = aggregated_metrics | metrics_rc
        return aggregated_losses, final_metrics


class GenFLPrior(MonitoringStrategy):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    @staticmethod  # TODO transfer this evaluate_fn
    def get_evaluate_fn(net, test_loader, loss_fn, device, dryrun):
        def evaluate_fn(server_round: int, parameters: Parameters, config: Dict[str, Union[int, float, str]]):
            params_dict = zip(net.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)

            # Evaluate the model on the testset
            log(INFO, f"Server-side Evaluate: Server round {server_round}")
            with torch.no_grad():
                metrics = train.testNNet(
                    net,
                    test_loader,
                    loss_fn=loss_fn,
                    device=device,
                    dryrun=dryrun,
                )
            log(INFO, f"Server-side Evaluate: Server round {server_round}: metrics: {metrics}")
            return metrics["loss"], metrics

        return evaluate_fn


class GenFLPersonalized(MonitoringStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Personalized GenFL requires is_local_pbobj_cfg to be True
        assert self.on_fit_config_fn(0)["is_local_pbobj_cfg"]
        assert self.evaluate_server_every_n_rounds == math.inf  # Never evaluate on the server side (pointless !)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Tuple[ClientProxy, FitRes] | BaseException],
    ) -> Tuple[Parameters | None, Dict[str, Scalar]]:
        res = super().aggregate_fit(server_round, results, failures)
        if self.dryrun or (server_round == 1) or (server_round % self.save_every_n_rounds == 0):
            log(INFO, f"Saving round {server_round}: dict_metrics_to_save")
            dict_metrics_to_save = {f"{id}": (res.num_examples, res.metrics) for id, (_, res) in enumerate(results)}
            path = os.getcwd()
            torch.save(dict_metrics_to_save, path + f"/checkpoint_round_all_clients_metrics_{server_round}.pth")
        return res

    @staticmethod
    def get_evaluate_fn():
        def evaluate_fn(server_round: int, parameters: Parameters, config: Dict[str, Union[int, float, str]]):
            log(INFO, f"Server-side Evaluate: Server round {server_round}: NO EVALUATION")
            return None

        return evaluate_fn
