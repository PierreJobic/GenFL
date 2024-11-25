import hydra

from omegaconf import DictConfig, OmegaConf

from core import utils


@hydra.main(config_path="conf", config_name="genfl_posterior")
def runexp(cfg: DictConfig):
    import os
    import math

    from pathlib import Path
    from logging import INFO
    from hydra.core.hydra_config import HydraConfig
    from hydra.utils import get_original_cwd
    from omegaconf import open_dict

    import torch
    import numpy as np
    import flwr as fl

    from flwr.server.client_manager import SimpleClientManager
    from flwr.common.logger import log

    from core import utils, model, client, loss, strategy as strat, server as svr, metrics

    DEVICE = torch.device("cpu")
    OmegaConf.resolve(cfg)

    A = (cfg.prior_type == "learnt" and cfg.perc_val != 0.0) or "Dziugaite" in cfg.model_name
    B = cfg.prior_type == "rand" and cfg.perc_val == 0.0
    assert A or B

    log(INFO, "\n" + OmegaConf.to_yaml(cfg))
    log(INFO, HydraConfig.get().job.override_dirname)

    seed = cfg.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    eval('setattr(torch.backends.cudnn, "deterministic", True)')
    eval('setattr(torch.backends.cudnn, "benchmark", False)')

    if cfg.compute_bound_only:
        log(INFO, f"Resuming from checkpoint (bound only): {cfg.resume_checkpoint}")
        round = cfg.resume_checkpoint_round
        log(INFO, f"Resuming from checkpoint round (bound only): {round}")
        dict_to_load = torch.load(cfg.resume_checkpoint + f"/checkpoint_round_{round}.pth")

    if cfg.prior_type == "rand":
        with open_dict(cfg):
            cfg.pbobj.dropout_prob = 0.0  # TODO: remove this or make an assert

    if cfg.model_name == "ProbNNet4lDziugaite":
        net0_cls = model.NNet4lDziugaite
        net_cls = model.ProbNNet4lDziugaite
        loss_fn = loss.loss_logistic()
        rho_prior = cfg.log_lambda_prior  # -3
    elif cfg.model_name == "ProbNNet4lPerez":
        net0_cls = model.NNet4lPerez
        net_cls = model.ProbNNet4lPerez
        loss_fn = loss.loss_cross_entropy_bounded()
        rho_prior = math.log(math.exp(cfg.sigma_prior) - 1.0)
    net0 = net0_cls().to(DEVICE)
    if cfg.prior_type == "learnt" and not cfg.get("compute_bound_only", False):
        log(INFO, f"Resuming from checkpoint prior: {cfg.resume_checkpoint_prior}")
        old_cfg = OmegaConf.load(cfg.resume_checkpoint_prior + "/.hydra/config.yaml")
        model_load_path = utils.get_last_from_pattern(cfg.resume_checkpoint_prior + "/checkpoint_round_*.pth")
        round = int(model_load_path.split("_")[-1].split(".")[0])
        log(INFO, f"Resuming from checkpoint prior: round {round}")
        dict_to_load = torch.load(cfg.resume_checkpoint_prior + f"/checkpoint_round_{round}.pth")
        net0.load_state_dict(dict_to_load["model"])

        # Datasets must be the same, so that trainsets and prior_sets are the same
        assert cfg.data_name == old_cfg.data_name
        assert cfg.partition_type == old_cfg.partition_type
        assert cfg.perc_data == old_cfg.perc_data
        if not ("Dziugaite" in cfg.model_name):
            assert cfg.perc_train == old_cfg.perc_train
            assert cfg.perc_val == old_cfg.perc_val
        assert cfg.num_clients == old_cfg.num_clients
        assert cfg.seed == old_cfg.seed

    net = net_cls(rho_prior=rho_prior, init_net=net0).to(DEVICE)
    checkpoint_parameters = fl.common.ndarrays_to_parameters([val.cpu().numpy() for _, val in net.state_dict().items()])

    client_fn, testloader, client_global_sizes = client.gen_client_genfl_posterior_fn(
        # Data parameters
        data_path=cfg.data_path,
        partition_type=cfg.partition_type,
        perc_data=cfg.perc_data,
        perc_val=cfg.perc_val,
        perc_test=cfg.perc_test,
        # Module parameters
        net_cls=net_cls,
        rho_prior=rho_prior,
        init_net=net0,
        loss_fn=loss_fn,
        # Optim parameters
        batch_size_client=cfg.batch_size_client,
        batch_size_server=cfg.batch_size_server,
        # Federated Learning parameters
        num_clients=cfg.num_clients,
        # Misc
        device=DEVICE,
        seed=seed,
        logistic=cfg.logistic,
    )

    train_size = client_global_sizes["train"]
    with open_dict(cfg):
        cfg.pbobj.n_posterior = train_size
        cfg.pbobj.n_bound = train_size
        log(
            INFO,
            f"Computed values: n_posterior: {cfg.pbobj.n_posterior} | "
            f"n_bound: {cfg.pbobj.n_bound} | nb classes: {cfg.pbobj.classes}",
        )

    evaluate_fn = strat.GenFLPosterior.get_evaluate_fn(
        net=net,
        test_loader=testloader,
        pbobj_cfg=cfg.pbobj,
        samples_ensemble=cfg.samples_ensemble,
        loss_fn=loss_fn,
        device=DEVICE,
        dryrun=cfg.dryrun,
    )

    config_fn = strat.get_universal_config_fn(
        optimizer_name=cfg.optimizer_name,
        learning_rate=cfg.learning_rate,
        momentum=cfg.momentum,
        num_epochs=cfg.num_epochs,
        device=DEVICE,
        dryrun=cfg.dryrun,
        toolarge=cfg.toolarge,
        is_local_pbobj_cfg=cfg.is_local_pbobj_cfg,
        pbobj_cfg=cfg.pbobj,
        compute_risk_certificates_every_n_rounds=cfg.compute_risk_certificates_every_n_rounds,
        compute_client_risk_certificates_every_n_rounds=cfg.compute_client_risk_certificates_every_n_rounds,
        train_size=cfg.pbobj.n_posterior,
    )

    if cfg.model_name == "ProbNNet4lDziugaite":
        strategy_cls = strat.GenFLDziugaite
        # the next line works aswell, however bound computation is different
        # strategy_cls = strat.GenFLPosterior
    elif cfg.model_name == "ProbNNet4lPerez":
        strategy_cls = strat.GenFLPosterior
    strategy = strategy_cls(
        # FedAvg Parameters
        initial_parameters=checkpoint_parameters,
        min_available_clients=cfg.num_clients,
        accept_failures=cfg.get("accept_failures", False),
        # # Train
        min_fit_clients=max(1, int(cfg.num_clients * cfg.client_fraction)),
        fraction_fit=cfg.client_fraction,
        on_fit_config_fn=config_fn,
        fit_metrics_aggregation_fn=metrics.metrics_aggregation_universal_fn,
        # # Evaluate
        min_evaluate_clients=1,
        fraction_evaluate=cfg.fraction_evaluate,
        on_evaluate_config_fn=config_fn,
        evaluate_metrics_aggregation_fn=metrics.metrics_aggregation_universal_fn,
        evaluate_fn=evaluate_fn,
        # GenFL Parameters
        net=net,
        pbobj_cfg=cfg.pbobj,
        # Monitoring Parameters
        save_every_n_rounds=cfg.save_every_n_rounds,
        evaluate_client_every_n_rounds=cfg.evaluate_client_every_n_rounds,
        evaluate_server_every_n_rounds=cfg.evaluate_server_every_n_rounds,
        compute_risk_certificates_every_n_rounds=cfg.compute_risk_certificates_every_n_rounds,
        compute_client_risk_certificates_every_n_rounds=cfg.compute_client_risk_certificates_every_n_rounds,
        dryrun=cfg.dryrun,
    )

    # Specify Ray config
    if "cpus_per_task" in HydraConfig.get().launcher:
        num_cpus = HydraConfig.get().launcher.cpus_per_task
        log(INFO, f"Detecting num_cpus: {num_cpus}")
        ray_init_args = {
            "include_dashboard": False,
            "runtime_env": {"working_dir": get_original_cwd()},
            "num_cpus": num_cpus,
        }
    else:
        ray_init_args = {
            "include_dashboard": False,
            "runtime_env": {"working_dir": get_original_cwd()},
        }

    # Client resources
    client_resources = {}

    # Start simulation
    client_manager = SimpleClientManager()

    server_init = svr.SavingServer(
        client_manager=client_manager,
        strategy=strategy,
        compute_bound_only=cfg.compute_bound_only,
    )

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        client_resources=client_resources,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
        server=server_init,
        ray_init_args=ray_init_args,
    )

    file_suffix: str = (
        f"{cfg.partition_type}"
        f"_C={cfg.num_clients}"
        f"_B={cfg.batch_size_client}"
        f"_E={cfg.num_epochs}"
        f"_R={cfg.num_rounds}"
        f"_O={cfg.pbobj.objective}"
        f"_P={cfg.prior_type}"
    )

    cwd = os.getcwd()

    np.save(
        Path(cwd) / Path(f"hist{file_suffix}"),
        history,  # type: ignore
    )


if __name__ == "__main__":
    OmegaConf.register_new_resolver("my_subdir_suffix", utils.my_subdir_suffix_impl)
    runexp()
