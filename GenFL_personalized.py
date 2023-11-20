import hydra

from omegaconf import DictConfig, OmegaConf, open_dict

from core import utils


@hydra.main(config_path="conf", config_name="genfl_personalized")
def runexp(cfg: DictConfig):
    import os
    import math

    from pathlib import Path
    from logging import INFO
    from hydra.core.hydra_config import HydraConfig
    from hydra.utils import get_original_cwd

    import torch
    import numpy as np
    import flwr as fl

    from flwr.server.client_manager import SimpleClientManager
    from flwr.common.logger import log

    from core import model, client, loss, strategy as strat, server as svr, metrics

    DEVICE = torch.device("cpu")

    assert cfg.partition_type != "exact_iid"
    assert cfg.perc_val > 0.0 or cfg.prior_type == "rand"
    assert cfg.perc_test > 0.0

    if cfg.objective == "prior":
        log(INFO, "#" * 10 + " PRIOR TRAINING " + "#" * 10)
        import GenFL_prior

        GenFL_prior.runexp(cfg)  # It is a prior learnt with special config

    elif cfg.objective == "personalized":
        log(INFO, "#" * 10 + " PERSONALIZED TRAINING " + "#" * 10)
        assert cfg.get("resume_checkpoint_prior", False)
        OmegaConf.resolve(cfg)
        log(INFO, "\n" + OmegaConf.to_yaml(cfg))
        log(INFO, HydraConfig.get().job.override_dirname)
        seed = cfg.seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        eval('setattr(torch.backends.cudnn, "deterministic", True)')
        eval('setattr(torch.backends.cudnn, "benchmark", False)')

        # Resuming from checkpoint
        log(INFO, f"Resuming from checkpoint prior: {cfg.resume_checkpoint_prior}")
        old_cfg = OmegaConf.load(cfg.resume_checkpoint_prior + "/.hydra/config.yaml")
        model_load_path = utils.get_last_from_pattern(cfg.resume_checkpoint_prior + "/checkpoint_round_*.pth")
        round = int(model_load_path.split("_")[-1].split(".")[0])
        log(INFO, f"Resuming from checkpoint posterior: round {round}")
        dict_to_load = torch.load(cfg.resume_checkpoint_prior + f"/checkpoint_round_{round}.pth")

        net0_cls = model.NNet4lPerez
        net0 = net0_cls().to(DEVICE)
        rho_prior = math.log(math.exp(cfg.sigma_prior) - 1.0)
        net_cls = model.ProbNNet4lPerez
        net0.load_state_dict(dict_to_load["model"])
        net = net_cls(rho_prior=rho_prior, init_net=net0).to(DEVICE)
        checkpoint_parameters = fl.common.ndarrays_to_parameters(
            [val.cpu().numpy() for _, val in net.state_dict().items()]
        )
        # Datasets must be the same, so that trainsets, prior_sets, testsets are the same
        assert cfg.perc_data == old_cfg.perc_data
        assert cfg.perc_train == old_cfg.perc_train
        assert cfg.perc_val == old_cfg.perc_val
        assert cfg.perc_test == old_cfg.perc_test
        assert cfg.partition_type == old_cfg.partition_type
        assert cfg.num_clients == old_cfg.num_clients

        loss_fn = loss.loss_cross_entropy_bounded()

        client_fn, testloader, client_global_sizes = client.gen_client_genfl_personalized_fn(
            # Data parameters
            data_path=cfg.data_path,
            partition_type=cfg.partition_type,
            perc_val=cfg.perc_val,
            perc_test=cfg.perc_test,
            perc_data=cfg.perc_data,
            # Module parameters
            net_cls=net_cls,
            rho_prior=rho_prior,
            loss_fn=loss_fn,
            # Optim parameters
            batch_size_client=cfg.batch_size_client,
            batch_size_server=cfg.batch_size_server,
            # Federated Learning parameters
            num_clients=cfg.num_clients,
            # Misc
            seed=seed,
        )
        with open_dict(cfg):
            cfg.pbobj.n_posterior = client_global_sizes["train"]
            cfg.pbobj.n_bound = client_global_sizes["train"]
            log(
                INFO,
                f"Computed values: n_posterior: {cfg.pbobj.n_posterior} | "
                f"n_bound: {cfg.pbobj.n_bound} | nb classes: {cfg.pbobj.classes}",
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

        strategy = strat.GenFLPersonalized(
            # FedAvg Parameters
            initial_parameters=checkpoint_parameters,
            min_available_clients=cfg.num_clients,
            accept_failures=cfg.get("accept_failures", False),
            # # Train
            min_fit_clients=max(1, int(cfg.num_clients * cfg.client_fraction)),
            fraction_fit=cfg.client_fraction,
            on_fit_config_fn=config_fn,
            fit_metrics_aggregation_fn=metrics.metrics_aggregation_universal_personalized_fn,
            # # Evaluate
            min_evaluate_clients=1,
            fraction_evaluate=cfg.fraction_evaluate,
            on_evaluate_config_fn=config_fn,
            evaluate_metrics_aggregation_fn=metrics.metrics_aggregation_universal_personalized_fn,
            evaluate_fn=strat.GenFLPersonalized.get_evaluate_fn(),
            # GenFL Parameters
            net=net,
            pbobj_cfg=cfg.pbobj,
            # Monitoring Parameters
            save_every_n_rounds=cfg.save_every_n_rounds,
            evaluate_client_every_n_rounds=math.inf,
            evaluate_server_every_n_rounds=math.inf,
            compute_risk_certificates_every_n_rounds=cfg.compute_risk_certificates_every_n_rounds,
            compute_client_risk_certificates_every_n_rounds=cfg.compute_client_risk_certificates_every_n_rounds,
            dryrun=cfg.dryrun,
            device=DEVICE,
        )

        # (optional) specify Ray config
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
        starting_round = 0
        history = None
        server_init = svr.SavingServer(
            starting_round=starting_round,
            history=history,
            client_manager=client_manager,
            strategy=strategy,
        )
        total_rounds = cfg.num_rounds
        history = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=cfg.num_clients,
            client_resources=client_resources,
            config=fl.server.ServerConfig(num_rounds=total_rounds),
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
            f"_T={cfg.objective}"
        )

        cwd = os.getcwd()

        np.save(
            Path(cwd) / Path(f"hist{file_suffix}"),
            history,  # type: ignore
        )
    else:
        raise ValueError(f"Unknown objective: {cfg.objective}")


if __name__ == "__main__":
    OmegaConf.register_new_resolver("my_subdir_suffix", utils.my_subdir_suffix_impl)
    runexp()
