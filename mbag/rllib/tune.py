import os

import ray
from ray import tune
from ray.tune.schedulers.async_hyperband import ASHAScheduler

from sacred import Experiment
from sacred import SETTINGS as SACRED_SETTINGS

from .train import make_mbag_sacred_config

ex = Experiment("tune_mbag")
SACRED_SETTINGS.CONFIG.READ_ONLY_CONFIG = False


make_mbag_sacred_config(ex)


@ex.config
def add_tune_config(
    run,
    config,
    log_dir,
    experiment_name,
    policies_to_train,
    mask_goal,
    save_freq,
):
    train_batch_size_max = 20000
    sgd_minibatch_size_max = 4000
    num_sgd_iter_max = 10
    lr_min = 1e-5
    lr_max = 1e-1

    config.update(
        {
            "train_batch_size": tune.qrandint(1000, train_batch_size_max, 1000),
            "sgd_minibatch_size": tune.qrandint(100, sgd_minibatch_size_max, 100),
            "num_sgd_iter": tune.randint(1, num_sgd_iter_max + 1),
            "lr": tune.loguniform(lr_min, lr_max),
            "evaluation_interval": None,
        }
    )
    if run != "distillation_prediction":
        config.update(
            {
                "gamma": tune.uniform(0.9, 1),
                "kl_target": tune.loguniform(0.001, 1),
                "grad_clip": 1,
                "vf_loss_coeff": tune.loguniform(1e-5, 1e-2),
            }
        )

    if config.get("entropy_coeff_schedule") is not None:
        config["entropy_coeff_schedule"] = [
            (0, tune.loguniform(1e-4, 1)),
            (tune.loguniform(1, 1e7), 0),
        ]

    if config["env_config"].get("rewards") is not None:
        config["env_config"]["rewards"].update(
            {
                "noop": tune.uniform(-2, 0),
                "place_wrong": tune.uniform(-1, 1),
            }
        )

    for policy_id, (_, _, _, policy_config) in config["multiagent"]["policies"].items():
        if policy_id not in policies_to_train:
            continue

        policy_config["model"]["custom_model_config"].update(
            {
                "embedding_size": tune.qrandint(4, 16, 4),
                "position_embedding_size": tune.qrandint(6, 18, 6),
                "use_extra_features": False
                if mask_goal
                else tune.choice([False, True]),
                "num_conv_1_layers": tune.randint(1, 5),
                "num_layers": tune.randint(1, 5),
                "filter_size": tune.choice([3, 5]),
                "hidden_channels": tune.qrandint(8, 64, 8),
                "hidden_size": tune.qrandint(16, 64, 8),
                "num_block_id_layers": tune.randint(1, 5),
                "num_unet_layers": tune.randint(0, 6),
                "num_location_layers": tune.randint(1, 5),
                "num_heads": tune.choice([1, 2, 4]),
            }
        )

    time_attr = "time_total_s"
    if "distillation_prediction" in run:
        if config["validation"]["input"] is not None:
            tune_metric = (
                f"info/learner/{policies_to_train[0]}/"
                "learner_stats/validation/smoothed_cross_entropy"
            )
        else:
            tune_metric = f"info/learner/{policies_to_train[0]}/initial_cross_entropy"
        mode = "min"
    else:
        tune_metric = "custom_metrics/goal_similarity_mean"
        mode = "max"
    max_t = 4 * 60 * 60
    grace_period = 10 * 60
    scheduler_params = {  # noqa: F841
        "time_attr": time_attr,
        "metric": tune_metric,
        "max_t": max_t,
        "grace_period": grace_period,
    }

    time_budget_h = 96
    time_budget_s = time_budget_h * 60 * 60
    tune_log_dir = os.path.join(log_dir, experiment_name)
    num_samples = -1
    tune_params = {  # noqa: F841
        "name": "tune",
        "config": config,
        "time_budget_s": time_budget_s,
        "num_samples": num_samples,
        "local_dir": tune_log_dir,
        "mode": mode,
        "checkpoint_freq": save_freq,
    }


@ex.automain
def main(run, scheduler_params, tune_params):
    ray.init(ignore_reinit_error=True)

    scheduler = ASHAScheduler(**scheduler_params)
    analysis = tune.run(
        run,
        scheduler=scheduler,
        **tune_params,
    )
    return analysis
