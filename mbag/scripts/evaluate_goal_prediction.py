import os
import pickle
import zipfile
from logging import Logger
from typing import Dict, List, Optional, TypedDict, cast

import numpy as np
import ray
import torch
import tqdm
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_policy import TorchPolicy
from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2
from ray.rllib.utils import merge_dicts  # type: ignore
from ray.rllib.utils.typing import PolicyID
from sacred import SETTINGS, Experiment
from sacred.observers import FileStorageObserver

import mbag
from mbag.environment.types import (
    CURRENT_BLOCKS,
    CURRENT_PLAYER,
    GOAL_BLOCKS,
    LAST_INTERACTED,
    OTHER_PLAYER,
    PLAYER_LOCATIONS,
)
from mbag.evaluation.episode import MbagEpisode
from mbag.rllib.os_utils import available_cpu_count
from mbag.rllib.torch_models import MbagTorchModel
from mbag.rllib.training_utils import load_trainer

SETTINGS.CONFIG.READ_ONLY_CONFIG = False
SETTINGS.CONFIG


ex = Experiment("evaluate_goal_prediction")


@ex.config
def sacred_config():
    run = "MbagAlphaZero"  # noqa: F841
    checkpoint = ""  # noqa: F841
    policy_id = "assistant"  # noqa: F841
    config_updates = {  # noqa: F841
        "num_workers": 0,
        "evaluation_num_workers": 0,
        "num_envs_per_worker": 1,
        "num_gpus": 1 if torch.cuda.is_available() else 0,
    }
    extra_config_updates = {}  # noqa: F841

    minibatch_size = 128  # noqa: F841

    evaluate_dir = ""  # noqa: F841
    player_index = 1  # noqa: F841
    convert_to_assistant_perspective = False  # noqa: F841

    experiment_tag = "goal_predictions"  # noqa: F841
    out_dir = os.path.join(evaluate_dir, experiment_tag)  # noqa: F841
    save_blocks_and_logits = False  # noqa: F841

    observer = FileStorageObserver(out_dir)
    ex.observers.append(observer)


class GoalPredictionResult(TypedDict, total=False):
    goal_logits: Optional[np.ndarray]
    goal_blocks: Optional[np.ndarray]
    cross_entropy: float
    cross_entropy_by_last_interacted: Dict[int, float]
    cross_entropy_different: float
    cross_entropy_different_by_last_interacted: Dict[int, float]


@ex.automain
def main(  # noqa: C901
    run: str,
    checkpoint: str,
    policy_id: PolicyID,
    config_updates: dict,
    extra_config_updates: dict,
    evaluate_dir: str,
    player_index: int,
    convert_to_assistant_perspective: bool,
    minibatch_size: int,
    save_blocks_and_logits: bool,
    observer: FileStorageObserver,
    _log: Logger,
):
    ray.init(
        num_cpus=available_cpu_count(),
        ignore_reinit_error=True,
        include_dashboard=False,
    )
    mbag.logger.setLevel(_log.getEffectiveLevel())

    config_updates = merge_dicts(
        config_updates,
        extra_config_updates,
    )
    trainer = load_trainer(checkpoint, run, config_updates)
    policy = trainer.get_policy(policy_id)
    assert isinstance(policy, (TorchPolicy, TorchPolicyV2))
    model = policy.model
    assert isinstance(model, MbagTorchModel)
    model.eval()

    with zipfile.ZipFile(os.path.join(evaluate_dir, "episodes.zip")) as episodes_zip:
        with episodes_zip.open("episodes.pickle") as episodes_file:
            episodes = cast(List[MbagEpisode], pickle.load(episodes_file))

    episode_results: List[GoalPredictionResult] = []

    for episode_index, episode in enumerate(episodes):
        player_world_obs = np.stack(
            [obs[player_index][0] for obs in episode.obs_history], axis=0
        )
        player_inventory_obs = np.stack(
            [obs[player_index][1] for obs in episode.obs_history], axis=0
        )
        player_timesteps = np.array(
            [obs[player_index][2] for obs in episode.obs_history]
        )

        if convert_to_assistant_perspective:
            assert (
                len(episode.obs_history[0]) == 1
            ), "Only single player episodes supported with convert_to_assistant_perspective."

            # Convert player locations and last interacted to the assistant's perspective.
            player_locations = player_world_obs[:, PLAYER_LOCATIONS]
            player_locations[player_locations == CURRENT_PLAYER] = OTHER_PLAYER
            last_interacted = player_world_obs[:, LAST_INTERACTED]
            last_interacted[last_interacted == CURRENT_PLAYER] = OTHER_PLAYER

            # Add inventory observation for the assistant.
            player_inventory_obs = np.concatenate(
                [
                    np.zeros_like(player_inventory_obs),
                    player_inventory_obs,
                ],
                axis=1,
            )

        episode_len = len(episode.obs_history)
        episode_batch = SampleBatch(
            {
                SampleBatch.OBS: np.concatenate(
                    [
                        player_world_obs.reshape(episode_len, -1),
                        player_inventory_obs.reshape(episode_len, -1),
                        player_timesteps[:, None],
                    ],
                    axis=1,
                ),
            }
        )

        state_in = [
            torch.tensor(state_piece) for state_piece in policy.get_initial_state()
        ]
        goal_logits_batches: List[np.ndarray] = []

        for minibatch_start in tqdm.trange(
            0, episode.length, minibatch_size, desc=f"Episode {episode_index}"
        ):
            minibatch = episode_batch.slice(
                minibatch_start, minibatch_start + minibatch_size
            ).copy()
            assert len(minibatch) <= minibatch_size

            # for state_piece_index, state_piece in enumerate(state_in):
            #     minibatch[f"state_in_{state_piece_index}"] = state_piece[None]
            # minibatch[SampleBatch.SEQ_LENS] = np.array([len(minibatch)])
            policy._lazy_tensor_dict(minibatch, device=policy.devices[0])
            minibatch.set_training(False)

            with torch.no_grad():
                _, state_out = model(
                    minibatch,
                    [state_piece[None] for state_piece in state_in],
                    np.array([len(minibatch)]),
                )
            goal_logits_batch = model.goal_predictor().cpu().detach().numpy()
            goal_logits_batches.append(goal_logits_batch)

            state_in = [state_out_piece[-1] for state_out_piece in state_out]

        goal_logits = np.concatenate(goal_logits_batches, axis=0)
        world_obs = np.stack([obs[0][0] for obs in episode.obs_history], axis=0)
        world_obs = world_obs[: episode.length]

        goal_blocks = world_obs[:, GOAL_BLOCKS]
        last_interacted = world_obs[:, LAST_INTERACTED]
        start_blocks = world_obs[0, CURRENT_BLOCKS]
        different = goal_blocks != start_blocks[None]

        goal_logits_flat = (
            torch.from_numpy(goal_logits).permute(0, 2, 3, 4, 1).flatten(end_dim=-2)
        )
        goal_blocks_flat = torch.from_numpy(goal_blocks).flatten()
        last_interacted_flat = torch.from_numpy(last_interacted).flatten()
        different_flat = torch.from_numpy(different).flatten()

        goal_logprobs_flat = torch.nn.functional.log_softmax(goal_logits_flat, dim=-1)
        goal_nll = -goal_logprobs_flat[
            torch.arange(len(goal_blocks_flat)), goal_blocks_flat.long()
        ]
        cross_entropy = float(goal_nll.mean().item())
        cross_entropy_different = float(goal_nll[different_flat].mean().item())
        cross_entropy_by_last_interacted = {
            int(last_interacted_value): float(
                goal_nll[last_interacted_flat == last_interacted_value].mean().item()
            )
            for last_interacted_value in np.unique(last_interacted_flat)
        }
        cross_entropy_different_by_last_interacted = {
            int(last_interacted_value): float(
                goal_nll[
                    different_flat & (last_interacted_flat == last_interacted_value)
                ]
                .mean()
                .item()
            )
            for last_interacted_value in np.unique(last_interacted_flat)
        }

        episode_results.append(
            {
                "goal_logits": goal_logits if save_blocks_and_logits else None,
                "goal_blocks": goal_blocks if save_blocks_and_logits else None,
                "cross_entropy": cross_entropy,
                "cross_entropy_by_last_interacted": cross_entropy_by_last_interacted,
                "cross_entropy_different": cross_entropy_different,
                "cross_entropy_different_by_last_interacted": cross_entropy_different_by_last_interacted,
            }
        )

    trainer.stop()

    assert isinstance(observer.dir, str)
    with open(
        os.path.join(observer.dir, "episode_results.pickle"), "wb"
    ) as episode_results_file:
        pickle.dump(episode_results, episode_results_file)

    for episode_result in episode_results:
        del episode_result["goal_logits"]
        del episode_result["goal_blocks"]

    return {
        "episode_results": episode_results,
    }
