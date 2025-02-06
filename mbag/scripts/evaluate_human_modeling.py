import os
from datetime import datetime
from logging import Logger
from typing import Iterable, List, Optional, TypedDict, cast

import numpy as np
import torch
import tqdm
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.offline import JsonReader
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_policy import TorchPolicy
from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2
from ray.rllib.utils import merge_dicts  # type: ignore
from ray.rllib.utils.typing import PolicyID
from sacred import SETTINGS, Experiment
from sacred.observers import FileStorageObserver

import mbag
from mbag.rllib.alpha_zero import MbagAlphaZeroPolicy
from mbag.rllib.human_data import EPISODE_DIR, PARTICIPANT_ID
from mbag.rllib.mixture_model import MixtureModel
from mbag.rllib.torch_models import MbagTorchModel
from mbag.rllib.training_utils import load_policy

SETTINGS.CONFIG.READ_ONLY_CONFIG = False
SETTINGS.CONFIG


ex = Experiment("evaluate_human_modeling")


@ex.config
def sacred_config():
    checkpoint = ""  # noqa: F841
    policy_id = "human"  # noqa: F841
    config_updates = {  # noqa: F841
        "num_workers": 0,
        "evaluation_num_workers": 0,
        "num_envs_per_worker": 1,
        "num_gpus": 1 if torch.cuda.is_available() else 0,
    }
    extra_config_updates = {}  # noqa: F841

    human_data_dir = ""  # noqa: F841
    participant_ids = None

    # This is just used for testing to limit the amount of time spent on each episode.
    max_episode_len = None  # noqa: F841

    minibatch_size = 128  # noqa: F841

    experiment_tag = ""
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir_name = "evaluate_human_modeling_"
    if experiment_tag:
        out_dir_name += f"{experiment_tag}_"
    if participant_ids is not None:
        participant_ids_str = "_".join(map(str, participant_ids))
        out_dir_name += f"participants_{participant_ids_str}_"
    out_dir_name += time_str
    out_dir = os.path.join(checkpoint, out_dir_name)

    observer = FileStorageObserver(out_dir)
    ex.observers.append(observer)


class EpisodeHumanModelingEvaluationResults(TypedDict):
    episode_id: int
    episode_dir: str
    participant_id: int
    length: int

    accuracy: float
    cross_entropy: float

    logprobs: List[float]


class HumanModelingEvaluationResults(TypedDict):
    episode_results: List[EpisodeHumanModelingEvaluationResults]
    accuracy: float
    cross_entropy: float


@ex.automain
def main(  # noqa: C901
    checkpoint: str,
    policy_id: PolicyID,
    config_updates: dict,
    extra_config_updates: dict,
    human_data_dir: str,
    participant_ids: Optional[List[int]],
    max_episode_len: Optional[int],
    minibatch_size: int,
    observer,
    _log: Logger,
):
    mbag.logger.setLevel(_log.getEffectiveLevel())

    episode_results: List[EpisodeHumanModelingEvaluationResults] = []
    total_timesteps = 0

    config_updates = merge_dicts(
        config_updates,
        extra_config_updates,
    )
    policy = load_policy(
        checkpoint,
        policy_id,
        config_updates=config_updates,
    )
    assert isinstance(policy, (TorchPolicy, TorchPolicyV2))

    episodes: List[SampleBatch] = list(
        cast(Iterable[SampleBatch], JsonReader(human_data_dir).read_all_files())
    )
    for episode in episodes:
        del episode[SampleBatch.INFOS]  # Avoid errors when slicing the episode.
        episode[SampleBatch.AGENT_INDEX][:] = 0

        if max_episode_len is not None and len(episode) > max_episode_len:
            episode = episode.slice(0, max_episode_len)

        episode_id = int(episode[SampleBatch.EPS_ID][0])
        assert np.all(episode[SampleBatch.EPS_ID] == episode_id)
        episode_dir = str(episode[EPISODE_DIR][0])
        participant_id = int(episode[PARTICIPANT_ID][0])

        if participant_ids is not None and participant_id not in participant_ids:
            continue

        state_in = policy.get_initial_state()

        if isinstance(policy.model, MixtureModel):
            state_in[0][:] = np.log(1 / len(policy.model.components))

        correct_batches: List[np.ndarray] = []
        logprob_batches: List[np.ndarray] = []

        for minibatch_start in tqdm.trange(
            0, len(episode), minibatch_size, desc=episode_dir
        ):
            minibatch = episode.slice(
                minibatch_start, minibatch_start + minibatch_size
            ).copy()
            assert len(minibatch) <= minibatch_size
            minibatch.decompress_if_needed()

            actions = torch.tensor(minibatch[SampleBatch.ACTIONS])
            del minibatch[SampleBatch.ACTIONS]

            for state_piece_index, state_piece in enumerate(state_in):
                minibatch[f"state_in_{state_piece_index}"] = state_piece[None]
            minibatch[SampleBatch.SEQ_LENS] = np.array([len(minibatch)])
            minibatch.set_training(False)
            if not isinstance(policy, MbagAlphaZeroPolicy):
                # AlphaZero takes in numpy inputs.
                policy._lazy_tensor_dict(minibatch, device=policy.devices[0])

            compute_actions_kwargs = {}
            if isinstance(policy, MbagAlphaZeroPolicy):
                compute_actions_kwargs["compute_full_support_action_dist"] = True

            _, state_out, extra_fetches = policy.compute_actions_from_input_dict(
                minibatch, **compute_actions_kwargs
            )

            assert SampleBatch.ACTION_DIST_INPUTS in extra_fetches
            action_dist_inputs = torch.tensor(
                extra_fetches[SampleBatch.ACTION_DIST_INPUTS]
            )

            assert policy.dist_class is not None
            assert issubclass(policy.dist_class, TorchDistributionWrapper)
            assert isinstance(policy.model, TorchModelV2)
            action_dist = policy.dist_class(
                action_dist_inputs,  # type: ignore[arg-type]
                policy.model,
            )
            actions = actions.to(action_dist_inputs.device)
            correct_batches.append(
                (actions == cast(torch.Tensor, action_dist.deterministic_sample()))
                .detach()
                .cpu()
                .numpy()
            )
            logprob_batches.append(
                cast(torch.Tensor, action_dist.logp(actions)).detach().cpu().numpy()
            )

            state_in = [state_out_piece[-1] for state_out_piece in state_out]

        logprobs = np.concatenate(logprob_batches)
        # Ignore human actions which are masked due to being invalid.
        mask = logprobs > MbagTorchModel.MASK_LOGIT
        cross_entropy = -np.mean(logprobs[mask])
        accuracy = np.mean(np.concatenate(correct_batches)[mask])

        episode_results.append(
            {
                "episode_id": episode_id,
                "episode_dir": episode_dir,
                "participant_id": participant_id,
                "length": int(np.sum(mask)),
                "cross_entropy": float(cross_entropy),
                "accuracy": float(accuracy),
                "logprobs": logprobs.tolist(),
            }
        )
        total_timesteps += len(episode)

    overall_cross_entropy = 0
    overall_accuracy = 0
    for episode_result in tqdm.tqdm(episode_results):
        overall_cross_entropy += episode_result["cross_entropy"] * (
            episode_result["length"] / total_timesteps
        )
        overall_accuracy += episode_result["accuracy"] * (
            episode_result["length"] / total_timesteps
        )

    return {
        "episode_results": episode_results,
        "accuracy": overall_accuracy,
        "cross_entropy": overall_cross_entropy,
    }
