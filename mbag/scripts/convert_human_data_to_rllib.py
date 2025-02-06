import glob
import logging
import os
from typing import List, Optional

import numpy as np
from braceexpand import braceexpand
from ray.rllib.offline.json_writer import JsonWriter
from ray.rllib.policy.sample_batch import MultiAgentBatch, SampleBatch
from sacred import Experiment

from mbag.compatibility_utils import convert_old_config_to_new
from mbag.environment.config import DEFAULT_CONFIG
from mbag.environment.mbag_env import MbagConfigDict
from mbag.evaluation.episode import MbagEpisode
from mbag.rllib.human_data import (
    convert_episode_to_sample_batch,
    load_episode,
    repair_missing_player_locations,
)

ex = Experiment()


@ex.config
def sacred_config():
    data_dir = ""
    data_glob = os.path.join(data_dir, "**", "episode.{pkl,zip}")  # noqa: F841
    # Episode info file to load the MBAG config from.
    load_mbag_config_from: Optional[str] = None  # noqa: F841

    mbag_config: MbagConfigDict = {  # noqa: F841
        "world_size": (11, 10, 10),
        "abilities": {"teleportation": False, "flying": True, "inf_blocks": False},
    }
    include_noops = False  # noqa: F841
    include_noops_for_other_player_actions = True  # noqa: F841
    action_delay = DEFAULT_CONFIG["malmo"]["action_delay"]  # noqa: F841
    flat_actions = True  # noqa: F841
    flat_observations = True  # noqa: F841
    offset_rewards = False  # noqa: F841
    place_wrong_reward = 0  # noqa: F841
    player_indices = [0]  # noqa: F841
    inventory_player_indices = player_indices  # noqa: F841
    policy_id = None  # noqa: F841
    max_seq_len = None  # noqa: F841
    remove_malmo_observations = True  # noqa: F841

    # sequence_overlap > 1 outputs multiple copies of each episode split into sequences
    # that are overlapping.
    sequence_overlap = 1  # noqa: F841

    experiment_tag = None
    experiment_name = "rllib"
    if experiment_tag is not None:
        experiment_name += f"_{experiment_tag}"
    if not include_noops:
        experiment_name += "_no_noops"
    elif include_noops_for_other_player_actions:
        experiment_name += "_with_noops"
    else:
        experiment_name += "_with_own_noops"
    experiment_name += "_flat_actions" if flat_actions else "_tuple_actions"
    experiment_name += "_flat_observations" if flat_observations else ""
    experiment_name += (
        f"_place_wrong_reward_{place_wrong_reward}" if place_wrong_reward != 0 else ""
    )
    experiment_name += f"_repaired_player_{'_'.join(map(str, player_indices))}"
    experiment_name += f"_inventory_{'_'.join(map(str, inventory_player_indices))}"
    experiment_name += f"_seq_{max_seq_len}" if max_seq_len is not None else ""
    experiment_name += f"_overlap_{sequence_overlap}" if sequence_overlap > 1 else ""
    out_root = data_dir
    out_dir = os.path.join(out_root, experiment_name)  # noqa: F841


@ex.automain
def main(  # noqa: C901
    data_dir: str,
    data_glob: str,
    load_mbag_config_from: Optional[str],
    out_dir: str,
    mbag_config: MbagConfigDict,
    include_noops: bool,
    include_noops_for_other_player_actions: bool,
    action_delay: float,
    flat_actions: bool,
    flat_observations: bool,
    offset_rewards: bool,
    place_wrong_reward: float,
    player_indices: List[int],
    inventory_player_indices: List[int],
    policy_id: Optional[str],
    max_seq_len: Optional[int],
    remove_malmo_observations: bool,
    sequence_overlap: int,
    _log: logging.Logger,
):
    episode: MbagEpisode

    if load_mbag_config_from is not None:
        _log.info(f"loading environment config from {load_mbag_config_from}...")
        episode = load_episode(load_mbag_config_from)
        mbag_config = episode.env_config

    episode_fnames: List[str] = []
    for expanded_glob in braceexpand(data_glob):
        episode_fnames.extend(glob.glob(expanded_glob, recursive=True))
    if not episode_fnames:
        raise FileNotFoundError(f"No episode files found matching {data_glob}.")

    if os.path.exists(out_dir):
        raise FileExistsError(f"Output directory {out_dir} already exists.")

    json_writer = JsonWriter(out_dir)

    for episode_fname in sorted(episode_fnames):
        try:
            _log.info(f"reading {episode_fname}...")
            episode = load_episode(episode_fname)
        except Exception:
            _log.exception(f"failed to read {episode_fname}")
            continue

        _log.info("repairing missing player locations if necessary...")
        episode = repair_missing_player_locations(episode, mbag_config=mbag_config)

        assert episode_fname[: len(data_dir)] == data_dir
        episode_dir = os.path.dirname(episode_fname)[len(data_dir) :].lstrip(
            os.path.sep
        )
        participant_id = -1
        for path_part in episode_dir.split(os.path.sep):
            if path_part.startswith("participant_"):
                participant_id = int(path_part[len("participant_") :])
                break

        if hasattr(episode, "env_config"):
            _log.info("using env config from EpisodeInfo")
            mbag_config = episode.env_config

        mbag_config = convert_old_config_to_new(mbag_config)

        if place_wrong_reward != 0:
            mbag_config["rewards"]["place_wrong"] = place_wrong_reward
            for player_config in mbag_config["players"]:
                if "place_wrong" in player_config["rewards"]:
                    player_config["rewards"]["place_wrong"] = place_wrong_reward

        for player_index in player_indices:
            _log.info(f"converting to RLlib format for player {player_index}...")
            sample_batch = convert_episode_to_sample_batch(
                episode,
                player_index=player_index,
                inventory_player_indices=inventory_player_indices,
                participant_id=participant_id,
                episode_dir=episode_dir,
                mbag_config=mbag_config,
                offset_rewards=offset_rewards,
                place_wrong_reward=place_wrong_reward,
                include_noops=include_noops,
                include_noops_for_other_player_actions=include_noops_for_other_player_actions,
                flat_actions=flat_actions,
                flat_observations=flat_observations,
                action_delay=action_delay,
                remove_malmo_observations=remove_malmo_observations,
            )

            if len(sample_batch) == 0:
                _log.info("skipping empty trajectory")
                continue
            total_reward = sample_batch[SampleBatch.REWARDS].sum()
            _log.info(
                "episode info: participant ID=%d length=%d total reward=%.1f",
                participant_id,
                len(sample_batch),
                total_reward,
            )
            if place_wrong_reward == 0:
                if total_reward != episode.cumulative_reward:
                    _log.error(
                        "total reward mismatch: %.1f != %.1f",
                        total_reward,
                        episode.cumulative_reward,
                    )
                assert total_reward == episode.cumulative_reward

            _log.info("saving trajectory...")
            for overlap_index in range(sequence_overlap):
                if max_seq_len is not None:
                    start_index = overlap_index * max_seq_len // sequence_overlap
                    remaining_length = len(sample_batch) - start_index
                    seq_lens: List[int] = [start_index] if start_index > 0 else []
                    while remaining_length > 0:
                        seq_lens.append(min(remaining_length, max_seq_len))
                        remaining_length -= max_seq_len
                    sample_batch[SampleBatch.SEQ_LENS] = np.array(seq_lens)
                    _log.debug(
                        f"saving with seq_lens={sample_batch[SampleBatch.SEQ_LENS]}"
                    )

                if policy_id is not None:
                    multi_agent_batch = MultiAgentBatch(
                        {
                            policy_id: sample_batch,
                        },
                        env_steps=len(sample_batch),
                    )
                    json_writer.write(multi_agent_batch)
                else:
                    json_writer.write(sample_batch)

    return {"mbag_config": mbag_config, "out_dir": out_dir}
