import copy
import random
import zipfile
from datetime import datetime, timedelta
from typing import Optional, Sequence, Union

import numpy as np
from ray.rllib.evaluation import SampleBatchBuilder
from ray.rllib.policy.sample_batch import SampleBatch

from mbag.agents.action_distributions import MbagActionDistribution
from mbag.compatibility_utils import OldHumanDataUnpickler
from mbag.environment.actions import MbagAction, MbagActionTuple
from mbag.environment.blocks import MinecraftBlocks
from mbag.environment.config import DEFAULT_CONFIG, MbagConfigDict
from mbag.environment.types import CURRENT_PLAYER, GOAL_BLOCKS, PLAYER_LOCATIONS
from mbag.evaluation.episode import MbagEpisode

PARTICIPANT_ID = "participant_id"
EPISODE_DIR = "episode_dir"


def load_episode(episode_fname: str) -> MbagEpisode:
    if episode_fname.endswith(".zip"):
        with zipfile.ZipFile(episode_fname, "r") as zip_file:
            try:
                with zip_file.open("episode.pkl", "r") as episode_file:
                    episode = OldHumanDataUnpickler(episode_file).load()
            except KeyError:
                with zip_file.open("episodes.pickle", "r") as episode_file:
                    (episode,) = OldHumanDataUnpickler(episode_file).load()
    else:
        with open(episode_fname, "rb") as episode_file:
            episode = OldHumanDataUnpickler(episode_file).load()
    if not isinstance(episode, MbagEpisode):
        raise ValueError(f"Invalid episode info in {episode_fname} ({type(episode)})")
    return episode


def convert_episode_to_sample_batch(  # noqa: C901
    episode: MbagEpisode,
    *,
    player_index: int,
    # Players to include in the inventory observations.
    inventory_player_indices: Optional[Sequence[int]] = None,
    participant_id: int = -1,
    episode_dir: str = "",
    mbag_config: Optional[MbagConfigDict] = None,
    offset_rewards=False,
    place_wrong_reward: float = 0,
    include_noops=True,
    include_noops_for_other_player_actions=True,
    flat_actions=False,
    flat_observations=False,
    remove_malmo_observations=False,
    action_delay=DEFAULT_CONFIG["malmo"]["action_delay"],
) -> SampleBatch:
    if mbag_config is None:
        mbag_config = episode.env_config

    if inventory_player_indices is None:
        inventory_player_indices = range(mbag_config["num_players"])

    if mbag_config["rewards"]["own_reward_prop"] != 0:
        raise ValueError("This function only supports own_reward_prop=0.")

    sample_batch_builder = SampleBatchBuilder()
    episode_id = random.randrange(int(1e18))
    t = 0
    # Keep track of any rewards received during intermediate NOOPs.
    intermediate_rewards: float = 0
    prev_action_time: Optional[datetime] = None
    prev_action: Union[int, MbagActionTuple] = 0 if flat_actions else (0, 0, 0)
    current_time: Optional[datetime] = None
    for i in range(episode.length):
        obs = episode.obs_history[i][player_index]
        info = episode.info_history[i][player_index]
        reward = episode.reward_history[i]
        if offset_rewards:
            reward = episode.reward_history[i + 1]
        assert reward == sum(
            info["goal_dependent_reward"] + info["goal_independent_reward"]
            for info in episode.info_history[i]
        )
        action = info["action"]
        actions = [info["action"] for info in episode.info_history[i]]
        if include_noops_for_other_player_actions:
            not_noop = any(action.action_type != MbagAction.NOOP for action in actions)
        else:
            not_noop = action.action_type != MbagAction.NOOP
        world_obs = obs[0]

        for other_player_index in range(mbag_config["num_players"]):
            other_info = episode.info_history[i][other_player_index]
            other_action = other_info["action"]
            if (
                other_action.action_type == MbagAction.PLACE_BLOCK
                and not other_info["action_correct"]
                and world_obs[(GOAL_BLOCKS,) + other_action.block_location]
                != MinecraftBlocks.AIR
            ):
                reward += place_wrong_reward
            if (
                other_action.action_type == MbagAction.BREAK_BLOCK
                and other_info["action_correct"]
                and world_obs[(GOAL_BLOCKS,) + other_action.block_location]
                != MinecraftBlocks.AIR
            ):
                reward -= place_wrong_reward

        intermediate_rewards += reward

        if info["malmo_observations"]:
            if prev_action_time is None:
                prev_action_time = info["malmo_observations"][0][0]
            current_time = info["malmo_observations"][-1][0]
        if remove_malmo_observations:
            info = {
                **info,
                "malmo_observations": [],
            }

        if (include_noops and action_delay == 0) or not_noop:
            action_id: Union[int, MbagActionTuple]
            if flat_actions:
                action_id = MbagActionDistribution.get_flat_action(
                    mbag_config, action.to_tuple()
                )
            else:
                action_id = action.to_tuple()
            world_obs = obs[0]
            inventory_obs = obs[1]
            if inventory_obs.ndim == 1:
                # Old observations, which may be present in old human data,
                # only had the block counts for the given player, not for
                # all players. We need to add the block counts for all players
                # in this case.
                inventory_obs_pieces = [inventory_obs]
                for other_player_index in inventory_player_indices:
                    if other_player_index != player_index:
                        if other_player_index >= len(episode.obs_history[i]):
                            # The other player's observation is missing.
                            other_inventory = np.zeros_like(inventory_obs)
                        else:
                            other_inventory = episode.obs_history[i][
                                other_player_index
                            ][1]
                        inventory_obs_pieces.append(other_inventory)
                inventory_obs = np.stack(inventory_obs_pieces, axis=0)
            else:
                # Modify inventory_obs to only include the specified players.
                assert player_index in inventory_player_indices
                inventory_obs_pieces = [inventory_obs[0]]

                for inventory_player_index in inventory_player_indices:
                    if inventory_player_index == player_index:
                        continue
                    inventory_index = inventory_player_index
                    if inventory_index < player_index:
                        inventory_index += 1

                    if inventory_index < inventory_obs.shape[0]:
                        inventory_obs_pieces.append(inventory_obs[inventory_index])
                    else:
                        # The inventory observation is missing for this player.
                        inventory_obs_pieces.append(np.zeros_like(inventory_obs[0]))
                inventory_obs = np.stack(inventory_obs_pieces, axis=0)
            # The inventory obs should be zeros if the player has infinite blocks.
            if mbag_config["abilities"]["inf_blocks"]:
                inventory_obs = np.zeros_like(inventory_obs)
            obs = world_obs, inventory_obs, np.array(t)
            if flat_observations:
                obs = np.concatenate([obs_piece.flat for obs_piece in obs])

            # Add NOOPs based on action delay if necessary.
            if include_noops and action_delay > 0:
                assert prev_action_time is not None and current_time is not None
                while current_time > prev_action_time + timedelta(seconds=action_delay):
                    prev_action_time += timedelta(seconds=action_delay)
                    noop_action: Union[int, MbagActionTuple] = (
                        0 if flat_actions else (0, 0, 0)
                    )
                    sample_batch_builder.add_values(
                        **{
                            SampleBatch.T: t,
                            SampleBatch.EPS_ID: episode_id,
                            SampleBatch.AGENT_INDEX: player_index,
                            SampleBatch.OBS: obs,
                            SampleBatch.ACTIONS: noop_action,
                            SampleBatch.PREV_ACTIONS: prev_action,
                            SampleBatch.ACTION_PROB: 1.0,
                            SampleBatch.ACTION_LOGP: 0.0,
                            SampleBatch.REWARDS: intermediate_rewards,
                            SampleBatch.DONES: False,
                            SampleBatch.INFOS: {
                                "timestamp": current_time,
                            },
                            PARTICIPANT_ID: participant_id,
                            EPISODE_DIR: episode_dir,
                        }
                    )
                    intermediate_rewards = 0
                    prev_action = noop_action
                    t += 1
                prev_action_time = current_time

            if info.get("timestamp") is None and current_time is not None:
                info = {
                    **info,
                    "timestamp": current_time,
                }

            sample_batch_builder.add_values(
                **{
                    SampleBatch.T: t,
                    SampleBatch.EPS_ID: episode_id,
                    SampleBatch.AGENT_INDEX: player_index,
                    SampleBatch.OBS: obs,
                    SampleBatch.ACTIONS: action_id,
                    SampleBatch.PREV_ACTIONS: prev_action,
                    SampleBatch.ACTION_PROB: 1.0,
                    SampleBatch.ACTION_LOGP: 0.0,
                    SampleBatch.REWARDS: intermediate_rewards,
                    SampleBatch.DONES: False,
                    SampleBatch.INFOS: info,
                    PARTICIPANT_ID: participant_id,
                    EPISODE_DIR: episode_dir,
                }
            )
            intermediate_rewards = 0
            prev_action = action_id
            t += 1
    return sample_batch_builder.build_and_reset()


def repair_missing_player_locations(
    episode: MbagEpisode,
    *,
    mbag_config: Optional[MbagConfigDict] = None,
) -> MbagEpisode:
    """
    In some of the human data, the observations seem to be missing the current player's
    location. This function tries to repair the missing locations by using the
    previous locations combined with the actions taken by the player.
    """

    repaired_episode = copy.deepcopy(episode)

    if mbag_config is None:
        mbag_config = episode.env_config
    width, height, depth = mbag_config["world_size"]
    num_players = mbag_config.get("num_players", len(episode.obs_history[0]))

    for player_index in range(num_players):
        prev_world_obs, _, _ = repaired_episode.obs_history[0][player_index]
        if not np.any(prev_world_obs[PLAYER_LOCATIONS] == CURRENT_PLAYER):
            raise ValueError(
                f"Player {player_index} location is missing in the first observation"
            )

        for t in range(1, episode.length):
            world_obs, _, _ = repaired_episode.obs_history[t][player_index]
            prev_action = repaired_episode.info_history[t - 1][player_index]["action"]

            if np.any(world_obs[PLAYER_LOCATIONS] == CURRENT_PLAYER):
                pass  # The player location is already present in the observation.
            else:
                prev_xs, prev_ys, prev_zs = np.where(
                    prev_world_obs[PLAYER_LOCATIONS] == CURRENT_PLAYER
                )
                (prev_x,) = set(prev_xs)
                (prev_z,) = set(prev_zs)
                prev_y = np.min(prev_ys)

                x, y, z = prev_x, prev_y, prev_z
                if prev_action.action_type == MbagAction.MOVE_POS_X:
                    x += 1
                elif prev_action.action_type == MbagAction.MOVE_NEG_X:
                    x -= 1
                elif prev_action.action_type == MbagAction.MOVE_POS_Y:
                    y += 1
                elif prev_action.action_type == MbagAction.MOVE_NEG_Y:
                    y -= 1
                elif prev_action.action_type == MbagAction.MOVE_POS_Z:
                    z += 1
                elif prev_action.action_type == MbagAction.MOVE_NEG_Z:
                    z -= 1

                assert 0 <= x < width and 0 <= y < height and 0 <= z < depth
                world_obs[PLAYER_LOCATIONS, x, y, z] = CURRENT_PLAYER
                if y + 1 < height:
                    world_obs[PLAYER_LOCATIONS, x, y + 1, z] = CURRENT_PLAYER

            prev_world_obs = world_obs

    return repaired_episode
