from typing import Optional, cast

import numpy as np
from ray.rllib.evaluation import SampleBatch

from mbag.agents.action_distributions import MbagActionDistribution
from mbag.environment.blocks import MinecraftBlocks
from mbag.environment.config import MbagConfigDict
from mbag.environment.types import (
    CURRENT_BLOCKS,
    GOAL_BLOCKS,
    MbagInventoryObs,
    MbagObs,
    MbagWorldObsArray,
    num_world_obs_channels,
)


def randomly_permute_block_types(
    batch: SampleBatch,
    *,
    flat_actions=False,
    flat_observations=False,
    env_config: Optional[MbagConfigDict] = None,
    keep_dirt_at_ground_level=False,
) -> SampleBatch:
    new_batch = batch.copy()

    if flat_actions:
        if env_config is None:
            raise ValueError("env_config must be provided if flat_actions is True")
        action_mapping = MbagActionDistribution.get_action_mapping(env_config)
        old_action_block_locations = action_mapping[batch[SampleBatch.ACTIONS]][:, 0]
        old_action_block_ids = action_mapping[batch[SampleBatch.ACTIONS]][:, 2]
        if SampleBatch.PREV_ACTIONS in batch:
            old_prev_action_block_locations = action_mapping[
                batch[SampleBatch.PREV_ACTIONS]
            ][:, 0]
            old_prev_action_block_ids = action_mapping[batch[SampleBatch.PREV_ACTIONS]][
                :, 2
            ]
    else:
        old_action_block_locations = batch[SampleBatch.ACTIONS][0]
        old_action_block_ids = batch[SampleBatch.ACTIONS][2]
        if SampleBatch.PREV_ACTIONS in batch:
            old_prev_action_block_locations = batch[SampleBatch.PREV_ACTIONS][0]
            old_prev_action_block_ids = batch[SampleBatch.PREV_ACTIONS][2]
    new_action_block_ids = np.empty_like(old_action_block_ids)
    if SampleBatch.PREV_ACTIONS in batch:
        new_prev_action_block_ids = np.empty_like(old_prev_action_block_ids)

    if flat_observations:
        if env_config is None:
            raise ValueError("env_config must be provided if flat_observations is True")
        width, height, depth = env_config["world_size"]
        num_players = env_config["num_players"]
        world_obs_size = num_world_obs_channels * width * height * depth
        inventory_obs_size = num_players * MinecraftBlocks.NUM_BLOCKS
        new_world_obs = cast(
            MbagWorldObsArray,
            new_batch[SampleBatch.OBS][:, :world_obs_size].reshape(
                (-1, num_world_obs_channels, width, height, depth)
            ),
        )
        assert np.shares_memory(new_world_obs, new_batch[SampleBatch.OBS])
        new_inventory_obs = cast(
            MbagInventoryObs,
            new_batch[SampleBatch.OBS][
                :, world_obs_size : world_obs_size + inventory_obs_size
            ].reshape((-1, num_players, MinecraftBlocks.NUM_BLOCKS)),
        )
        assert np.shares_memory(new_inventory_obs, new_batch[SampleBatch.OBS])
        old_world_obs = cast(
            MbagWorldObsArray,
            batch[SampleBatch.OBS][:, :world_obs_size].reshape(
                (-1, num_world_obs_channels, width, height, depth)
            ),
        )
        assert np.shares_memory(old_world_obs, batch[SampleBatch.OBS])
        old_inventory_obs = cast(
            MbagInventoryObs,
            batch[SampleBatch.OBS][
                :, world_obs_size : world_obs_size + inventory_obs_size
            ].reshape((-1, num_players, MinecraftBlocks.NUM_BLOCKS)),
        )
        assert np.shares_memory(old_inventory_obs, batch[SampleBatch.OBS])
    else:
        new_world_obs, new_inventory_obs, _ = cast(MbagObs, new_batch[SampleBatch.OBS])
        old_world_obs, old_inventory_obs, _ = cast(MbagObs, batch[SampleBatch.OBS])
    assert new_world_obs is not old_world_obs
    assert new_inventory_obs is not old_inventory_obs

    placeable_block_ids = np.array(list(MinecraftBlocks.PLACEABLE_BLOCK_IDS))

    _, _, width, height, depth = old_world_obs.shape

    if SampleBatch.SEQ_LENS in batch:
        seq_lens = batch[SampleBatch.SEQ_LENS]
    else:
        seq_lens = np.full(len(batch), 1)

    seq_begin = 0
    for seq_len in seq_lens:
        seq_end = seq_begin + seq_len

        block_map = np.arange(MinecraftBlocks.NUM_BLOCKS)
        block_map[placeable_block_ids] = np.random.permutation(placeable_block_ids)
        inverse_block_map = np.argsort(block_map)

        # Permute current and goal blocks.
        new_world_obs[seq_begin:seq_end, CURRENT_BLOCKS] = block_map[
            old_world_obs[seq_begin:seq_end, CURRENT_BLOCKS]
        ]
        new_world_obs[seq_begin:seq_end, GOAL_BLOCKS] = block_map[
            old_world_obs[seq_begin:seq_end, GOAL_BLOCKS]
        ]

        # Permute inventories.
        new_inventory_obs[seq_begin:seq_end] = old_inventory_obs[
            seq_begin:seq_end, :, inverse_block_map
        ]

        # Permute actions.
        new_action_block_ids[seq_begin:seq_end] = block_map[
            old_action_block_ids[seq_begin:seq_end]
        ]
        if SampleBatch.PREV_ACTIONS in batch:
            new_prev_action_block_ids[seq_begin:seq_end] = block_map[
                old_prev_action_block_ids[seq_begin:seq_end]
            ]

        if keep_dirt_at_ground_level:
            if env_config is not None and not env_config["abilities"]["inf_blocks"]:
                raise ValueError(
                    "keep_dirt_at_ground_level only works with inf_blocks=True"
                )
            _, old_action_y, _ = np.unravel_index(
                old_action_block_locations[seq_begin:seq_end],
                (width, height, depth),
            )
            dirt = MinecraftBlocks.NAME2ID["dirt"]
            new_ground_level = new_world_obs[seq_begin:seq_end, :, :, 1, :]
            old_ground_level = old_world_obs[seq_begin:seq_end, :, :, 1, :]
            new_ground_level[:, CURRENT_BLOCKS] = np.where(
                old_ground_level[:, CURRENT_BLOCKS] == dirt,
                dirt,
                new_ground_level[:, CURRENT_BLOCKS],
            )
            new_ground_level[:, GOAL_BLOCKS] = np.where(
                old_ground_level[:, GOAL_BLOCKS] == dirt,
                dirt,
                new_ground_level[:, GOAL_BLOCKS],
            )
            actions_at_ground_level = old_action_y == 1
            new_action_block_ids[seq_begin:seq_end][actions_at_ground_level] = (
                old_action_block_ids[seq_begin:seq_end][actions_at_ground_level]
            )
            if SampleBatch.PREV_ACTIONS in batch:
                _, old_prev_action_y, _ = np.unravel_index(
                    old_prev_action_block_locations[seq_begin:seq_end],
                    (width, height, depth),
                )
                prev_actions_at_ground_level = old_prev_action_y == 1
                new_prev_action_block_ids[seq_begin:seq_end][
                    prev_actions_at_ground_level
                ] = old_prev_action_block_ids[seq_begin:seq_end][
                    prev_actions_at_ground_level
                ]

        seq_begin = seq_end

    assert seq_end == len(batch)

    if flat_actions:
        new_batch[SampleBatch.ACTIONS] += (
            (new_action_block_ids - old_action_block_ids) * width * height * depth
        )
        if SampleBatch.PREV_ACTIONS in batch:
            new_batch[SampleBatch.PREV_ACTIONS] += (
                (new_prev_action_block_ids - old_prev_action_block_ids)
                * width
                * height
                * depth
            )
    else:
        new_batch[SampleBatch.ACTIONS][2][:] = new_action_block_ids
        if SampleBatch.PREV_ACTIONS in batch:
            new_batch[SampleBatch.PREV_ACTIONS][2][:] = new_prev_action_block_ids

    return new_batch
