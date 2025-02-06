from __future__ import annotations

import copy
import logging
import random
import time
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple, cast

import numpy as np
from gymnasium import spaces
from typing_extensions import Literal

from .actions import MbagAction, MbagActionTuple
from .blocks import MinecraftBlocks
from .config import (
    DEFAULT_CONFIG,
    DEFAULT_PLAYER_CONFIG,
    MbagConfigDict,
    RewardsConfigDict,
)
from .goals import ALL_GOAL_GENERATORS
from .malmo.ai_actions import (
    MalmoGiveAIAction,
    MalmoMoveAIAction,
    MalmoPlaceBreakAIAction,
)
from .malmo.malmo_interface import MalmoInterface
from .malmo.malmo_state import MalmoState
from .schedule import ConstantSchedule, PiecewiseSchedule, Schedule
from .state import MbagStateDict
from .types import (
    CURRENT_BLOCK_STATES,
    CURRENT_BLOCKS,
    CURRENT_PLAYER,
    GOAL_BLOCK_STATES,
    GOAL_BLOCKS,
    INVENTORY_NUM_SLOTS,
    LAST_INTERACTED,
    NO_INTERACTION,
    OTHER_PLAYER,
    PLAYER_LOCATIONS,
    STACK_SIZE,
    BlockLocation,
    FacingDirection,
    MbagInfoDict,
    MbagInventory,
    MbagInventoryObs,
    MbagObs,
    MbagWorldObsArray,
    WorldLocation,
    get_block_counts_in_inventory,
    num_world_obs_channels,
)

logger = logging.getLogger(__name__)


class MbagEnv(object):
    config: MbagConfigDict
    current_blocks: MinecraftBlocks
    goal_blocks: MinecraftBlocks
    player_locations: List[WorldLocation]
    player_directions: List[FacingDirection]
    player_inventories: List[MbagInventory]
    palette_x: int
    last_interacted: np.ndarray
    previous_step_timestamp: float
    timestep: int
    global_timestep: int
    maximum_goal_percentages: List[float]
    timesteps_with_no_progress: int

    human_action_queues: List[List[MbagActionTuple]]
    """A queue for each player of human actions from MalmoInterface."""

    BLOCKS_TO_GIVE = 5
    """The number of blocks given in a GIVE_BLOCK action."""

    def __init__(self, config: MbagConfigDict):
        self.config = self.get_config(config)

        self.world_obs_shape = (num_world_obs_channels,) + self.config["world_size"]
        self.observation_space = spaces.Tuple(
            (
                spaces.Box(0, 255, self.world_obs_shape, dtype=np.uint8),
                spaces.Box(
                    0,
                    INVENTORY_NUM_SLOTS * STACK_SIZE,
                    (
                        self.config["num_players"],
                        MinecraftBlocks.NUM_BLOCKS,
                    ),
                    dtype=np.int32,
                ),
                spaces.Box(0, 0x7FFFFFFF, (), dtype=np.int32),
            )
        )

        # Actions consist of an (action_type, block_location, block_id) tuple.
        # Not all action types use block_location and block_id. See MbagAction for
        # more details.
        self.action_space = spaces.Tuple(
            (
                spaces.Discrete(MbagAction.NUM_ACTION_TYPES),
                spaces.Discrete(int(np.prod(self.config["world_size"]))),
                spaces.Discrete(MinecraftBlocks.NUM_BLOCKS),
            )
        )

        goal_generator = self.config["goal_generator"]
        goal_generator_config = self.config["goal_generator_config"]
        if isinstance(goal_generator, str):
            goal_generator_class = ALL_GOAL_GENERATORS[goal_generator]
        else:
            goal_generator_class = goal_generator
        self.goal_generator = goal_generator_class(goal_generator_config)

        if not self.config["abilities"]["flying"]:
            raise NotImplementedError("lack of flying ability is not yet implemented")

        if self.config["malmo"]["use_malmo"]:
            self.malmo_interface = MalmoInterface(self.config)

        self.global_timestep = 0

        self.is_first_episode = True
        self.any_step_since_last_reset = True

        # Initialize reward schedules.
        self._reward_schedules: List[Dict[str, Schedule]] = []
        for player_index in range(self.config["num_players"]):
            player_reward_schedule: Dict[str, Schedule] = {}
            reward_config = self._get_reward_config_for_player(player_index)
            for key, value in reward_config.items():
                if isinstance(value, list):
                    player_reward_schedule[key] = PiecewiseSchedule(
                        endpoints=value, outside_value=value[-1][-1]
                    )
                elif isinstance(value, (float, int)):
                    player_reward_schedule[key] = ConstantSchedule(float(value))
                else:
                    raise ValueError(f"Invalid reward config for {key}: {value}")
            self._reward_schedules.append(player_reward_schedule)

    @staticmethod
    def get_config(partial_config: MbagConfigDict) -> MbagConfigDict:
        """Get a fully populated config dict by adding defaults where necessary."""

        partial_config = copy.deepcopy(partial_config)
        config = copy.deepcopy(DEFAULT_CONFIG)
        config.update(partial_config)
        if isinstance(config["world_size"], list):
            config["world_size"] = tuple(config["world_size"])

        config["malmo"] = copy.deepcopy(DEFAULT_CONFIG["malmo"])
        config["malmo"].update(partial_config.get("malmo", {}))

        config["rewards"] = copy.deepcopy(DEFAULT_CONFIG["rewards"])
        config["rewards"].update(partial_config.get("rewards", {}))

        if len(config["players"]) != config["num_players"]:
            raise ValueError(
                f"MBAG config dictionary specifies {config['num_players']} player(s) "
                f"but has configuration for {len(config['players'])} player(s)"
            )

        for player_index, partial_player_config in list(enumerate(config["players"])):
            player_config = copy.deepcopy(DEFAULT_PLAYER_CONFIG)
            player_config.update(partial_player_config)

            partial_rewards_config = player_config["rewards"]
            player_config["rewards"] = copy.deepcopy(config["rewards"])
            player_config["rewards"].update(partial_rewards_config)

            if player_config["is_human"] and not config["malmo"]["use_malmo"]:
                logger.warning(
                    f"Player {player_index} is specified as human but Malmo is not "
                    "enabled"
                )

            config["players"][player_index] = player_config

        if (
            config["malmo"]["video_dir"] is not None
            and not config["malmo"]["use_spectator"]
        ):
            raise ValueError("Video recording requires using a spectator")

        return config

    def update_global_timestep(self, global_timestep: int) -> None:
        # TODO: remove?
        self.global_timestep = global_timestep

    def _randomly_place_players(self):
        width, height, depth = self.config["world_size"]
        self.player_locations = []
        for player_index in range(self.config["num_players"]):
            player_location: WorldLocation = (-1, -1, -1)
            # Generate random locations until we find a valid one.
            while not self._is_valid_player_space(player_location, player_index):
                player_location = (
                    random.randrange(width) + 0.5,
                    random.randrange(height),
                    random.randrange(depth) + 0.5,
                )
            self.player_locations.append(player_location)

    def reset(
        self, *, force_regenerate_goal=False
    ) -> Tuple[List[MbagObs], List[MbagInfoDict]]:
        """Reset Minecraft environment and return player observations for each player."""

        if self.config["malmo"]["use_malmo"]:
            # End any previous Malmo episode before starting the new one.
            self.malmo_interface.end_episode()

        if self.is_first_episode and self.config.get(
            "randomize_first_episode_length", False
        ):
            self.timestep = random.randrange(self.config["horizon"])
        else:
            self.timestep = 0

        self.current_blocks = MinecraftBlocks(self.config["world_size"])
        self.current_blocks.blocks[:, 0, :] = MinecraftBlocks.BEDROCK
        self.current_blocks.blocks[:, 1, :] = MinecraftBlocks.NAME2ID["dirt"]

        self.last_interacted = np.zeros(self.config["world_size"], dtype=np.int8)
        self.last_interacted[:] = NO_INTERACTION

        if force_regenerate_goal or self.any_step_since_last_reset:
            # Generating goals is expensive, so don't do it if there haven't been
            # any steps taken since the last reset (unless force_regenerate_goal
            # is True).
            self.goal_blocks = self._generate_goal()
            self.any_step_since_last_reset = False

        # Place players in the world.
        if self.config["random_start_locations"]:
            self._randomly_place_players()
        else:
            self.player_locations = [
                (
                    (i % self.config["world_size"][0]) + 0.5,
                    2,
                    int(i / self.config["world_size"][0]) + 0.5,
                )
                for i in range(self.config["num_players"])
            ]
        self.player_directions = [(0, 0) for _ in range(self.config["num_players"])]
        self.player_inventories = [
            np.zeros((INVENTORY_NUM_SLOTS, 2), dtype=np.int32)
            for _ in range(self.config["num_players"])
        ]

        # Set initial inventory if the user has infinite blocks
        if self.config["abilities"]["inf_blocks"]:
            for i in range(self.config["num_players"]):
                if not self.config["players"][i]["is_human"]:
                    continue

                for j in range(2, 10):
                    self.player_inventories[i][j][0] = j
                    self.player_inventories[i][j][1] = 1

        if not self.config["abilities"]["inf_blocks"]:
            self._copy_palette_from_goal()

        self.initial_goal_similarities: List[float] = []
        for player_index in range(self.config["num_players"]):
            self.initial_goal_similarities.append(
                self._get_goal_similarity(
                    self.current_blocks[:],
                    self.goal_blocks[:],
                    partial_credit=True,
                    player_index=player_index,
                ).sum()
            )
        width, height, depth = self.config["world_size"]
        self.max_goal_similarity = width * height * depth

        if self.config["malmo"]["use_malmo"]:
            self.malmo_interface.reset(
                self.current_blocks,
                self.goal_blocks,
                self.player_locations,
            )
            self.human_action_queues = [[] for _ in range(self.config["num_players"])]

        self.previous_step_timestamp = time.time()

        obs_list = [
            self._get_player_obs(player_index)
            for player_index in range(self.config["num_players"])
        ]
        info_list = [
            self._get_player_info(player_index)
            for player_index in range(self.config["num_players"])
        ]

        self.maximum_goal_percentages = [info["goal_percentage"] for info in info_list]
        self.timesteps_with_no_progress = 0

        return obs_list, info_list

    def step(
        self, action_tuples: List[MbagActionTuple]
    ) -> Tuple[List[MbagObs], List[float], List[bool], List[MbagInfoDict]]:
        assert (
            len(action_tuples) == self.config["num_players"]
        ), "Wrong number of actions."

        reward: float = 0
        own_rewards: List[float] = [0 for _ in range(self.config["num_players"])]
        optional_infos: List[Optional[MbagInfoDict]] = [
            None for _ in range(self.config["num_players"])
        ]

        # Process give block actions before movement actions
        action_tuples_sorted_labeled = sorted(
            list(enumerate(action_tuples)),
            key=lambda player_index_action_tuple: (
                0 if player_index_action_tuple[1][0] == MbagAction.GIVE_BLOCK else 1
            ),
        )

        for player_index, player_action_tuple in action_tuples_sorted_labeled:
            # For each player, if they are acting this timestep, step the player,
            # otherwise execute NOOP.
            if (
                self.timestep % self.config["players"][player_index]["timestep_skip"]
                == 0
            ):
                player_reward, player_info = self._step_player(
                    player_index, player_action_tuple
                )
            else:
                player_reward, player_info = self._step_player(
                    player_index,
                    (MbagAction.NOOP, 0, 0),
                )
            reward += player_reward
            own_rewards[player_index] = player_reward
            optional_infos[player_index] = player_info

        infos: List[MbagInfoDict] = []
        for player_index, info in enumerate(optional_infos):
            assert info is not None
            info["goal_similarity"] = self._get_goal_similarity(
                self.current_blocks[:],
                self.goal_blocks[:],
            ).sum()
            info["goal_percentage"] = self._get_goal_percentage(player_index)
            infos.append(info)

        if self.config["malmo"]["use_malmo"]:
            self.malmo_interface.update_goal_percentage(infos[0]["goal_percentage"])
            self._wait_for_malmo_and_sync()
            self._add_human_actions_and_malmo_observations_to_info(infos)

        self.previous_step_timestamp = time.time()

        self.timestep += 1

        if (
            self.current_blocks.blocks[self.palette_x]
            != self.goal_blocks.blocks[self.palette_x]
        ).any() and not self.config["abilities"]["inf_blocks"]:
            self._copy_palette_from_goal()
            if self.config["malmo"]["use_malmo"]:
                logger.info("copying palette from goal")
                self.malmo_interface.copy_palette_from_goal()

        # Update timesteps_with_no_progress.
        self.timesteps_with_no_progress += 1
        for player_index, info in enumerate(infos):
            previous_max_goal_percentage = self.maximum_goal_percentages[player_index]
            if info["goal_percentage"] > previous_max_goal_percentage:
                self.maximum_goal_percentages[player_index] = info["goal_percentage"]
                self.timesteps_with_no_progress = 0

        obs = [
            self._get_player_obs(player_index)
            for player_index in range(self.config["num_players"])
        ]
        rewards = [
            self._get_player_reward(player_index, reward, own_reward)
            for player_index, own_reward in enumerate(own_rewards)
        ]
        dones = [self._done()] * self.config["num_players"]

        if dones[0] and self.config["malmo"]["use_malmo"]:
            self.malmo_interface.end_episode()

        if dones[0]:
            self.is_first_episode = False
        self.any_step_since_last_reset = True

        return obs, rewards, dones, infos

    def _generate_goal(self) -> MinecraftBlocks:
        # Generate a goal with buffer of at least 1 on the sides, top, and bottom.
        world_size = self.config["world_size"]

        goal_size = (world_size[0] - 2, world_size[1] - 2, world_size[2] - 2)
        if not self.config["abilities"]["inf_blocks"]:
            goal_size = (world_size[0] - 3, world_size[1] - 2, world_size[2] - 2)

        self.palette_x = world_size[0] - 1

        small_goal = self.goal_generator.generate_goal(goal_size)

        goal = self.current_blocks.copy()

        shape = small_goal.size

        goal.blocks[1 : shape[0] + 1, 1 : shape[1] + 1, 1 : shape[2] + 1] = (
            small_goal.blocks
        )
        goal.block_states[1 : shape[0] + 1, 1 : shape[1] + 1, 1 : shape[2] + 1] = (
            small_goal.block_states
        )

        if not self.config["abilities"]["inf_blocks"]:
            for index, block in enumerate(MinecraftBlocks.PLACEABLE_BLOCK_IDS):
                if index >= world_size[2]:
                    break
                goal.blocks[self.palette_x, 2, index] = block
                goal.block_states[self.palette_x, 2, index] = 0

        # logger.debug(goal.blocks)
        return goal

    def _copy_palette_from_goal(self):
        # Copy over the palette from the goal generator
        palette_blocks = self.goal_blocks.blocks[self.palette_x]
        palette_block_states = self.goal_blocks.block_states[self.palette_x]
        self.current_blocks.blocks[self.palette_x] = palette_blocks
        self.current_blocks.block_states[self.palette_x] = palette_block_states

    def _step_player(
        self, player_index: int, action_tuple: MbagActionTuple
    ) -> Tuple[float, MbagInfoDict]:
        action = MbagAction(action_tuple, self.config["world_size"])
        goal_dependent_reward = 0.0
        goal_independent_reward = 0.0

        noop: bool = True
        # marks if an action 'correct' meaning it directly contributed to the goal
        action_correct: bool = False

        if action.action_type == MbagAction.NOOP:
            pass
        elif action.action_type in [MbagAction.PLACE_BLOCK, MbagAction.BREAK_BLOCK]:
            prev_block = self.current_blocks[action.block_location]
            prev_inventory_block_counts = get_block_counts_in_inventory(
                self.player_inventories[player_index]
            )
            noop = not self._handle_place_break(player_index, action)

            # Calculate reward based on progress towards goal.
            if (
                action.action_type == MbagAction.BREAK_BLOCK
                and action.block_location[0] == self.palette_x
                and not self.config["abilities"]["inf_blocks"]
            ):
                # Palette block was broken.
                # TODO: shouldn't we check if the user actually broke the block?
                # might be worth adding a test to make sure the reward only comes
                # through if they did
                new_inventory_block_counts = get_block_counts_in_inventory(
                    self.player_inventories[player_index]
                )
                goal_independent_reward += (
                    np.count_nonzero(new_inventory_block_counts)
                    - np.count_nonzero(prev_inventory_block_counts)
                ) * self._get_reward(
                    player_index, "get_resources", self.global_timestep
                )
            else:
                # Non-palette block was placed or broken.
                new_block = self.current_blocks[action.block_location]
                goal_block = self.goal_blocks[action.block_location]
                prev_goal_similarity = self._get_goal_similarity(
                    prev_block,
                    goal_block,
                    partial_credit=True,
                    player_index=player_index,
                )
                new_goal_similarity = self._get_goal_similarity(
                    new_block,
                    goal_block,
                    partial_credit=True,
                    player_index=player_index,
                )
                goal_dependent_reward += new_goal_similarity - prev_goal_similarity
                action_correct = (
                    action.action_type == MbagAction.PLACE_BLOCK
                    and goal_dependent_reward > 0
                ) or (
                    action.action_type == MbagAction.BREAK_BLOCK
                    and goal_dependent_reward >= 0
                )
                if not noop and not action_correct:
                    goal_dependent_reward += self._get_reward(
                        player_index, "incorrect_action", self.global_timestep
                    )
        elif (
            action.action_type in MbagAction.MOVE_ACTION_TYPES
            and not self.config["abilities"]["teleportation"]
        ):
            noop = not self._handle_move(player_index, action)
        elif (
            action.action_type == MbagAction.GIVE_BLOCK
            and not self.config["abilities"]["inf_blocks"]
        ):
            noop = 0 == self._handle_give_block(player_index, action)

        if noop:
            goal_independent_reward += self._get_reward(
                player_index, "noop", self.global_timestep
            )
        else:
            goal_independent_reward += self._get_reward(
                player_index, "action", self.global_timestep
            )

        reward = goal_dependent_reward + goal_independent_reward

        info = self._get_player_info(
            player_index,
            goal_dependent_reward=goal_dependent_reward,
            goal_independent_reward=goal_independent_reward,
            own_reward=reward,
            attempted_action=action,
            action=action if not noop else MbagAction.noop_action(),
            action_correct=action_correct and not noop,
            # goal_similarity and goal_percentage get updated later in the step()
            # method once all players' actions have been processed.
            include_goal_similarity_and_goal_percentage=False,
        )

        return reward, info

    def _handle_place_break(self, player_index: int, action: MbagAction) -> bool:
        # Check if the player can give/receive the block.
        if not self.config["abilities"]["inf_blocks"]:
            if action.action_type == MbagAction.PLACE_BLOCK:
                if not self._try_take_player_blocks(
                    action.block_id, player_index, False
                ):
                    return False
            elif action.action_type == MbagAction.BREAK_BLOCK:
                broken_block_id = self.current_blocks.blocks[action.block_location]
                if not self._try_give_player_blocks(
                    broken_block_id, player_index, False
                ):
                    return False

        # Try to place or break a block.
        if self.config["abilities"]["teleportation"]:
            place_break_result = self.current_blocks.try_break_place(
                cast(Literal[1, 2], action.action_type),
                action.block_location,
                action.block_id,
                random_seed=self.timestep,
            )
        else:
            if self._collides_with_players(
                action.block_location, player_index, check_below_feet=False
            ):
                place_break_result = None
            else:
                place_break_result = self.current_blocks.try_break_place(
                    cast(Literal[1, 2], action.action_type),
                    action.block_location,
                    action.block_id,
                    player_location=self.player_locations[player_index],
                    other_player_locations=self.player_locations[:player_index]
                    + self.player_locations[player_index + 1 :],
                    is_human=self.config["players"][player_index]["is_human"],
                    random_seed=self.timestep,
                )

        if place_break_result is None:
            return False

        player_location, click_location = place_break_result
        if self.config["abilities"]["teleportation"]:
            self.player_locations[player_index] = player_location
        self.last_interacted[action.block_location] = player_index

        if not self.config["abilities"]["inf_blocks"]:
            # Actually give or take the block.
            if action.action_type == MbagAction.PLACE_BLOCK:
                assert self._try_take_player_blocks(action.block_id, player_index, True)
            elif action.action_type == MbagAction.BREAK_BLOCK:
                assert self._try_give_player_blocks(broken_block_id, player_index, True)

        if (
            self.config["malmo"]["use_malmo"]
            and not self.config["players"][player_index]["is_human"]
        ):
            viewpoint = np.array(player_location)
            viewpoint[1] += 1.6
            delta = np.array(click_location) - viewpoint
            delta /= np.sqrt((delta**2).sum())
            yaw = np.rad2deg(np.arctan2(-delta[0], delta[2]))
            pitch = np.rad2deg(-np.arcsin(delta[1]))
            self.player_directions[player_index] = (yaw, pitch)

            self.malmo_interface.queue_ai_action(
                player_index,
                MalmoPlaceBreakAIAction(
                    action,
                    yaw,
                    pitch,
                    (
                        player_location
                        if self.config["abilities"]["teleportation"]
                        else None
                    ),
                ),
            )

        return True

    def _handle_move(self, player_index: int, action: MbagAction) -> bool:
        """
        Handle a move action.
        Returns whether the action was successful or not
        """

        action_type = action.action_type
        player_x, player_y, player_z = self.player_locations[player_index]
        # Snap player to center of block.
        player_x = int(player_x) + 0.5
        player_y = int(player_y)
        player_z = int(player_z) + 0.5
        dx, dy, dz = MbagAction.MOVE_ACTION_DELTAS[action_type]
        new_player_location: WorldLocation = (
            player_x + dx,
            player_y + dy,
            player_z + dz,
        )

        if not self._is_valid_player_space(
            new_player_location,
            player_index,
            ignore_other_players=self.config["players"][player_index]["is_human"],
        ):
            return False

        self.player_locations[player_index] = new_player_location

        if (
            self.config["malmo"]["use_malmo"]
            and not self.config["players"][player_index]["is_human"]
        ):
            self.malmo_interface.queue_ai_action(
                player_index, MalmoMoveAIAction(action, new_player_location)
            )

        return True

    def _handle_give_block(self, giver_player_index: int, action: MbagAction) -> int:
        """
        Handles giving blocks to a player.
        """

        block_id = action.block_id
        recipient_location = (
            action.block_location[0] + 0.5,
            action.block_location[1],
            action.block_location[2] + 0.5,
        )

        # Check if non-human players can reach the location specified (has to be within one block
        # in all directions).
        if not self.config["players"][giver_player_index]["is_human"]:
            gx, gy, gz = self.player_locations[giver_player_index]
            rx, ry, rz = recipient_location
            if not self.config["abilities"]["teleportation"] and (
                abs(gx - rx) > 1 or abs(gy - ry) > 1 or abs(gz - rz) > 1
            ):
                return 0

        # Find player index at the location specified
        recipient_player_index: Optional[int] = None
        for player_index in range(self.config["num_players"]):
            player_x, player_y, player_z = self.player_locations[player_index]
            if (
                abs(player_x - recipient_location[0]) < 0.5
                and abs(player_y - recipient_location[1]) < 0.5
                and abs(player_z - recipient_location[2]) < 0.5
            ):
                recipient_player_index = player_index
                break
        if (
            recipient_player_index is None
            or recipient_player_index == giver_player_index
        ):
            return 0

        if self.config["players"][giver_player_index]["is_human"]:
            blocks_to_give = 1
        else:
            block_counts = get_block_counts_in_inventory(
                self.player_inventories[giver_player_index]
            )
            blocks_to_give = min(self.BLOCKS_TO_GIVE, block_counts[block_id])

        if not self.config["abilities"]["inf_blocks"]:
            if not (
                self._try_take_player_blocks(
                    block_id, giver_player_index, False, blocks_to_give
                )
                and self._try_give_player_blocks(
                    block_id, recipient_player_index, False, blocks_to_give
                )
            ):
                return 0

            assert self._try_take_player_blocks(
                block_id, giver_player_index, True, blocks_to_give
            )
            assert self._try_give_player_blocks(
                block_id, recipient_player_index, True, blocks_to_give
            )

        if (
            self.config["malmo"]["use_malmo"]
            and not self.config["players"][giver_player_index]["is_human"]
        ):
            self.malmo_interface.queue_ai_action(
                giver_player_index,
                MalmoGiveAIAction(action, recipient_player_index, blocks_to_give),
            )

        return blocks_to_give

    def _get_inventory_obs(self, player_index: int) -> MbagInventoryObs:
        """
        Gets the array representation of the given player's inventory.
        """

        inventory_obs = np.zeros(
            (self.config["num_players"], MinecraftBlocks.NUM_BLOCKS), dtype=np.int32
        )

        player_indices = [player_index] + [
            other_player_index
            for other_player_index in range(self.config["num_players"])
            if other_player_index != player_index
        ]
        for player_tag, player_index in enumerate(player_indices):
            player_inventory = self.player_inventories[player_index]
            inventory_obs[player_tag] = get_block_counts_in_inventory(player_inventory)

        return inventory_obs

    def _try_give_player_blocks(
        self, block_id: int, player_index: int, give: bool, num_to_give: int = 1
    ) -> bool:
        """
        Attempts to give to player_index the given number of blocks of type block_id.
        Returns whether the give was successful. If give is False, then the inventory
        slot is not incremented.
        """

        player_inventory = self.player_inventories[player_index]
        # Slots must be empty or of the same type.
        available_slots = (player_inventory[:, 0] == block_id) | (
            player_inventory[:, 1] == 0
        )
        available_space = np.sum(STACK_SIZE - player_inventory[available_slots, 1])

        if available_space < num_to_give:
            return False
        else:
            if give:
                left_to_give = num_to_give
                for slot in np.where(available_slots)[0]:
                    player_inventory[slot, 0] = block_id
                    if left_to_give <= STACK_SIZE - player_inventory[slot, 1]:
                        player_inventory[slot, 1] += left_to_give
                        left_to_give = 0
                        break
                    else:
                        left_to_give -= STACK_SIZE - player_inventory[slot, 1]
                        player_inventory[slot, 1] = STACK_SIZE
                assert left_to_give == 0
            return True

    def _try_take_player_blocks(
        self,
        block_id: int,
        player_index: int,
        take: bool,
        num_to_take: int = 1,
    ) -> bool:
        """
        Attempts to take from player_index the given number of blocks of type block_id.
        Returns whether the take was successful. If take is False, then the inventory
        slot is not decremented.
        """
        player_inventory = self.player_inventories[player_index]
        total_blocks = np.sum(player_inventory[player_inventory[:, 0] == block_id, 1])
        if total_blocks < num_to_take:
            return False
        else:
            if take:
                matching_inventory_slots = np.where(
                    (player_inventory[:, 0] == block_id) & (player_inventory[:, 1] > 0)
                )[0]
                left_to_take = num_to_take
                for slot in matching_inventory_slots:
                    if left_to_take <= player_inventory[slot, 1]:
                        player_inventory[slot, 1] -= left_to_take
                        left_to_take = 0
                        break
                    else:
                        left_to_take -= player_inventory[slot, 1]
                        player_inventory[slot, 1] = 0
                assert left_to_take == 0
            return True

    def _is_valid_player_space(
        self,
        player_location: WorldLocation,
        player_index: int,
        ignore_other_players: bool = False,
    ) -> bool:
        proposed_block: BlockLocation = (
            int(np.floor(player_location[0])),
            int(np.floor(player_location[1])),
            int(np.floor(player_location[2])),
        )
        # Check if block is out of bounds.
        for i in range(3):
            if (
                proposed_block[i] < 0
                or proposed_block[i] >= self.config["world_size"][i]
            ):
                return False

        if not self.current_blocks.blocks[proposed_block] == MinecraftBlocks.AIR:
            return False

        if proposed_block[1] < self.config["world_size"][1] - 1:
            if (
                not self.current_blocks.blocks[
                    proposed_block[0], proposed_block[1] + 1, proposed_block[2]
                ]
                == MinecraftBlocks.AIR
            ):
                return False

        if ignore_other_players:
            return True
        else:
            return not self._collides_with_players(proposed_block, player_index)

    def _collides_with_players(
        self, proposed_block, player_id: int, check_below_feet: bool = True
    ) -> bool:
        for i in range(len(self.player_locations)):
            if i == player_id:
                continue

            player_x, player_y, player_z = self.player_locations[i]
            block_x, block_y, block_z = proposed_block
            colliding_y_locations = [player_y, player_y + 1]
            if check_below_feet:
                colliding_y_locations.append(player_y - 1)
            if (
                block_x == player_x - 0.5
                and block_z == player_z - 0.5
                and block_y in colliding_y_locations
            ):
                return True

        return False

    def _get_goal_similarity(
        self,
        current_block: Tuple[np.ndarray, np.ndarray],
        goal_block: Tuple[np.ndarray, np.ndarray],
        partial_credit: bool = False,
        player_index: Optional[int] = None,
    ):
        """
        Get the similarity between this block and the goal block, used to calculate
        the reward. The reward is the different between this value before and after the
        player's action.
        """

        current_block_id, current_block_state = current_block
        goal_block_id, goal_block_state = goal_block

        similarity = np.zeros(
            np.broadcast(current_block_id, goal_block_id).shape, dtype=float
        )
        if partial_credit:
            # Give partial credit for placing the wrong block type.
            assert player_index is not None
            similarity[
                (goal_block_id != MinecraftBlocks.AIR)
                & (current_block_id != MinecraftBlocks.AIR)
            ] = self._get_reward(player_index, "place_wrong", self.global_timestep)
        similarity[goal_block_id == current_block_id] = 1
        return similarity

    def _get_goal_percentage(self, player_index: int) -> float:
        similarity: float = self._get_goal_similarity(
            self.current_blocks[:],
            self.goal_blocks[:],
            partial_credit=True,
            player_index=player_index,
        ).sum()
        return (similarity - self.initial_goal_similarities[player_index]) / (
            self.max_goal_similarity - self.initial_goal_similarities[player_index]
        )

    def _get_player_obs(self, player_index: int) -> MbagObs:
        world_obs = np.zeros(self.world_obs_shape, np.uint8)
        world_obs[CURRENT_BLOCKS] = self.current_blocks.blocks
        world_obs[CURRENT_BLOCK_STATES] = self.current_blocks.block_states
        if self.config["players"][player_index]["goal_visible"]:
            world_obs[GOAL_BLOCKS] = self.goal_blocks.blocks
            world_obs[GOAL_BLOCK_STATES] = self.goal_blocks.block_states

        # Player markers for the observation: the current player is marked with 1
        # and then other players are marked starting with 2, 3, ...
        player_marker_map = {
            player_index: CURRENT_PLAYER,
            **{
                other_player_index: other_player_marker + OTHER_PLAYER
                for other_player_marker, other_player_index in enumerate(
                    list(range(player_index))
                    + list(range(player_index + 1, self.config["num_players"]))
                )
            },
        }

        # Add locations to the observation if the locations are actually meaningful
        # (i.e., if players do not have teleportation abilities).
        if not self.config["abilities"]["teleportation"]:
            check_for_overlap = self.config["_check_for_overlapping_players"] and not (
                self.config["malmo"]["use_malmo"]
                or any(
                    player_config["is_human"]
                    for player_config in self.config["players"]
                )
            )

            for other_player_index, other_player_location in enumerate(
                self.player_locations
            ):
                if other_player_index == player_index:
                    continue
                self._add_player_location_to_world_obs(
                    world_obs,
                    other_player_location,
                    player_marker_map[other_player_index],
                    check_for_overlap=check_for_overlap,
                )

            # Always add the current player's location to the observation last
            # to ensure that will overwrite any other player's location.
            self._add_player_location_to_world_obs(
                world_obs,
                self.player_locations[player_index],
                CURRENT_PLAYER,
                check_for_overlap=check_for_overlap,
            )

        for other_player_index in range(self.config["num_players"]):
            world_obs[LAST_INTERACTED][self.last_interacted == other_player_index] = (
                player_marker_map[other_player_index]
            )

        return (
            world_obs,
            self._get_inventory_obs(player_index),
            np.array(self.timestep, dtype=np.int32),
        )

    def _get_player_info(
        self,
        player_index: int,
        goal_dependent_reward: float = 0,
        goal_independent_reward: float = 0,
        own_reward: float = 0,
        attempted_action: MbagAction = MbagAction.noop_action(),
        action: MbagAction = MbagAction.noop_action(),
        action_correct: bool = False,
        include_goal_similarity_and_goal_percentage: bool = True,
    ) -> MbagInfoDict:
        info: MbagInfoDict = {
            "goal_similarity": (
                self._get_goal_similarity(
                    self.current_blocks[:],
                    self.goal_blocks[:],
                ).sum()
                if include_goal_similarity_and_goal_percentage
                else np.nan
            ),
            "goal_percentage": (
                self._get_goal_percentage(player_index)
                if include_goal_similarity_and_goal_percentage
                else np.nan
            ),
            "goal_dependent_reward": goal_dependent_reward,
            "goal_independent_reward": goal_independent_reward,
            "own_reward": own_reward,
            "own_reward_prop": self._get_own_reward_prop(player_index),
            "attempted_action": attempted_action,
            "action": action,
            "action_correct": action_correct,
            "malmo_observations": [],
            "human_action": (MbagAction.NOOP, 0, 0),
            "timestamp": datetime.now(),
        }
        return info

    def _add_player_location_to_world_obs(
        self,
        world_obs: MbagWorldObsArray,
        player_location: WorldLocation,
        marker: int,
        check_for_overlap: bool = True,
    ):
        x, y_feet, z = player_location
        x, y_feet, z = int(np.floor(x)), int(np.floor(y_feet)), int(np.floor(z))
        for y in (
            [y_feet, y_feet + 1]
            if y_feet + 1 < self.config["world_size"][1]
            else [y_feet]
        ):
            if check_for_overlap:
                if world_obs[PLAYER_LOCATIONS, x, y, z] != 0:
                    raise PlayersOverlappingError()
            world_obs[PLAYER_LOCATIONS, x, y, z] = marker

    def _get_reward(
        self, player_index: int, reward: str, global_timestep: int
    ) -> float:
        return self._reward_schedules[player_index][reward].value(global_timestep)

    def _get_reward_config_for_player(self, player_index: int) -> RewardsConfigDict:
        return self.config["players"][player_index]["rewards"]

    def _get_own_reward_prop(self, player_index: int) -> float:
        return self._get_reward(player_index, "own_reward_prop", self.global_timestep)

    def _get_player_reward(
        self, player_index: int, reward: float, own_reward: float
    ) -> float:
        own_reward_prop = self._get_own_reward_prop(player_index)
        return own_reward_prop * own_reward + (1 - own_reward_prop) * reward

    def _wait_for_malmo_and_sync(self):
        """
        Wait until action_delay seconds have passed since the end of the previous
        environment timestep. However, if any human actions are detected, then the
        delay is cut short in order to process the human actions. Any detected human
        actions are added to self.human_action_queues.

        If no human actions are detected and no AI actions are running after the delay,
        then update the environment state from the latest Malmo state.
        """

        if (
            time.time() - self.previous_step_timestamp
            > self.config["malmo"]["action_delay"]
        ):
            logger.warn(
                "environment step took longer than Malmo action_delay; action_delay "
                "may need to be increased to achieve consistent step rate"
            )

        any_human_actions = False
        while True:
            human_actions, malmo_state = (
                self.malmo_interface.get_human_actions_and_malmo_state()
            )
            for player_index, action_tuple in human_actions:
                self.human_action_queues[player_index].append(action_tuple)
            for player_index, action_queue in enumerate(self.human_action_queues):
                if (
                    len(action_queue) > 0
                    and self.config["players"][player_index]["is_human"]
                ):
                    any_human_actions = True
            if any_human_actions or (
                time.time() - self.previous_step_timestamp
                >= self.config["malmo"]["action_delay"]
            ):
                break
            time.sleep(0.01)

        if not any_human_actions and self.malmo_interface.running_ai_actions():
            logger.warning(
                "AI actions did not finish during action delay; environment and "
                "Malmo may be out of sync"
            )

        if not any_human_actions and not self.malmo_interface.running_ai_actions():
            # If there are no pending human actions coming from Malmo and no
            # pending AI actions going to Malmo, then Malmo and the environment
            # should be in sync. We call _update_state_from_malmo to ensure that
            # any remaining discrepancies are resolved.
            self._update_state_from_malmo(malmo_state)

    def _add_human_actions_and_malmo_observations_to_info(
        self, infos: List[MbagInfoDict]
    ):
        malmo_observations = self.malmo_interface.get_malmo_observations()

        for player_index, info in enumerate(infos):
            human_action_tuple: MbagActionTuple = (MbagAction.NOOP, 0, 0)
            if len(self.human_action_queues[player_index]) > 0:
                human_action_tuple = self.human_action_queues[player_index].pop(0)
            info["human_action"] = human_action_tuple
            if (
                human_action_tuple[0] != MbagAction.NOOP
                and not self.config["players"][player_index]["is_human"]
            ):
                human_action = MbagAction(human_action_tuple, self.config["world_size"])
                logger.warning(
                    f"received human action for non-human player {player_index}: "
                    f"{human_action}"
                )

            info["malmo_observations"] = [
                (timestamp, malmo_observation)
                for timestamp, observation_player_index, malmo_observation in malmo_observations
                if observation_player_index == player_index
            ]

    def _update_state_from_malmo(self, malmo_state: MalmoState):
        self._update_blocks_from_malmo(malmo_state)
        self._update_player_inventories_from_malmo(malmo_state)
        self._update_player_locations_from_malmo(malmo_state)

    def _update_blocks_from_malmo(self, malmo_state: MalmoState):
        malmo_blocks = malmo_state.blocks

        location: BlockLocation
        for location in cast(
            Sequence[BlockLocation],
            map(
                tuple,
                np.argwhere((malmo_blocks.blocks != self.current_blocks.blocks)),
            ),
        ):
            logger.warning(
                f"block discrepancy at {location}: "
                "expected "
                f"{MinecraftBlocks.ID2NAME[self.current_blocks.blocks[location]]} "
                f"but received "
                f"{MinecraftBlocks.ID2NAME[malmo_blocks.blocks[location]]} "
                "from Malmo"
            )

        self.current_blocks.blocks[:] = malmo_blocks.blocks

    def _update_player_inventories_from_malmo(
        self,
        malmo_state: MalmoState,
    ):
        if self.config["abilities"]["inf_blocks"]:
            pass
        else:
            for player_index in range(self.config["num_players"]):
                # Make sure inventory in Malmo matches up with what's in our inventory.
                player_inventory = self.player_inventories[player_index]
                malmo_inventory = malmo_state.player_inventories[player_index]
                for slot in np.nonzero(
                    np.any(malmo_inventory != player_inventory, axis=1)
                )[0]:
                    logger.warning(
                        f"inventory discrepancy at slot {slot}: "
                        f"expected {player_inventory[slot, 1]} x "
                        f"{MinecraftBlocks.ID2NAME[player_inventory[slot, 0]]} "
                        f"but received {malmo_inventory[slot, 1]} x "
                        f"{MinecraftBlocks.ID2NAME[malmo_inventory[slot, 0]]} "
                        "from Malmo"
                    )
                    player_inventory[slot] = malmo_inventory[slot]

    def _update_player_locations_from_malmo(self, malmo_state: MalmoState):
        if not self.config["abilities"]["teleportation"]:
            for player_index in range(self.config["num_players"]):
                if self.config["players"][player_index]["is_human"]:
                    # Location discrepancies for human players are handled by the
                    # human action detector in MalmoInterface.
                    continue

                # Make sure position is as expected.
                malmo_location = malmo_state.player_locations[player_index]
                if any(
                    abs(malmo_coord - stored_coord) > 1e-4
                    for malmo_coord, stored_coord in zip(
                        malmo_location, self.player_locations[player_index]
                    )
                ):
                    logger.warning(
                        f"location discrepancy for player {player_index}: "
                        f"expected {self.player_locations[player_index]} but received "
                        f"{malmo_location} from Malmo"
                    )
                    self.player_locations[player_index] = malmo_location

    def _done(self) -> bool:
        done = self.timestep >= self.config["horizon"]
        if self.config["terminate_on_goal_completion"]:
            done = done or self.current_blocks == self.goal_blocks
        if self.config["truncate_on_no_progress_timesteps"] is not None:
            done = (
                done
                or self.timesteps_with_no_progress
                >= self.config["truncate_on_no_progress_timesteps"]
            )
        return done

    def get_state(self) -> MbagStateDict:
        return {
            "current_blocks": self.current_blocks.copy(),
            "goal_blocks": self.goal_blocks.copy(),
            "player_locations": list(self.player_locations),
            "player_directions": list(self.player_directions),
            "player_inventories": [
                inventory.copy() for inventory in self.player_inventories
            ],
            "last_interacted": self.last_interacted.copy(),
            "timestep": self.timestep,
        }

    def set_state_no_obs(self, state: MbagStateDict) -> None:
        if self.config["malmo"]["use_malmo"]:
            raise RuntimeError("Cannot set state when using Malmo.")
        if self.config["truncate_on_no_progress_timesteps"]:
            raise RuntimeError(
                "Environment state does not include the information necessary "
                "to implement truncate_on_no_progress_timesteps=True."
            )

        if len(state["player_locations"]) != self.config["num_players"]:
            raise ValueError(
                f"player_locations has length {len(state['player_locations'])} "
                f"but expected {self.config['num_players']}"
            )
        if len(state["player_directions"]) != self.config["num_players"]:
            raise ValueError(
                f"player_directions has length {len(state['player_directions'])} "
                f"but expected {self.config['num_players']}"
            )
        if len(state["player_inventories"]) != self.config["num_players"]:
            raise ValueError(
                f"player_inventories has length {len(state['player_inventories'])} "
                f"but expected {self.config['num_players']}"
            )
        if state["last_interacted"].max() >= self.config["num_players"]:
            raise ValueError(
                f"last_interacted has maximum value {state['last_interacted'].max()} "
                f"but expected at most {self.config['num_players'] - 1}"
            )

        self.current_blocks = state["current_blocks"].copy()
        self.goal_blocks = state["goal_blocks"].copy()
        self.player_locations = list(state["player_locations"])
        self.player_directions = list(state["player_directions"])
        self.player_inventories = [
            inventory.copy() for inventory in state["player_inventories"]
        ]
        self.last_interacted = state["last_interacted"].copy()
        self.timestep = state["timestep"]

    def set_state(self, state: MbagStateDict) -> List[MbagObs]:
        self.set_state_no_obs(state)

        return [
            self._get_player_obs(player_index)
            for player_index in range(self.config["num_players"])
        ]


class PlayersOverlappingError(Exception):
    pass
