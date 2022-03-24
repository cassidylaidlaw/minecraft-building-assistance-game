from typing import List, Optional, TYPE_CHECKING, Sequence, Tuple, Type, cast
from typing_extensions import Literal, TypedDict
import numpy as np
from gym import spaces
import time
import logging
from operator import add

from .blocks import MinecraftBlocks
from .types import (
    BlockLocation,
    WorldLocation,
    MbagAction,
    MbagActionTuple,
    MbagInfoDict,
    MbagObs,
    WorldSize,
    FacingDirection,
    num_world_obs_channels,
)
from .goals.goal_generator import GoalGenerator
from .goals.simple import RandomGoalGenerator

if TYPE_CHECKING:
    from .malmo import MalmoObservationDict

logger = logging.getLogger(__name__)


class MalmoConfigDict(TypedDict):
    use_malmo: bool
    """
    Whether to connect to a real Minecraft instance with Project Malmo.
    """

    use_spectator: bool
    """
    Adds in a spectator player to observe the game from a 3rd person point of view.
    """

    video_dir: Optional[str]
    """
    Optional directory to record video from the game into.
    """


class RewardsConfigDict(TypedDict):
    noop: int
    """
    The reward for doing any action which does nothing. This is usually either zero,
    or negative to discourage noops.
    """

    place_wrong: float
    """
    The reward for placing a block which is not correct, but in a place where a block
    should go. The negative of this is also given for breaking a block which is not
    correct.
    """


class AbilitiesConfigDict(TypedDict):
    teleportation: bool
    """
    Whether the agent can teleport or must move block by block
    """

    flying: bool
    """
    Whether the agent can fly or if the agent must be standing on a block at all times
    """


class MbagConfigDict(TypedDict, total=False):
    num_players: int
    horizon: int
    world_size: WorldSize

    goal_generator: Tuple[Type[GoalGenerator], dict]

    goal_visibility: List[bool]
    """
    List with one boolean for each player, indicating if the player can observe the
    goal.
    """

    malmo: MalmoConfigDict
    """Configuration options for connecting to Minecraft with Project Malmo."""

    rewards: RewardsConfigDict
    """Configuration options for environment reward."""

    abilities: AbilitiesConfigDict
    """Configuration for limits placed on the agent (flying, teleportation, inventory, etc...) """


DEFAULT_CONFIG: MbagConfigDict = {
    "num_players": 1,
    "horizon": 50,
    "world_size": (5, 5, 5),
    "goal_generator": (RandomGoalGenerator, {}),
    "goal_visibility": [True, False],
    "malmo": {
        "use_malmo": False,
        "use_spectator": False,
        "video_dir": None,
    },
    "rewards": {
        "noop": 0,
        "place_wrong": 0,
    },
    "abilities": {"teleportation": True, "flying": True},
}


class MbagEnv(object):

    current_blocks: MinecraftBlocks
    goal_blocks: MinecraftBlocks
    player_locations: List[WorldLocation]
    player_directions: List[FacingDirection]
    timestep: int

    def __init__(self, config: MbagConfigDict):
        self.config = DEFAULT_CONFIG
        self.config.update(config)
        if isinstance(self.config["world_size"], list):
            self.config["world_size"] = tuple(self.config["world_size"])

        self.world_obs_shape = (num_world_obs_channels,) + self.config["world_size"]
        self.observation_space = spaces.Tuple(
            (spaces.Box(0, 255, self.world_obs_shape, dtype=np.uint8),)
        )
        self.player_locations = [(0, 2, 0) for _ in range(self.config["num_players"])]
        self.player_directions = [(0, 0) for _ in range(self.config["num_players"])]
        # Actions consist of an (action_type, block_location, block_id) tuple.
        # Not all action types use block_location and block_id. See MbagAction for
        # more details.
        self.action_space = spaces.Tuple(
            (
                spaces.Discrete(MbagAction.NUM_ACTION_TYPES),
                spaces.Discrete(np.prod(self.config["world_size"])),
                spaces.Discrete(MinecraftBlocks.NUM_BLOCKS),
            )
        )

        GoalGeneratorClass, goal_generator_config = self.config["goal_generator"]
        self.goal_generator = GoalGeneratorClass(goal_generator_config)

        if self.config["malmo"]["use_malmo"]:
            from .malmo import MalmoClient

            self.malmo_client = MalmoClient()

    def reset(self) -> List[MbagObs]:
        self.timestep = 0

        self.current_blocks = MinecraftBlocks(self.config["world_size"])
        self.current_blocks.blocks[:, 0, :] = MinecraftBlocks.BEDROCK
        self.current_blocks.blocks[:, 1, :] = MinecraftBlocks.NAME2ID["dirt"]

        self.goal_blocks = self._generate_goal()

        self.player_locations = [
            (
                (i % self.config["world_size"][0]) + 0.5,
                2,
                int(i / self.config["world_size"][0]) + 0.5,
            )
            for i in range(self.config["num_players"])
        ]

        if self.config["malmo"]["use_malmo"]:
            self.malmo_client.start_mission(self.config, self.goal_blocks)
            if self.config["malmo"]["use_spectator"]:
                logger.warn("use_spectator is not yet implemented")
            if self.config["malmo"]["video_dir"] is not None:
                logger.warn("video_dir is not yet implemented")
            time.sleep(1)  # Wait a second for the environment to load.

            # Make all players fly.
            for player_index in range(self.config["num_players"]):
                for _ in range(2):
                    self.malmo_client.send_command(player_index, "jump 1")
                    time.sleep(0.1)
                    self.malmo_client.send_command(player_index, "jump 0")
                    time.sleep(0.1)
                self.malmo_client.send_command(
                    player_index,
                    "tp " + " ".join(map(str, self.player_locations[player_index])),
                )

        return [
            self._get_player_obs(player_index)
            for player_index in range(self.config["num_players"])
        ]

    def step(
        self, action_tuples: List[MbagActionTuple]
    ) -> Tuple[List[MbagObs], List[float], List[bool], List[MbagInfoDict]]:
        assert (
            len(action_tuples) == self.config["num_players"]
        ), "Wrong number of actions."

        reward: float = 0
        infos: List[MbagInfoDict] = []

        for player_index, player_action_tuple in enumerate(action_tuples):
            player_reward, player_info = self._step_player(
                player_index, player_action_tuple
            )
            reward += player_reward
            infos.append(player_info)

        self.timestep += 1

        if self.config["malmo"]["use_malmo"]:
            time.sleep(self.malmo_client.ACTION_DELAY)
            self._update_state_from_malmo()

        obs = [
            self._get_player_obs(player_index)
            for player_index in range(self.config["num_players"])
        ]
        rewards = [reward] * self.config["num_players"]
        dones = [self._done()] * self.config["num_players"]

        if dones[0] and self.config["malmo"]["use_malmo"]:
            for player_index in range(self.config["num_players"]):
                self.malmo_client.send_command(player_index, "quit")

        return obs, rewards, dones, infos

    def _generate_goal(self) -> MinecraftBlocks:
        # Generate a goal with buffer of at least 1 on the sides and 2 on the bottom.
        world_size = self.config["world_size"]
        small_goal = self.goal_generator.generate_goal(
            (world_size[0] - 2, world_size[1] - 1, world_size[2] - 2)
        )

        goal = self.current_blocks.copy()
        goal.blocks[1:-1, 1:, 1:-1] = small_goal.blocks
        goal.block_states[1:-1, 1:, 1:-1] = small_goal.block_states
        return goal

    def _step_player(
        self, player_index: int, action_tuple: MbagActionTuple
    ) -> Tuple[float, MbagInfoDict]:
        action = MbagAction(action_tuple, self.config["world_size"])
        reward: float = 0

        noop: bool = True

        # print(action_tuple)

        if action.action_type == MbagAction.NOOP:
            pass
        elif action.action_type in [MbagAction.PLACE_BLOCK, MbagAction.BREAK_BLOCK]:
            prev_block = self.current_blocks[action.block_location]
            goal_block = self.goal_blocks[action.block_location]

            # Try to place or break a block
            if self._collides_with_players(action.block_location, player_index):
                place_break_result = None
            elif self.config["abilities"]["teleportation"]:
                place_break_result = self.current_blocks.try_break_place(
                    cast(Literal[1, 2], action.action_type),
                    action.block_location,
                    action.block_id,
                )
            else:
                place_break_result = self.current_blocks.try_break_place(
                    cast(Literal[1, 2], action.action_type),
                    action.block_location,
                    action.block_id,
                    player_location=self.player_locations[player_index],
                )

            # print("Tried placing ", action.block_location)
            if place_break_result is not None:
                # print("Place block successful")
                noop = False
                if self.config["abilities"]["teleportation"]:
                    self.player_locations[player_index] = (
                        place_break_result[0][0],
                        place_break_result[0][1],
                        place_break_result[0][2],
                    )

            if place_break_result is not None and self.config["malmo"]["use_malmo"]:
                player_location, click_location = place_break_result

                if self.config["abilities"]["teleportation"]:
                    self.malmo_client.send_command(
                        player_index,
                        "tp " + " ".join(map(str, player_location)),
                    )

                viewpoint = np.array(player_location)
                viewpoint[1] += 1.6
                delta = np.array(click_location) - viewpoint
                delta /= np.sqrt((delta**2).sum())
                yaw = np.rad2deg(np.arctan2(-delta[0], delta[2]))
                pitch = np.rad2deg(-np.arcsin(delta[1]))
                self.malmo_client.send_command(player_index, f"setYaw {yaw}")
                self.malmo_client.send_command(player_index, f"setPitch {pitch}")
                self.player_directions[player_index] = (yaw, pitch)

                if action.action_type == MbagAction.PLACE_BLOCK:
                    self.malmo_client.send_command(
                        player_index, f"swapInventoryItems 0 {action.block_id}"
                    )
                    time.sleep(0.1)  # Give time to swap item to hand and teleport.
                    self.malmo_client.send_command(player_index, "use 1")
                    time.sleep(0.1)  # Give time to place block.
                    self.malmo_client.send_command(
                        player_index, f"swapInventoryItems 0 {action.block_id}"
                    )
                else:
                    time.sleep(0.1)  # Give time to teleport.
                    self.malmo_client.send_command(player_index, "attack 1")

            # Calculate reward based on progress towards goal.
            new_block = self.current_blocks[action.block_location]
            prev_goal_similarity = self._get_goal_similarity(
                prev_block, goal_block, partial_credit=True
            )
            new_goal_similarity = self._get_goal_similarity(
                new_block, goal_block, partial_credit=True
            )
            reward = new_goal_similarity - prev_goal_similarity
        elif action.action_type in [
            MbagAction.MOVE_POS_X,
            MbagAction.MOVE_NEG_X,
            MbagAction.MOVE_POS_Y,
            MbagAction.MOVE_NEG_Y,
            MbagAction.MOVE_POS_Z,
            MbagAction.MOVE_NEG_Z,
        ]:
            if not self.config["abilities"]["teleportation"]:
                noop = False

                player_location_list = list(self.player_locations[player_index])

                action_mask = {
                    MbagAction.MOVE_POS_X: [[1, 0, 0], "moveeast 1"],
                    MbagAction.MOVE_NEG_X: [[-1, 0, 0], "movewest 1"],
                    MbagAction.MOVE_POS_Y: [[0, 1, 0], "tp"],
                    MbagAction.MOVE_NEG_Y: [[0, -1, 0], "tp"],
                    MbagAction.MOVE_POS_Z: [[0, 0, 1], "movesouth 1"],
                    MbagAction.MOVE_NEG_Z: [[0, 0, -1], "movenorth 1"],
                }
                new_player_location = list(
                    map(
                        add,
                        action_mask[action.action_type][0],
                        player_location_list,
                    )
                )

                print("Old", player_location_list)
                print("Proposed", new_player_location)
                if self._is_valid_player_space(
                    tuple(new_player_location), player_index
                ):
                    player_location_list = new_player_location

                    self.player_locations[player_index] = (
                        player_location_list[0],
                        player_location_list[1],
                        player_location_list[2],
                    )

                    # print(str(action_mask[action.action_type][1]))
                    if self.config["malmo"]["use_malmo"]:
                        if action_mask[action.action_type][1] != "tp":
                            self.malmo_client.send_command(
                                player_index, str(action_mask[action.action_type][1])
                            )
                        else:
                            self.malmo_client.send_command(
                                player_index,
                                "tp " + " ".join(map(str, tuple(player_location_list))),
                            )
                else:
                    noop = True

                print("New", player_location_list)
        if noop:
            reward += self.config["rewards"]["noop"]

        info: MbagInfoDict = {
            "goal_similarity": self._get_goal_similarity(
                self.current_blocks[:],
                self.goal_blocks[:],
            ).sum(),
        }

        return reward, info

    def _is_valid_player_space(self, nearest_block, player_index: int):
        proposed_block = [
            int(np.floor(nearest_block[0])),
            nearest_block[1],
            int(np.floor(nearest_block[2])),
        ]
        # print(proposed_block)
        for i in range(3):
            if (
                proposed_block[i] < 0
                or proposed_block[i] >= self.config["world_size"][i]
            ):
                return False

        if (
            not self.current_blocks.blocks[
                proposed_block[0], proposed_block[1], proposed_block[2]
            ]
            == MinecraftBlocks.AIR
        ):
            return False

        if proposed_block[1] < self.config["world_size"][1] - 1:
            if (
                not self.current_blocks.blocks[
                    proposed_block[0], proposed_block[1] + 1, proposed_block[2]
                ]
                == MinecraftBlocks.AIR
            ):
                return False

        return not self._collides_with_players(proposed_block, player_index)

    def _collides_with_players(self, proposed_block, player_id: int):
        for i in range(len(self.player_locations)):

            if i == player_id:
                continue

            player = self.player_locations[i]
            if (
                proposed_block[0] == player[0] - 0.5
                and proposed_block[2] == player[2] - 0.5
                and (proposed_block[1] in (player[1] - 1, player[1], player[1] + 1))
            ):
                return True

        return False

    def _get_goal_similarity(
        self,
        current_block: Tuple[np.ndarray, np.ndarray],
        goal_block: Tuple[np.ndarray, np.ndarray],
        partial_credit: bool = False,
    ):
        """
        Get the similarity between this block and the goal block, used to calculate
        the reward. The reward is the different between this value before and after the
        player's action.
        """

        current_block_id, current_block_state = current_block
        goal_block_id, goal_block_state = goal_block

        similarity = np.zeros_like(current_block_id, dtype=float)
        if partial_credit:
            # Give partial credit for placing the wrong block type.
            similarity[
                (goal_block_id != MinecraftBlocks.AIR)
                & (current_block_id != MinecraftBlocks.AIR)
            ] = self.config["rewards"]["place_wrong"]
        similarity[goal_block_id == current_block_id] = 1
        return similarity

    def _get_player_obs(self, player_index: int) -> MbagObs:
        world_obs = np.zeros(self.world_obs_shape, np.uint8)
        world_obs[0] = self.current_blocks.blocks
        world_obs[1] = self.current_blocks.block_states

        if self.config["goal_visibility"][player_index]:
            world_obs[2] = self.goal_blocks.blocks
            world_obs[3] = self.goal_blocks.block_states

        player_position = self.player_locations[player_index]

        return (world_obs, player_position)

    def _update_state_from_malmo(self):
        malmo_state = self.malmo_client.get_observation(0)
        if malmo_state is None:
            return

        malmo_blocks = MinecraftBlocks.from_malmo_grid(
            self.config["world_size"], malmo_state["world"]
        )
        malmo_goal = MinecraftBlocks.from_malmo_grid(
            self.config["world_size"], malmo_state["goal"]
        )

        location: BlockLocation
        for location in cast(
            Sequence[BlockLocation],
            map(tuple, np.argwhere(malmo_blocks.blocks != self.current_blocks.blocks)),
        ):
            logger.warning(
                f"block discrepancy at {location}: "
                "expected "
                f"{MinecraftBlocks.ID2NAME[self.current_blocks.blocks[location]]} "
                f"but received "
                f"{MinecraftBlocks.ID2NAME[malmo_blocks.blocks[location]]} "
                "from Malmo"
            )
        for location in cast(
            Sequence[BlockLocation],
            map(tuple, np.argwhere(malmo_goal.blocks != self.goal_blocks.blocks)),
        ):
            logger.error(
                f"goal discrepancy at {location}: "
                "expected "
                f"{MinecraftBlocks.ID2NAME[self.goal_blocks.blocks[location]]} "
                f"but received {MinecraftBlocks.ID2NAME[malmo_goal.blocks[location]]} "
                "from Malmo"
            )

        for player_index in range(self.config["num_players"]):
            malmo_player_state: Optional[MalmoObservationDict]
            if player_index == 0:
                malmo_player_state = malmo_state
            else:
                malmo_player_state = self.malmo_client.get_observation(player_index)
            if malmo_player_state is None:
                continue

            # Make sure inventory is organized as expected.
            inventory_block_ids = [
                MinecraftBlocks.NAME2ID[
                    malmo_player_state[f"InventorySlot_{slot}_item"]  # type: ignore
                ]
                for slot in range(36)
            ]
            for block_id in MinecraftBlocks.PLACEABLE_BLOCK_IDS:
                if inventory_block_ids[block_id] != block_id:
                    logger.warning(
                        f"inventory discrepancy at slot {block_id}: "
                        f"expected {MinecraftBlocks.ID2NAME[block_id]} "
                        "but received "
                        f"{MinecraftBlocks.ID2NAME[inventory_block_ids[block_id]]} "
                        "from Malmo"
                    )
                    swap_slot = inventory_block_ids.index(block_id)
                    self.malmo_client.send_command(
                        player_index, f"swapInventoryItems {block_id} {swap_slot}"
                    )
                    time.sleep(0.1)

            if not self.config["abilities"]["teleportation"]:
                # Make sure position is as expected.
                malmo_location = (
                    malmo_player_state["XPos"],
                    malmo_player_state["YPos"],
                    malmo_player_state["ZPos"],
                )
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

        self.current_blocks.blocks = malmo_blocks.blocks

    def _done(self) -> bool:
        return (
            self.timestep >= self.config["horizon"]
            or self.current_blocks == self.goal_blocks
        )
