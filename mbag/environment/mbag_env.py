import json
from math import floor
from typing import (
    Dict,
    List,
    Optional,
    TYPE_CHECKING,
    Tuple,
    Type,
    Union,
    Sequence,
    cast,
)
from typing_extensions import Literal, TypedDict
import numpy as np
from gym import spaces
import time
import copy
import random
import logging


from .blocks import MinecraftBlocks
from .types import (
    BlockLocation,
    MbagActionType,
    MbagInventory,
    MbagWorldObsArray,
    WorldLocation,
    MbagAction,
    MbagActionTuple,
    MbagInfoDict,
    MbagObs,
    WorldSize,
    FacingDirection,
    MbagInventoryObs,
    num_world_obs_channels,
    CURRENT_BLOCKS,
    CURRENT_BLOCK_STATES,
    GOAL_BLOCKS,
    GOAL_BLOCK_STATES,
    LAST_INTERACTED,
)
from .goals import (
    ALL_GOAL_GENERATORS,
    GoalGenerator,
    TransformedGoalGenerator,
    GoalGeneratorConfig,
)

if TYPE_CHECKING:
    from .malmo import MalmoObservationDict

logger = logging.getLogger(__name__)


class MalmoConfigDict(TypedDict, total=False):
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


class RewardsConfigDict(TypedDict, total=False):
    noop: float
    """
    The reward for doing any action which does nothing. This is usually either zero,
    or negative to discourage noops.
    """

    action: float
    """
    The reward for doing any action which is not a noop. This could be negative to
    introduce some cost for acting.
    """

    place_wrong: float
    """
    The reward for placing a block which is not correct, but in a place where a block
    should go. The negative of this is also given for breaking a block which is not
    correct.
    """

    own_reward_prop: float
    """
    A number from 0 to 1. At 0, it gives the normal reward function which takes into
    account all players actions. At 1, it gives only reward for actions that the
    specific player took.
    """

    own_reward_prop_horizon: Optional[int]
    """
    Decay own_reward_prop to 0 over this horizon. This requires calling
    set_global_timestep on the environment to update the global timestep.
    """

    get_resources: float
    """
    A number from 0 to 1. The reward for getting a resource by mining the palette.
    Not sure if strictly necessary.
    """


class AbilitiesConfigDict(TypedDict):
    teleportation: bool
    """
    Whether the agent can teleport or must move block by block.
    """

    flying: bool
    """
    Whether the agent can fly or if the agent must be standing on a block at all times.
    Not implemented yet!
    """

    inf_blocks: bool
    """
    True - agent has infinite blocks to build with
    False - agent must manage resources and inventory
    """


class MbagPlayerConfigDict(TypedDict, total=False):
    player_name: Optional[str]
    """A player name that will be displayed in Minecraft if connected via Malmo."""

    goal_visible: bool
    """Whether the player can observe the goal."""

    is_human: bool
    """
    Whether this player is a human interacting via Malmo. Setting this to True requires
    Malmo to be configured.
    """

    timestep_skip: int
    """
    How often the player can interact with the environment, i.e. 1 means every
    timestep, 5 means only every 5th timestep.
    """

    rewards: RewardsConfigDict
    """
    Optional per-player reward configuration. Any unpopulated keys are overridden by
    values from the overall rewards config dict.
    """


class MbagConfigDict(TypedDict, total=False):
    num_players: int
    horizon: int
    world_size: WorldSize

    goal_generator: Union[Type[GoalGenerator], str]
    goal_generator_config: GoalGeneratorConfig

    players: List[MbagPlayerConfigDict]
    """List of player configuration dictionaries."""

    malmo: MalmoConfigDict
    """Configuration options for connecting to Minecraft with Project Malmo."""

    rewards: RewardsConfigDict
    """
    Configuration options for environment reward. To configure on a per-player basis,
    use rewards key in player configuration dictionary.
    """

    abilities: AbilitiesConfigDict
    """
    Configuration for limits placed on the players (e.g., can they teleport, do they
    have to gather resources, etc.).
    """


DEFAULT_PLAYER_CONFIG: MbagPlayerConfigDict = {
    "player_name": None,
    "goal_visible": True,
    "is_human": False,
    "timestep_skip": 1,
    "rewards": {},
}


DEFAULT_CONFIG: MbagConfigDict = {
    "num_players": 1,
    "horizon": 50,
    "world_size": (5, 5, 5),
    "goal_generator": TransformedGoalGenerator,
    "goal_generator_config": {
        "goal_generator": "random",
        "goal_generator_config": {},
        "transforms": [
            {"transform": "add_grass"},
        ],
    },
    "players": [{}],
    "malmo": {
        "use_malmo": False,
        "use_spectator": False,
        "video_dir": None,
    },
    "rewards": {
        "noop": 0.0,
        "action": 0.0,
        "place_wrong": 0.0,
        "own_reward_prop": 0.0,
        "own_reward_prop_horizon": None,
        "get_resources": 0,
    },
    "abilities": {"teleportation": True, "flying": True, "inf_blocks": True},
}

NO_ONE = 0
CURRENT_PLAYER = 1
OTHER_PLAYER = 2
NO_INTERACTION = -1


class MbagStateDict(TypedDict):
    current_blocks: MinecraftBlocks
    goal_blocks: MinecraftBlocks
    player_locations: List[WorldLocation]
    player_directions: List[FacingDirection]
    player_inventories: List[MbagInventory]
    player_block_looking: List[WorldLocation]
    player_bock_breaking: List[WorldLocation]
    last_interacted: np.ndarray
    timestep: int


class MbagEnv(object):
    config: MbagConfigDict
    current_blocks: MinecraftBlocks
    goal_blocks: MinecraftBlocks
    player_locations: List[WorldLocation]
    player_directions: List[FacingDirection]
    player_inventories: List[MbagInventory]
    palette_x: int
    last_interacted: np.ndarray
    timestep: int
    global_timestep: int

    BLOCKS_TO_GIVE = 5
    """The number of blocks given in a GIVE_BLOCK action."""

    INVENTORY_NUM_SLOTS = 36
    """The number of stacks of items a player can carry."""

    STACK_SIZE = 64
    """The maximum number of blocks a player can carry in a stack."""

    def __init__(self, config: MbagConfigDict):
        self.config = self.get_config(copy.deepcopy(config))

        self.world_obs_shape = (num_world_obs_channels,) + self.config["world_size"]
        self.observation_space = spaces.Tuple(
            (
                spaces.Box(0, 255, self.world_obs_shape, dtype=np.uint8),
                spaces.Box(
                    0,
                    self.INVENTORY_NUM_SLOTS * self.STACK_SIZE,
                    (MinecraftBlocks.NUM_BLOCKS,),
                    dtype=int,
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
                spaces.Discrete(np.prod(self.config["world_size"])),
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
            from .malmo import MalmoClient

            self.malmo_client = MalmoClient()

        self.global_timestep = 0

    def get_config(self, partial_config: MbagConfigDict) -> MbagConfigDict:
        """Get a fully populated config dict by adding defaults where necessary."""

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
                raise ValueError(
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
        self.global_timestep = global_timestep

    def reset(self) -> List[MbagObs]:
        """Reset Minecraft environment and return player observations for each player."""
        self.timestep = 0

        self.current_blocks = MinecraftBlocks(self.config["world_size"])
        self.current_blocks.blocks[:, 0, :] = MinecraftBlocks.BEDROCK
        self.current_blocks.blocks[:, 1, :] = MinecraftBlocks.NAME2ID["dirt"]

        self.last_interacted = np.zeros(self.config["world_size"])
        self.last_interacted[:] = NO_INTERACTION

        self.goal_blocks = self._generate_goal()

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
            np.zeros((self.INVENTORY_NUM_SLOTS, 2), dtype=np.int32)
            for _ in range(self.config["num_players"])
        ]

        self.player_block_looking = [None for _ in range(self.config["num_players"])]
        self.player_block_breaking = [None for _ in range(self.config["num_players"])]

        if self.config["malmo"]["use_malmo"]:
            self.malmo_client.start_mission(
                self.config, self.current_blocks, self.goal_blocks
            )
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

        if not self.config["abilities"]["inf_blocks"]:
            self._copy_palette_from_goal()

        if self.config["malmo"]["use_malmo"]:
            # Let everything load in Malmo.
            time.sleep(self.malmo_client.ACTION_DELAY)

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
        own_rewards: List[float] = []
        infos: List[MbagInfoDict] = []

        for player_index, player_action_tuple in enumerate(action_tuples):
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
            own_rewards.append(player_reward)
            infos.append(player_info)

        self.timestep += 1

        if self.config["malmo"]["use_malmo"]:
            time.sleep(self.malmo_client.ACTION_DELAY)
            infos = self._update_state_from_malmo(infos)
            # print(infos)
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
            # Wait for a second for the final block to place and then end mission.
            time.sleep(1)
            self.malmo_client.end_mission()

        return obs, rewards, dones, infos

    def _generate_goal(self) -> MinecraftBlocks:
        # Generate a goal with buffer of at least 1 on the sides and bottom.
        world_size = self.config["world_size"]

        goal_size = (world_size[0] - 2, world_size[1] - 1, world_size[2] - 2)
        if self.config["abilities"]["inf_blocks"]:
            self.palette_x = -1
        else:
            goal_size = (world_size[0] - 3, world_size[1] - 1, world_size[2] - 2)
            self.palette_x = world_size[0] - 1

        small_goal = self.goal_generator.generate_goal(goal_size)

        goal = self.current_blocks.copy()

        if self.config["abilities"]["inf_blocks"]:
            goal.blocks[1:-1, 1:, 1:-1] = small_goal.blocks
            goal.block_states[1:-1, 1:, 1:-1] = small_goal.block_states
        else:
            goal.blocks[1:-2, 1:, 1:-1] = small_goal.blocks
            goal.block_states[1:-2, 1:, 1:-1] = small_goal.block_states

            for index, block in enumerate(MinecraftBlocks.PLACEABLE_BLOCK_IDS):
                if index >= world_size[2]:
                    break
                goal.blocks[self.palette_x, 2, index] = block
                goal.block_states[self.palette_x, 2, index] = 0

        # logger.debug(goal.blocks)
        return goal

    def _copy_palette_from_goal(self):
        # Copy over the palette from the goal generator.
        self.current_blocks.blocks[self.palette_x] = self.goal_blocks.blocks[
            self.palette_x
        ]
        self.current_blocks.block_states[
            self.palette_x
        ] = self.goal_blocks.block_states[self.palette_x]

        # Sync with Malmo.
        if self.config["malmo"]["use_malmo"]:
            width, height, depth = self.config["world_size"]
            goal_palette_x = self.palette_x + width + 1
            self.malmo_client.send_command(
                0,
                f"chat /clone {goal_palette_x} 0 0 "
                f"{goal_palette_x} {height - 1} {depth - 1} "
                f"{self.palette_x} 0 0",
            )

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
            prev_inventory_obs = self._get_inventory_obs(player_index)
            noop = not self._handle_place_break(player_index, action)

            # Calculate reward based on progress towards goal.
            if (
                action.action_type == MbagAction.BREAK_BLOCK
                and action.block_location[0] == self.palette_x
            ):
                # TODO: shouldn't we check if the user actually broke the block?
                # might be worth adding a test to make sure the reward only comes
                # through if they did
                new_inventory_obs = self._get_inventory_obs(player_index)
                goal_independent_reward += (
                    np.count_nonzero(new_inventory_obs)
                    - np.count_nonzero(prev_inventory_obs)
                ) * self._get_reward_config_for_player(player_index)["get_resources"]
            else:
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
        elif (
            action.action_type in MbagAction.MOVE_ACTION_TYPES
            and not self.config["abilities"]["teleportation"]
        ):
            noop = not self._handle_move(player_index, action.action_type)
        elif (
            action.action_type == MbagAction.GIVE_BLOCK
            and not self.config["abilities"]["inf_blocks"]
        ):
            noop = 0 == self._handle_give_block(
                player_index, action.block_id, action.block_location
            )

        if noop:
            goal_independent_reward += self._get_reward_config_for_player(player_index)[
                "noop"
            ]
        else:
            goal_independent_reward += self._get_reward_config_for_player(player_index)[
                "action"
            ]

        reward = goal_dependent_reward + goal_independent_reward

        if not self.config["abilities"]["inf_blocks"]:
            self._copy_palette_from_goal()

        info: MbagInfoDict = {
            "goal_similarity": self._get_goal_similarity(
                self.current_blocks[:],
                self.goal_blocks[:],
            ).sum(),
            "goal_dependent_reward": goal_dependent_reward,
            "goal_independent_reward": goal_independent_reward,
            "own_reward": reward,
            "own_reward_prop": self._get_own_reward_prop(player_index),
            "action": action if not noop else MbagAction.noop_action(),
            "action_correct": action_correct and not noop,
            "human_actions": [],
        }

        return reward, info

    def _handle_place_break(self, player_index: int, action: MbagAction) -> bool:
        prev_block = self.current_blocks[action.block_location]
        inventory_slot = 0

        if (
            not self.config["abilities"]["inf_blocks"]
            and action.action_type == MbagAction.PLACE_BLOCK
        ):
            inventory_slot = self._try_take_player_block(
                action.block_id, player_index, False
            )

        if inventory_slot < 0:
            return False

        # Try to place or break a block
        if self.config["abilities"]["teleportation"]:
            place_break_result = self.current_blocks.try_break_place(
                cast(Literal[1, 2], action.action_type),
                action.block_location,
                action.block_id,
            )
        else:
            if self._collides_with_players(action.block_location, player_index):
                place_break_result = None
            else:
                place_break_result = self.current_blocks.try_break_place(
                    cast(Literal[1, 2], action.action_type),
                    action.block_location,
                    action.block_id,
                    player_location=self.player_locations[player_index],
                    other_player_locations=self.player_locations[:player_index]
                    + self.player_locations[player_index + 1 :],
                )

        if place_break_result is None:
            return False

        if self.config["abilities"]["teleportation"]:
            self.player_locations[player_index] = (
                place_break_result[0][0],
                place_break_result[0][1],
                place_break_result[0][2],
            )

        self.last_interacted[action.block_location] = player_index

        if (
            self.config["malmo"]["use_malmo"]
            and not self.config["players"][player_index]["is_human"]
        ):
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
                if self.config["abilities"]["inf_blocks"]:
                    self.malmo_client.send_command(
                        player_index,
                        f"swapInventoryItems 0 {action.block_id}",
                    )
                    hotbar_slot = 0
                else:
                    player_inventory = self.player_inventories[player_index]
                    if inventory_slot < 9:
                        hotbar_slot = inventory_slot
                    else:
                        # Block is not in hotbar, need to swap it in.
                        hotbar_slot = random.randrange(9)
                        self.malmo_client.send_command(
                            player_index,
                            f"swapInventoryItems {hotbar_slot} {inventory_slot}",
                        )
                        (
                            player_inventory[hotbar_slot],
                            player_inventory[inventory_slot],
                        ) = (
                            player_inventory[inventory_slot].copy(),
                            player_inventory[hotbar_slot].copy(),
                        )

                self.malmo_client.send_command(
                    player_index, f"hotbar.{hotbar_slot + 1} 1"
                )
                self.malmo_client.send_command(
                    player_index, f"hotbar.{hotbar_slot + 1} 0"
                )
                time.sleep(0.1)  # Give time to swap item to hand and teleport.
                self.malmo_client.send_command(player_index, "use 1")
                time.sleep(0.1)  # Give time to place block.
                if self.config["abilities"]["inf_blocks"]:
                    self.malmo_client.send_command(
                        player_index,
                        f"swapInventoryItems 0 {action.block_id}",
                    )
            else:
                time.sleep(0.1)  # Give time to teleport.
                self.malmo_client.send_command(player_index, "attack 1")

        if not self.config["abilities"]["inf_blocks"]:
            if action.action_type == MbagAction.BREAK_BLOCK:
                # Give the block to the player. It looks like Malmo
                # automatically gives broken blocks to the player
                # so we pass give_in_malmo=False.

                # Not necessarily. In Minecraft broken block entities have random momentum so
                # sometimes the block will not be given to the player.
                self._try_give_player_block(
                    int(prev_block[0]), player_index, give_in_malmo=False
                )
            else:
                # Take block from player
                assert (
                    self._try_take_player_block(action.block_id, player_index, True)
                    >= 0
                )
        return True

    def _handle_move(self, player_index: int, action_type: MbagActionType) -> bool:
        """
        Handle a move action.
        Returns whether the action was successful or not
        """

        player_location = self.player_locations[player_index]

        action_mask: Dict[MbagActionType, Tuple[WorldLocation, str]] = {
            MbagAction.MOVE_POS_X: ((1, 0, 0), "moveeast 1"),
            MbagAction.MOVE_NEG_X: ((-1, 0, 0), "movewest 1"),
            MbagAction.MOVE_POS_Y: ((0, 1, 0), "tp"),
            MbagAction.MOVE_NEG_Y: ((0, -1, 0), "tp"),
            MbagAction.MOVE_POS_Z: ((0, 0, 1), "movesouth 1"),
            MbagAction.MOVE_NEG_Z: ((0, 0, -1), "movenorth 1"),
        }
        dx, dy, dz = action_mask[action_type][0]
        new_player_location: WorldLocation = (
            player_location[0] + dx,
            player_location[1] + dy,
            player_location[2] + dz,
        )

        if not self._is_valid_player_space(new_player_location, player_index):
            return False

        player_location = new_player_location
        self.player_locations[player_index] = player_location

        if (
            self.config["malmo"]["use_malmo"]
            and not self.config["players"][player_index]["is_human"]
        ):
            if action_mask[action_type][1] != "tp":
                self.malmo_client.send_command(
                    player_index, action_mask[action_type][1]
                )
            else:
                self.malmo_client.send_command(
                    player_index,
                    "tp " + " ".join(map(str, player_location)),
                )

        return True

    def _handle_give_block(
        self, giver_player_index: int, block_id: int, receiver_location: WorldLocation
    ) -> int:
        """
        Handles giving blocks to a player. Returns how many blocks were given.
        """

        receiver_player_location = (
            receiver_location[0] + 0.5,
            receiver_location[1],
            receiver_location[2] + 0.5,
        )
        logger.debug(self.player_locations)
        logger.debug(receiver_player_location)

        # Check if player can reach the location specified (has to be within one block
        # in all directions).
        gx, gy, gz = self.player_locations[giver_player_index]
        rx, ry, rz = receiver_player_location
        if not self.config["abilities"]["teleportation"] and (
            abs(gx - rx) > 1 or abs(gy - ry) > 1 or abs(gz - rz) > 1
        ):
            return 0

        logger.debug("Finding player index")
        # Find player index at the location specified
        try:
            receiver_player_index = self.player_locations.index(
                receiver_player_location
            )
        except ValueError:
            return 0

        logger.debug(self.player_inventories[giver_player_index])
        # Give the blocks
        for block_index in range(self.BLOCKS_TO_GIVE):
            success = False
            inventory_taken = self._try_take_player_block(
                block_id, giver_player_index, True
            )
            if inventory_taken >= 0:
                success = self._try_give_player_block(
                    block_id, receiver_player_index, True
                )

            if not success:
                return block_index
        logger.debug("Successfully gave block to player")
        logger.debug(self.player_inventories[giver_player_index])

        return self.BLOCKS_TO_GIVE

    def _try_give_player_block(
        self, block_id: int, player_index: int, give_in_malmo: bool = True
    ) -> bool:
        """
        Attempts to give player_index one block of block_id
        Returns a boolean whether the give was successful
        """
        selected_slot = -1

        # Find an appropriate slot
        player_inventory = self.player_inventories[player_index]
        matching_inventory_slots = np.where(
            (player_inventory[:, 0] == block_id)
            & (player_inventory[:, 1] < self.STACK_SIZE)
            & (player_inventory[:, 1] > 0)
        )[0]

        if matching_inventory_slots.size > 0:
            selected_slot = matching_inventory_slots[0]
        else:
            empty_inventory_slots = np.where(player_inventory[:, 1] == 0)[0]
            if empty_inventory_slots.size > 0:
                selected_slot = empty_inventory_slots[0]

        if selected_slot == -1:
            return False

        # Give the inventory item
        player_inventory[selected_slot, 0] = block_id
        player_inventory[selected_slot, 1] += 1

        if (
            self.config["malmo"]["use_malmo"]
            and give_in_malmo
            and not self.config["players"][player_index]["is_human"]
        ):
            player_name = self.malmo_client.get_player_name(player_index, self.config)
            block_name = MinecraftBlocks.ID2NAME[block_id]
            self.malmo_client.send_command(
                player_index, f"chat /give {player_name} {block_name}"
            )

        return True

    def _get_inventory_obs(self, player_index: int) -> MbagInventoryObs:
        """
        Gets the array representation of the given player's inventory.
        """

        player_inventory = self.player_inventories[player_index]
        inventory_obs: MbagInventoryObs = np.zeros(
            MinecraftBlocks.NUM_BLOCKS, dtype=int
        )  # 10 total blocks
        for i in range(player_inventory.shape[0]):
            inventory_obs[player_inventory[i][0]] += player_inventory[i][1]

        return inventory_obs

    def _try_take_player_block(
        self, block_id: int, player_index: int, take: bool
    ) -> int:
        """
        Attempts to take from player_index one block of block_id
        Returns the inventory slot that was decremented, or -1 on a failure.
        """
        selected_slot = -1

        # Find an appropriate slot
        player_inventory = self.player_inventories[player_index]

        matching_inventory_slots = np.where(
            (player_inventory[:, 0] == block_id) & (player_inventory[:, 1] > 0)
        )[0]
        if matching_inventory_slots.size > 0:
            selected_slot = matching_inventory_slots[0]
        else:
            return -1

        # Decrement the inventory item
        if take:
            player_inventory[selected_slot, 1] -= 1
            if player_inventory[selected_slot, 1] == 0:
                player_inventory[selected_slot, 0] = MinecraftBlocks.AIR

            if (
                self.config["malmo"]["use_malmo"]
                and not self.config["players"][player_index]["is_human"]
            ):
                player_name = self.malmo_client.get_player_name(
                    player_index, self.config
                )
                block_name = MinecraftBlocks.ID2NAME[block_id]

                self.malmo_client.send_command(
                    player_index, f"chat /clear {player_name} {block_name} 0 1"
                )

        return selected_slot

    def _is_valid_player_space(
        self, player_location: WorldLocation, player_index: int
    ) -> bool:
        """
        Check to see if the player is able to place a block from their current position.
        """

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

        return not self._collides_with_players(proposed_block, player_index)

    def _collides_with_players(self, proposed_block, player_id: int) -> bool:
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
            ] = self._get_reward_config_for_player(player_index)["place_wrong"]
        similarity[goal_block_id == current_block_id] = 1
        return similarity

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
            for other_player_index, other_player_location in enumerate(
                self.player_locations
            ):
                self._add_player_location_to_world_obs(
                    world_obs,
                    other_player_location,
                    player_marker_map[other_player_index],
                )

        for other_player_index in range(self.config["num_players"]):
            world_obs[LAST_INTERACTED][
                self.last_interacted == other_player_index
            ] = player_marker_map[other_player_index]

        return (
            world_obs,
            self._get_inventory_obs(player_index),
            np.array(self.timestep, dtype=np.int32),
        )

    def _add_player_location_to_world_obs(
        self, world_obs: MbagWorldObsArray, player_location: WorldLocation, marker: int
    ):
        x, y_feet, z = player_location
        x, y_feet, z = int(np.floor(x)), int(np.floor(y_feet)), int(np.floor(z))
        for y in (
            [y_feet, y_feet + 1]
            if y_feet + 1 < self.config["world_size"][1]
            else [y_feet]
        ):
            assert world_obs[4, x, y, z] == 0, "players are overlapping"
            world_obs[4, x, y, z] = marker

    def _get_reward_config_for_player(self, player_index: int) -> RewardsConfigDict:
        return self.config["players"][player_index]["rewards"]

    def _get_own_reward_prop(self, player_index: int) -> float:
        reward_config = self._get_reward_config_for_player(player_index)
        own_reward_prop = reward_config["own_reward_prop"]
        own_reward_prop_horizon = reward_config["own_reward_prop_horizon"]
        if own_reward_prop_horizon is not None:
            own_reward_prop *= max(
                1 - self.global_timestep / own_reward_prop_horizon, 0
            )
        return own_reward_prop

    def _get_player_reward(
        self, player_index: int, reward: float, own_reward: float
    ) -> float:
        own_reward_prop = self._get_own_reward_prop(player_index)
        return own_reward_prop * own_reward + (1 - own_reward_prop) * reward

    def _get_human_actions(
        self,
        malmo_player_state,
        historical_player_state,
        player_index,
        location_discrepancies,
    ):

        # Compares malmo state to real state and tries to extrapolate player actions

        actions_queue = []

        env_location = (
            round(self.player_locations[player_index][0]),
            round(self.player_locations[player_index][1]),
            round(self.player_locations[player_index][2]),
        )

        malmo_location = (
            round(malmo_player_state.get("XPos", env_location[0])),
            round(malmo_player_state.get("YPos", env_location[1])),
            round(malmo_player_state.get("ZPos", env_location[2])),
        )

        print("env location", env_location)
        print("malmo location", malmo_location)
        # print("observations")
        # print(malmo_player_state.get("events", []))
        # print("history")
        # print(historical_player_state)

        if env_location[0] != malmo_location[0]:  # floor
            diff = int(abs(env_location[0] - malmo_location[0]))
            for _ in range(diff):
                actions_queue.append(
                    (
                        MbagAction.MOVE_NEG_X
                        if env_location[0] > malmo_location[0]
                        else MbagAction.MOVE_POS_X,
                        0,
                        0,
                    )
                )
        if env_location[1] != malmo_location[1]:
            diff = int(abs(env_location[1] - malmo_location[1]))
            for _ in range(diff):
                actions_queue.append(
                    (
                        MbagAction.MOVE_NEG_Y
                        if env_location[1] > malmo_location[1]
                        else MbagAction.MOVE_POS_Y,
                        0,
                        0,
                    )
                )
        if env_location[2] != malmo_location[2]:
            diff = int(abs(env_location[2] - malmo_location[2]))
            for _ in range(diff):
                actions_queue.append(
                    (
                        MbagAction.MOVE_NEG_Z
                        if env_location[2] > malmo_location[2]
                        else MbagAction.MOVE_POS_Z,
                        0,
                        0,
                    )
                )

        # This is only for using

        for time, snapshot in reversed(historical_player_state):
            if (
                self.player_block_breaking[player_index]
                and self.player_block_looking[player_index] in location_discrepancies
            ):
                print("Breaking", self.player_block_looking[player_index])
                actions_queue.append(
                    (
                        MbagAction.BREAK_BLOCK,
                        int(
                            np.ravel_multi_index(
                                self.player_block_looking[player_index],
                                self.config["world_size"],
                            )
                        ),
                        0,
                    )
                )
                location_discrepancies.remove(self.player_block_looking[player_index])

            if "LineOfSight" in snapshot and snapshot["LineOfSight"]["inRange"]:
                block_location = (
                    int(snapshot["LineOfSight"]["x"]),
                    int(snapshot["LineOfSight"]["y"]),
                    int(snapshot["LineOfSight"]["z"]),
                )
                self.player_block_looking[player_index] = block_location

            for event in snapshot.get("events", []):
                if (
                    "command" not in event
                    or event["command"] != "use"
                    and event["command"] != "attack"
                ):
                    continue

                print(event)

                if event["command"] == "use":
                    actions_queue.append(
                        (
                            MbagAction.PLACE_BLOCK,
                            int(
                                np.ravel_multi_index(
                                    block_location,
                                    self.config["world_size"],
                                )
                            ),
                            self.current_blocks.blocks[block_location],
                        )
                    ),
                else:
                    if event["pressed"]:
                        self.player_block_breaking[
                            player_index
                        ] = self.player_block_looking[player_index]
                    else:
                        # axis_of_placement = 2
                        # if block_location[0] - snapshot.get("x", env_location[0]):
                        #     axis_of_placement = 0
                        # elif block_location[1] - snapshot.get("y", env_location[1]):
                        #     axis_of_placement = 1

                        # direction = (
                        #     malmo_location[axis_of_placement]
                        #     - block_location[axis_of_placement]
                        # )
                        # direction = int(abs(direction) / direction)
                        # block_location[axis_of_placement] += direction

                        self.player_block_breaking[player_index] = None

        print(actions_queue)
        return None

    def _update_state_from_malmo(self, infos):
        malmo_historical_state = self.malmo_client.get_observations(0)

        if len(malmo_historical_state) == 0:
            return infos

        _, malmo_state = malmo_historical_state[0]

        location_discrepancies = []
        if "world" in malmo_state and "goal" in malmo_state:
            malmo_blocks = MinecraftBlocks.from_malmo_grid(
                self.config["world_size"], malmo_state["world"]
            )
            malmo_goal = MinecraftBlocks.from_malmo_grid(
                self.config["world_size"], malmo_state["goal"]
            )

            location: BlockLocation
            for location in cast(
                Sequence[BlockLocation],
                map(
                    tuple,
                    np.argwhere(malmo_blocks.blocks != self.current_blocks.blocks),
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
                if MinecraftBlocks.ID2NAME[malmo_blocks.blocks[location]] == "air":
                    location_discrepancies.append(location)

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
            malmo_player_historical_state = []
            if player_index == 0:
                malmo_player_historical_state = malmo_historical_state
            else:
                malmo_player_historical_state = self.malmo_client.get_observations(
                    player_index
                )

            if len(malmo_player_historical_state) == 0:
                continue

            _, malmo_player_state = malmo_player_historical_state[0]

            # Record any actions
            # infos[player_index]["human_actions"] = malmo_player_state.get("events", [])
            # if infos[player_index]["human_actions"]:
            #     if logger.isEnabledFor(logging.DEBUG):
            #         logger.debug(
            #             "received human actions from Malmo: "
            #             + json.dumps(infos[player_index]["human_actions"], indent=4)
            #         )
            #     print(
            #         "received human actions from Malmo: "
            #         + json.dumps(infos[player_index]["human_actions"], indent=4)
            #     )

            if self.config["players"][player_index]["is_human"]:
                self._get_human_actions(
                    malmo_player_state,
                    malmo_player_historical_state,
                    player_index,
                    location_discrepancies[:],
                )

            malmo_inventory: MbagInventory = np.array(
                [
                    [
                        MinecraftBlocks.NAME2ID[
                            malmo_player_state[f"InventorySlot_{slot}_item"]  # type: ignore
                        ],
                        malmo_player_state[f"InventorySlot_{slot}_size"],  # type: ignore
                    ]
                    for slot in range(self.INVENTORY_NUM_SLOTS)
                    if f"InventorySlot_{slot}_size" in malmo_player_state
                ]
            )
            if self.config["abilities"]["inf_blocks"]:
                # Make sure inventory is organized as expected.
                for block_id in MinecraftBlocks.PLACEABLE_BLOCK_IDS:
                    if malmo_inventory[block_id, 0] != block_id:
                        logger.warning(
                            f"inventory discrepancy at slot {block_id}: "
                            f"expected {MinecraftBlocks.ID2NAME[block_id]} "
                            "but received "
                            f"{MinecraftBlocks.ID2NAME[malmo_inventory[block_id, 0]]} "
                            "from Malmo"
                        )
                        swap_slot = malmo_inventory[:, 0].tolist().index(block_id)
                        self.malmo_client.send_command(
                            player_index, f"swapInventoryItems {block_id} {swap_slot}"
                        )
                        time.sleep(0.1)
            else:
                # Make sure inventory in Malmo matches up with what's in our inventory.
                player_inventory = self.player_inventories[player_index]
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
        return infos

    def _done(self) -> bool:
        return (
            self.timestep >= self.config["horizon"]
            or self.current_blocks == self.goal_blocks
        )

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
