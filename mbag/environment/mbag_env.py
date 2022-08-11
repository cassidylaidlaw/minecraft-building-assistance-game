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
import logging


from .blocks import MinecraftBlocks
from .types import (
    BlockLocation,
    MbagActionType,
    MbagWorldObsArray,
    WorldLocation,
    MbagAction,
    MbagActionTuple,
    MbagInfoDict,
    MbagObs,
    WorldSize,
    FacingDirection,
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

    player_names: Optional[List[str]]
    """
    Optional list of player names.
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

    goal_generator: Union[Type[GoalGenerator], str]
    goal_generator_config: GoalGeneratorConfig

    goal_visibility: List[bool]
    """
    List with one boolean for each player, indicating if the player can observe the
    goal.
    """

    timestep_skip: List[int]
    """
    Each element is how often the corresponding player can interact with the
    environment, i.e. 1 means every timestep, 5 means only every 5th timestep.
    """

    malmo: MalmoConfigDict
    """Configuration options for connecting to Minecraft with Project Malmo."""

    rewards: Union[RewardsConfigDict, List[RewardsConfigDict]]
    """
    Configuration options for environment reward, optionally different for each player.
    """

    abilities: AbilitiesConfigDict
    """Configuration for limits placed on the agent (flying, teleportation, inventory, etc...) """


DEFAULT_CONFIG: MbagConfigDict = {
    "num_players": 1,
    "horizon": 50,
    "world_size": (5, 5, 5),
    "goal_generator": TransformedGoalGenerator,
    "goal_generator_config": {
        "goal_generator": "random",
        "goal_generator_config": {},
        "goal_transforms": [
            {"transform": "add_grass"},
        ],
    },
    "goal_visibility": [True, False],
    "timestep_skip": [1] * 10,
    "malmo": {
        "use_malmo": False,
        "player_names": None,
        "use_spectator": False,
        "video_dir": None,
    },
    "rewards": {
        "noop": 0.0,
        "action": 0.0,
        "place_wrong": 0.0,
        "own_reward_prop": 0.0,
        "own_reward_prop_horizon": None,
    },
    "abilities": {"teleportation": True, "flying": True},
}

NO_ONE = 0
CURRENT_PLAYER = 1
OTHER_PLAYER = 2
NO_INTERACTION = -1


class MbagEnv(object):
    config: MbagConfigDict
    current_blocks: MinecraftBlocks
    goal_blocks: MinecraftBlocks
    player_locations: List[WorldLocation]
    player_directions: List[FacingDirection]
    last_interacted: np.ndarray
    timestep: int
    global_timestep: int

    def __init__(self, config: MbagConfigDict):
        self.config = copy.deepcopy(DEFAULT_CONFIG)
        self.config.update(config)
        if isinstance(self.config["world_size"], list):
            self.config["world_size"] = tuple(self.config["world_size"])

        self.config["malmo"] = copy.deepcopy(DEFAULT_CONFIG["malmo"])
        self.config["malmo"].update(config.get("malmo", {}))
        passed_rewards_config = config.get("rewards", {})

        if isinstance(passed_rewards_config, list):
            rewards_configs = []
            for incomplete_rewards_config in passed_rewards_config:
                rewards_config = copy.deepcopy(DEFAULT_CONFIG["rewards"])
                assert isinstance(rewards_config, dict)
                rewards_config.update(incomplete_rewards_config)
                rewards_configs.append(rewards_config)
            self.config["rewards"] = rewards_configs
        else:
            self.config["rewards"] = copy.deepcopy(DEFAULT_CONFIG["rewards"])
            assert isinstance(self.config["rewards"], dict)
            self.config["rewards"].update(passed_rewards_config)

        if (
            self.config["malmo"]["video_dir"] is not None
            and not self.config["malmo"]["use_spectator"]
        ):
            raise ValueError("Video recording requires using a spectator")

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
            if self.timestep % self.config["timestep_skip"][player_index] == 0:
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
            self._update_state_from_malmo()
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
        # marks if an action 'correct' meaning it directly contributed to the goal
        action_correct: bool = False

        if action.action_type == MbagAction.NOOP:
            pass
        elif action.action_type in [MbagAction.PLACE_BLOCK, MbagAction.BREAK_BLOCK]:
            prev_block = self.current_blocks[action.block_location]
            goal_block = self.goal_blocks[action.block_location]

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

            if place_break_result is not None:
                noop = False
                if self.config["abilities"]["teleportation"]:
                    self.player_locations[player_index] = (
                        place_break_result[0][0],
                        place_break_result[0][1],
                        place_break_result[0][2],
                    )
                self.last_interacted[action.block_location] = player_index

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
                prev_block, goal_block, partial_credit=True, player_index=player_index
            )
            new_goal_similarity = self._get_goal_similarity(
                new_block, goal_block, partial_credit=True, player_index=player_index
            )
            reward = new_goal_similarity - prev_goal_similarity
            action_correct = (
                action.action_type == MbagAction.PLACE_BLOCK and reward > 0
            ) or (action.action_type == MbagAction.BREAK_BLOCK and reward >= 0)
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

                player_location = self.player_locations[player_index]

                action_mask: Dict[MbagActionType, Tuple[WorldLocation, str]] = {
                    MbagAction.MOVE_POS_X: ((1, 0, 0), "moveeast 1"),
                    MbagAction.MOVE_NEG_X: ((-1, 0, 0), "movewest 1"),
                    MbagAction.MOVE_POS_Y: ((0, 1, 0), "tp"),
                    MbagAction.MOVE_NEG_Y: ((0, -1, 0), "tp"),
                    MbagAction.MOVE_POS_Z: ((0, 0, 1), "movesouth 1"),
                    MbagAction.MOVE_NEG_Z: ((0, 0, -1), "movenorth 1"),
                }
                dx, dy, dz = action_mask[action.action_type][0]
                new_player_location: WorldLocation = (
                    player_location[0] + dx,
                    player_location[1] + dy,
                    player_location[2] + dz,
                )

                if self._is_valid_player_space(new_player_location, player_index):
                    player_location = new_player_location

                    self.player_locations[player_index] = player_location

                    if self.config["malmo"]["use_malmo"]:
                        if action_mask[action.action_type][1] != "tp":
                            self.malmo_client.send_command(
                                player_index, action_mask[action.action_type][1]
                            )
                        else:
                            self.malmo_client.send_command(
                                player_index,
                                "tp " + " ".join(map(str, player_location)),
                            )
                else:
                    noop = True
        if noop:
            reward += self._get_reward_config_for_player(player_index)["noop"]
        else:
            reward += self._get_reward_config_for_player(player_index)["action"]

        info: MbagInfoDict = {
            "goal_similarity": self._get_goal_similarity(
                self.current_blocks[:],
                self.goal_blocks[:],
            ).sum(),
            "own_reward": reward,
            "own_reward_prop": self._get_own_reward_prop(player_index),
            "action_type": action.action_type if not noop else MbagAction.NOOP,
            "action_correct": action_correct and not noop,
        }

        return reward, info

    def _is_valid_player_space(self, player_location: WorldLocation, player_index: int):
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
        player_index: Optional[int] = None,
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
        if self.config["goal_visibility"][player_index]:
            world_obs[GOAL_BLOCKS] = self.goal_blocks.blocks
            world_obs[GOAL_BLOCK_STATES] = self.goal_blocks.block_states

        # Add locations to the observation if the locations are actually meaningful
        # (i.e., if players do not have teleportation abilities).
        if not self.config["abilities"]["teleportation"]:
            # Current player location is marked with 1 in the observation.
            self._add_player_location_to_world_obs(
                world_obs, self.player_locations[player_index], CURRENT_PLAYER
            )
            # Now other player locations are marked starting with 2.
            for other_player_index, other_player_location in enumerate(
                self.player_locations[:player_index]
                + self.player_locations[player_index + 1 :]
            ):
                self._add_player_location_to_world_obs(
                    world_obs, other_player_location, other_player_index + OTHER_PLAYER
                )

        f = np.vectorize(self._observation_from_player_perspective)
        world_obs[LAST_INTERACTED] = f(self.last_interacted, player_index)

        return (world_obs,)

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

    def _observation_from_player_perspective(self, x: Optional[int], player_index: int):
        if x == NO_INTERACTION:
            return NO_ONE
        elif player_index == x:
            return CURRENT_PLAYER
        else:
            return player_index + OTHER_PLAYER

    def _get_reward_config_for_player(self, player_index: int) -> RewardsConfigDict:
        if isinstance(self.config["rewards"], list):
            return self.config["rewards"][player_index]
        else:
            return self.config["rewards"]

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
