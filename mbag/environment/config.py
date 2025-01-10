from typing import List, Literal, Optional, Tuple, Type, TypedDict, Union, cast

from .goals import GoalGenerator, GoalGeneratorConfig, TransformedGoalGenerator
from .types import WorldSize

RewardScheduleEndpoints = List[Tuple[int, float]]
RewardSchedule = Union[float, RewardScheduleEndpoints]


class MalmoConfigDict(TypedDict, total=False):
    use_malmo: bool
    """
    Whether to connect to a real Minecraft instance with Project Malmo.
    """

    use_spectator: bool
    """
    Adds in a spectator player to observe the game from a 3rd person point of view.
    """

    rotate_spectator: bool
    """
    If true, the spectator will slowly rotate around the play area so that different
    angles of the building can be seen.
    """

    restrict_players: bool
    """
    Places a group of barrier blocks around players that prevents them from leaving
    the test world
    """

    video_dir: Optional[str]
    """
    Optional directory to record video from the game into.
    """

    ssh_args: Optional[List[Optional[List[str]]]]
    """
    If one of the Malmo instances is running over an SSH tunnel, then the entry in this
    list of the corresponding player should be set to a list of arguments that are
    passed to ssh in order to access the remote machine. This will be used to
    automatically set up necessary port forwarding.
    """

    start_port: int
    """
    Port to start looking for Malmo instances at (default 10000).
    """

    action_delay: float
    """
    The number of seconds to wait after each step to allow actions to complete
    in Malmo.
    """


class RewardsConfigDict(TypedDict, total=False):
    noop: RewardSchedule
    """
    The reward for doing any action which does nothing. This is usually either zero,
    or negative to discourage noops.
    """

    action: RewardSchedule
    """
    The reward for doing any action which is not a noop. This could be negative to
    introduce some cost for acting.
    """

    incorrect_action: RewardSchedule
    """
    The reward for taking a place/break action which is not correct. This is usually
    negative to discourage incorrect actions.
    """

    place_wrong: RewardSchedule
    """
    The reward for placing a block which is not correct, but in a place where a block
    should go. The negative of this is also given for breaking a block which is not
    correct.
    """

    own_reward_prop: RewardSchedule
    """
    A number from 0 to 1. At 0, it gives the normal reward function which takes into
    account all players actions. At 1, it gives only reward for actions that the
    specific player took.
    """

    get_resources: RewardSchedule
    """
    The reward for getting a resource block from the palette that the player
    did not have in their inventory previously.
    """


RewardsConfigDictKey = Literal[
    "noop", "action", "place_wrong", "own_reward_prop", "get_resources"
]


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


class EnchantmentDict(TypedDict, total=False):
    id: int
    """
    String id of Enchantment
    """

    level: int
    """
    The level of the enchantment to give to the item
    """


class ItemDict(TypedDict):
    id: str
    """
    String id of a Minecraft item.
    """

    count: int
    """
    The number of this item to place in the player inventory
    """

    enchantments: List[EnchantmentDict]


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

    give_items: List[ItemDict]
    """
    A list of items to give to the player at the beginning of the game.
    """


class MbagConfigDict(TypedDict, total=False):
    num_players: int
    horizon: int
    world_size: WorldSize
    random_start_locations: bool

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

    randomize_first_episode_length: bool
    """
    If True, the first episode will have a random length between 1 and horizon. This
    can be useful when training with an algorithm like PPO so that fragments
    of episodes are not strongly correlated across environments.
    """

    terminate_on_goal_completion: bool
    """
    If True (the default), the environment will terminate when the goal is completed.
    Otherwise the episode will continue until the horizon is reached.
    """

    truncate_on_no_progress_timesteps: Optional[int]
    """
    If specified, then the episode will truncate if no progress is made for this many
    timesteps. Progress is defined as an increase in goal percentage over the previous
    maximum.
    """

    _check_for_overlapping_players: bool
    """
    This should always be set to True except in certain cases where planning needs to
    be done given an environment state that has overlapping players, e.g., if evaluating
    the cross entropy of an MCTS-based policy on human data that contains overlapping
    players.
    """


DEFAULT_PLAYER_CONFIG: MbagPlayerConfigDict = {
    "player_name": None,
    "goal_visible": True,
    "is_human": False,
    "timestep_skip": 1,
    "rewards": {},
    "give_items": [],
}


DEFAULT_CONFIG: MbagConfigDict = {
    "num_players": 1,
    "horizon": 50,
    "world_size": (5, 5, 5),
    "random_start_locations": False,
    "randomize_first_episode_length": False,
    "terminate_on_goal_completion": True,
    "truncate_on_no_progress_timesteps": None,
    "_check_for_overlapping_players": True,
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
        "rotate_spectator": True,
        "restrict_players": False,
        "video_dir": None,
        "ssh_args": None,
        "start_port": 10000,
        "action_delay": 0.8,
    },
    "rewards": {
        "noop": 0.0,
        "action": 0.0,
        "incorrect_action": 0.0,
        "place_wrong": 0.0,
        "own_reward_prop": 0.0,
        "get_resources": 0,
    },
    "abilities": {
        "teleportation": True,
        "flying": True,
        "inf_blocks": True,
    },
}


DEFAULT_HUMAN_GIVE_ITEMS: List[ItemDict] = [  # type: ignore
    {
        "id": item_id,
        "count": 1,
        "enchantments": [
            # Gives silk touch enchantment, level defaults to max.
            {
                "id": 33,
                "level": 1,
            },
            {
                "id": 34,  # Gives unbreaking enchantment.
                "level": 3,  # Manually set the level.
            },
        ],
    }
    for item_id in ["diamond_pickaxe", "diamond_axe", "diamond_shovel"]
] + [
    {
        "id": "shears",
        "count": 1,
        "enchantments": [],
    }
]


def _merge_configs(config_a, config_b):
    if isinstance(config_a, dict):
        if not isinstance(config_b, dict):
            raise ValueError(f"Cannot merge {config_a} with {config_b}")
        merged_config_dict = {}
        for key in config_a.keys() | config_b.keys():
            if key in config_b:
                if key in config_a:
                    merged_config_dict[key] = _merge_configs(
                        config_a[key], config_b[key]
                    )
                else:
                    merged_config_dict[key] = config_b[key]
            else:
                merged_config_dict[key] = config_a[key]
        return merged_config_dict
    elif isinstance(config_a, list):
        if not isinstance(config_b, list):
            raise ValueError(f"Cannot merge {config_a} with {config_b}")
        if len(config_a) != len(config_b):
            return config_b
        merged_config_list = [_merge_configs(a, b) for a, b in zip(config_a, config_b)]
        return merged_config_list
    else:
        return config_b


def merge_configs(config_a: MbagConfigDict, config_b: MbagConfigDict) -> MbagConfigDict:
    """
    Merge the two configuration dictionaries, with the values in config_b taking
    precedence over those in config_a.
    """

    return cast(MbagConfigDict, _merge_configs(config_a, config_b))
