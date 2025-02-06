import copy
import pickle
from typing import Union

from mbag.environment.config import MbagConfigDict, RewardsConfigDict


class OldHumanDataUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "mbag.environment.types" and name in [
            "MbagActionTuple",
            "MbagAction",
            "MbagActionType",
        ]:
            module = "mbag.environment.actions"
        return super().find_class(module, name)


def convert_old_rewards_config_to_new(
    old_rewards_config: Union[RewardsConfigDict, dict],
) -> RewardsConfigDict:
    new_rewards_config: RewardsConfigDict = {}

    if "own_reward_prop" in old_rewards_config:
        own_reward_prop_start = old_rewards_config["own_reward_prop"]
        assert isinstance(own_reward_prop_start, (int, float))
        own_reward_prop_horizon = old_rewards_config.get(
            "own_reward_prop_horizon", None
        )
        if own_reward_prop_horizon is None:
            new_rewards_config["own_reward_prop"] = own_reward_prop_start
        else:
            if (
                not isinstance(own_reward_prop_horizon, (int, float))
                or int(own_reward_prop_horizon) != own_reward_prop_horizon
            ):
                raise ValueError(
                    f"own_reward_prop_horizon must be an integer, got {own_reward_prop_horizon}"
                )
            new_rewards_config["own_reward_prop"] = [
                (0, own_reward_prop_start),
                (int(own_reward_prop_horizon), 0.0),
            ]

    if "noop" in old_rewards_config:
        new_rewards_config["noop"] = old_rewards_config["noop"]
    if "action" in old_rewards_config:
        new_rewards_config["action"] = old_rewards_config["action"]
    if "place_wrong" in old_rewards_config:
        new_rewards_config["place_wrong"] = old_rewards_config["place_wrong"]
    if "get_resources" in old_rewards_config:
        new_rewards_config["get_resources"] = old_rewards_config["get_resources"]

    return new_rewards_config


def convert_old_config_to_new(
    old_config: MbagConfigDict,
) -> MbagConfigDict:
    mbag_config = copy.deepcopy(old_config)
    if "rewards" in mbag_config:
        mbag_config["rewards"] = convert_old_rewards_config_to_new(
            mbag_config["rewards"]
        )
    for player_config in mbag_config.get("players", []):
        if "rewards" in player_config:
            player_config["rewards"] = convert_old_rewards_config_to_new(
                player_config["rewards"]
            )
    return mbag_config
