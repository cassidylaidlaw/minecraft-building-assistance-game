from ray.tune.registry import get_trainable_cls
from mbag.rllib.torch_models import MbagActionGateModel, MbagActionGateTransformerModel
from mbag.environment.mbag_env import MbagEnv
from gym import spaces
import numpy as np
from mbag.environment.types import (
    MbagAction,
)
from mbag.environment.mbag_env import MbagConfigDict
from mbag.rllib.rllib_env import MbagMultiAgentEnv
from mbag.rllib.train import ex, make_mbag_sacred_config


from mbag.environment.blocks import MinecraftBlocks


def test_transformer_action_gate():
    # environment_params: MbagConfigDict = {
    #         # "num_players": num_players,
    #         # "horizon": horizon,
    #         # "world_size": (width, height, depth),
    #         # "goal_generator_config": transformed_goal_generator_config,
    #         # "malmo": {
    #         #     "use_malmo": False,
    #         #     "player_names": None,
    #         # },
    #         # "goal_visibility": goal_visibility,
    #         # "timestep_skip": timestep_skip,
    #         # "rewards": {
    #         #     "noop": noop_reward,
    #         #     "action": action_reward,
    #         #     "place_wrong": place_wrong_reward,
    #         #     "own_reward_prop": own_reward_prop,
    #         #     "own_reward_prop_horizon": own_reward_prop_horizon,
    #         # },
    #         # "abilities": {
    #         #     "teleportation": teleportation,
    #         #     "flying": flying,
    #         # },
    #     }
    # env = MbagMultiAgentEnv(**environment_params)
    # world_obs_shape = (1, 1, 1, 1)
    # observation_space = spaces.Tuple(
    #     (spaces.Box(0, 255, world_obs_shape, dtype=np.uint8),)
    # )
    # action_space = spaces.Tuple(
    #         (
    #             spaces.Discrete(MbagAction.NUM_ACTION_TYPES),
    #             spaces.Discrete(np.prod((1, 1, 1))),
    #             spaces.Discrete(MinecraftBlocks.NUM_BLOCKS),
    #         )
    #     )
    # model = MbagActionGateTransformerModel(env.observation_space, env.action_space, 1, {}, "foo")
    # assert 1 == 1

    # config = {
    #         "model": "action_gate_transformer",
    #         "position_embedding_size": 6,
    #         "hidden_size": 39,
    #         "num_layers": 3,
    #         "num_heads": 1,
    #         "use_separated_transformer": True,

    #         }
    # simplest would be to initialize a model, check weights for backbone, train it for a bit, then check if the backbone weights have changed
    # also check if head has changed its weightssdfsdaf
    make_mbag_sacred_config(ex)
    sacred_config(None)["config"]
    print(config)

    trainer_class = get_trainable_cls(run)
    trainer = trainer_class(
        config,
    )
