import random
from typing import TYPE_CHECKING, Callable, Iterable, TypedDict, cast

import numpy as np

from ..blocks import MinecraftBlocks
from ..types import GOAL_BLOCK_STATES, GOAL_BLOCKS, WorldSize, num_world_obs_channels
from .goal_generator import GoalGenerator

if TYPE_CHECKING:
    from ray.rllib.policy.sample_batch import SampleBatch


class DemonstrationsGoalGeneratorConfig(TypedDict):
    demonstration_input: str
    data_augmentation: Callable[["SampleBatch"], "SampleBatch"]


class DemonstrationsGoalGenerator(GoalGenerator):

    default_config: DemonstrationsGoalGeneratorConfig = {
        "demonstration_input": "",
        "data_augmentation": lambda x: x,
    }
    config: DemonstrationsGoalGeneratorConfig

    def __init__(self, config):
        super().__init__(config)

        from ray.rllib.offline import JsonReader
        from ray.rllib.policy.sample_batch import SampleBatch, concat_samples

        demonstration_reader = JsonReader(self.config["demonstration_input"])
        demonstration_batch = concat_samples(
            list(cast(Iterable[SampleBatch], demonstration_reader.read_all_files()))
        )
        assert isinstance(demonstration_batch, SampleBatch)
        del demonstration_batch[SampleBatch.INFOS]

        self.demonstration_episodes = demonstration_batch.split_by_episode()
        self.num_remaining_goals = len(self.demonstration_episodes)

    def generate_goal(self, size: WorldSize) -> MinecraftBlocks:
        from ray.rllib.policy.sample_batch import SampleBatch

        episode = random.choice(self.demonstration_episodes)
        first_timestep = self.config["data_augmentation"](
            episode.slice(0, 1).decompress_if_needed()
        )

        width, height, depth = size
        world_obs_size = (
            (width + 2) * (height + 2) * (depth + 2) * num_world_obs_channels
        )
        world_obs = (
            cast(np.ndarray, first_timestep[SampleBatch.OBS])
            .flat[:world_obs_size]
            .reshape((num_world_obs_channels, width + 2, height + 2, depth + 2))
        )

        blocks = MinecraftBlocks(size)
        blocks.blocks[:] = world_obs[GOAL_BLOCKS, 1:-1, 1:-1, 1:-1]
        blocks.block_states[:] = world_obs[GOAL_BLOCK_STATES, 1:-1, 1:-1, 1:-1]
        return blocks
