from typing import Union

import numpy as np
from ray.rllib.policy.sample_batch import MultiAgentBatch, SampleBatch
from ray.rllib.utils.replay_buffers.replay_buffer import ReplayBuffer, StorageUnit


class PartialReplayBuffer(ReplayBuffer):
    """
    A replay buffer that only stores data added to it with a certain probability. This
    enables storing data from more episodes in the buffer, which is useful for training
    the model parts of the AlphaZero policy (goal and other agent action predictors).
    """

    def __init__(
        self,
        capacity: int = 10000,
        storage_unit: Union[str, StorageUnit] = "timesteps",
        storage_probability: float = 0.1,
        **kwargs,
    ):
        super().__init__(capacity, storage_unit, **kwargs)
        self.storage_probability = storage_probability

    def add(self, batch: Union[SampleBatch, MultiAgentBatch], **kwargs) -> None:
        if np.random.rand() < self.storage_probability:
            # Important to copy the batch because it's probably a slice of a larger
            # batch, so if we don't copy it will retain a reference and use more memory.
            super().add(batch.copy(), **kwargs)
