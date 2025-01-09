import logging
from typing import List, Optional, Union

import numpy as np
from ray.rllib.policy.sample_batch import MultiAgentBatch, SampleBatch
from ray.rllib.utils.replay_buffers.multi_agent_replay_buffer import (
    MultiAgentReplayBuffer,
    merge_dicts_with_warning,
)
from ray.rllib.utils.replay_buffers.replay_buffer import ReplayBuffer, StorageUnit
from ray.rllib.utils.typing import PolicyID, SampleBatchType

logger = logging.getLogger(__name__)


def timeslice_along_seq_lens(
    sample_batch: SampleBatch,
    seq_lens: Optional[List[int]] = None,
    zero_pad_max_seq_len: int = 0,
    zero_init_states: bool = True,
) -> List["SampleBatch"]:
    """
    Similar to Rllib timeslice_along_seq_lens_with_overlap, but fixes an issue
    with state_in_N being set incorrectly..
    """

    if seq_lens is None:
        seq_lens = sample_batch.get(SampleBatch.SEQ_LENS)
    else:
        logger.warning(
            "Found sequencing information in a batch that will be "
            "ignored when slicing. Ignore this warning if you know "
            "what you are doing."
        )

    if seq_lens is None:
        max_seq_len = zero_pad_max_seq_len
        logger.warning(
            "Trying to slice a batch along sequences without "
            "sequence lengths being provided in the batch. Batch will "
            "be sliced into slices of size "
            "{} = zero_pad_max_seq_len.".format(
                max_seq_len,
            )
        )
        num_seq_lens, last_seq_len = divmod(len(sample_batch), max_seq_len)
        seq_lens = [zero_pad_max_seq_len] * num_seq_lens + (
            [last_seq_len] if last_seq_len else []
        )

    assert (
        seq_lens is not None and len(seq_lens) > 0
    ), "Cannot timeslice along `seq_lens` when `seq_lens` is empty or None!"
    # Generate n slices based on seq_lens.
    start = 0
    slices = []
    for seq_len in seq_lens:
        pre_begin = start
        slice_begin = start
        end = start + seq_len
        slices.append((pre_begin, slice_begin, end))
        start += seq_len

    timeslices = []
    for seq_index, (begin, slice_begin, end) in enumerate(slices):
        zero_length = None
        data_begin = 0
        zero_init_states_ = zero_init_states
        if begin < 0:
            zero_length = 0
            data_begin = slice_begin
            zero_init_states_ = True
        else:
            eps_ids = sample_batch[SampleBatch.EPS_ID][begin if begin >= 0 else 0 : end]
            is_last_episode_ids = eps_ids == eps_ids[-1]
            if not is_last_episode_ids[0]:
                zero_length = int(sum(1.0 - is_last_episode_ids))
                data_begin = begin + zero_length
                zero_init_states_ = True

        if zero_length is not None:
            data = {
                k: np.concatenate(
                    [
                        np.zeros(shape=(zero_length,) + v.shape[1:], dtype=v.dtype),
                        v[data_begin:end],
                    ]
                )
                for k, v in sample_batch.items()
                if k != SampleBatch.SEQ_LENS
            }
        else:
            data = {
                k: v[begin:end]
                for k, v in sample_batch.items()
                if k != SampleBatch.SEQ_LENS
            }

        if zero_init_states_:
            i = 0
            key = "state_in_{}".format(i)
            while key in data:
                data[key] = np.zeros_like(sample_batch[key][0:1])
                # Del state_out_n from data if exists.
                data.pop("state_out_{}".format(i), None)
                i += 1
                key = "state_in_{}".format(i)
        else:
            i = 0
            key = "state_in_{}".format(i)
            while key in data:
                data[key] = sample_batch["state_in_{}".format(i)][
                    seq_index : seq_index + 1
                ]
                i += 1
                key = "state_in_{}".format(i)

        timeslices.append(SampleBatch(data, seq_lens=[end - begin]))

    # Zero-pad each slice if necessary.
    if zero_pad_max_seq_len > 0:
        for ts in timeslices:
            ts.right_zero_pad(max_seq_len=zero_pad_max_seq_len, exclude_states=True)

    return timeslices


class FixedMultiAgentReplayBuffer(MultiAgentReplayBuffer):
    def _add_to_underlying_buffer(
        self, policy_id: PolicyID, batch: SampleBatchType, **kwargs
    ) -> None:
        if self.storage_unit is not StorageUnit.SEQUENCES:
            super()._add_to_underlying_buffer(policy_id, batch, **kwargs)
            return

        # Merge kwargs, overwriting standard call arguments
        kwargs = merge_dicts_with_warning(self.underlying_buffer_call_args, kwargs)

        assert isinstance(batch, SampleBatch)
        assert self.replay_burn_in == 0

        timeslices = timeslice_along_seq_lens(
            sample_batch=batch,
            seq_lens=(
                batch.get(SampleBatch.SEQ_LENS)
                if self.replay_sequence_override
                else None
            ),
            zero_pad_max_seq_len=self.replay_sequence_length,
            zero_init_states=self.replay_zero_init_states,
        )
        for slice in timeslices:
            self.replay_buffers[policy_id].add(slice, **kwargs)


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
