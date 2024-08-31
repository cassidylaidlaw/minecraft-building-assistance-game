from typing import Any, List, cast

import numpy as np
import pytest

from mbag.agents.action_distributions import MbagActionDistribution
from mbag.environment import MbagEnv
from mbag.environment.types import MbagObs

try:
    import torch
    from ray.rllib.models.catalog import ModelCatalog
    from ray.rllib.policy.sample_batch import SampleBatch

    from mbag.rllib.mixture_model import MixtureModel
except ImportError:
    pass


@pytest.mark.uses_rllib
def test_mixture_model(*, recurrent=False):
    env = MbagEnv(
        {
            "goal_generator": "random",
            "horizon": 100,
            "abilities": {
                "teleportation": True,
                "flying": True,
                "inf_blocks": True,
            },
            "world_size": (5, 5, 5),
        }
    )
    flat_actions = MbagActionDistribution.get_action_mapping(env.config)
    num_flat_actions = len(flat_actions)

    model_config = {
        "custom_action_dist": "mbag_bilevel_categorical",
        "custom_model": "mbag_transformer_model",
        "custom_model_config": {
            "embedding_size": 8,
            "env_config": env.config,
            "hidden_size": 64,
            "interleave_lstm": recurrent,
            "line_of_sight_masking": False,
            "lstm_depth": None,
            "mask_action_distribution": True,
            "mask_goal": False,
            "mask_other_players": True,
            "num_action_layers": 2,
            "num_heads": 4,
            "num_layers": 8 if recurrent else 6,
            "num_lstm_layers": 1,
            "num_value_layers": 2,
            "position_embedding_size": 18,
            "scale_obs": True,
            "use_extra_features": True,
            "use_per_location_lstm": False,
            "use_prev_blocks": False,
            "use_separated_transformer": True,
            "vf_scale": 1.0,
        },
        "max_seq_len": 1500,
        "vf_share_layers": True,
    }

    preprocessor = ModelCatalog.get_preprocessor_for_space(env.observation_space)
    model = ModelCatalog.get_model_v2(
        env.observation_space,
        env.action_space,
        num_outputs=num_flat_actions,
        model_config={
            "custom_model": "mbag_mixture",
            "custom_action_dist": "mbag_bilevel_categorical",
            "custom_model_config": {
                "model_configs": [model_config, model_config],
            },
        },
        framework="torch",
    )
    assert isinstance(model, MixtureModel)
    model.eval()

    component_state_len = len(model.components[0].get_initial_state())

    # Construct a fake batch.
    batch_size = 8
    seq_len = 5
    obs_list: List[MbagObs] = []
    actions: List[int] = []
    for seq_index in range(batch_size):
        (obs,), _ = env.reset()
        obs_list.append(obs)
        for t in range(seq_len):
            world_obs, inventory_obs, timestep = obs
            action_mask: np.ndarray = MbagActionDistribution.get_mask_flat(
                env.config,
                (world_obs[None], inventory_obs[None], timestep[None]),
                line_of_sight_masking=False,
            )[0]
            action_dist = action_mask.astype(float)
            action_dist /= action_dist.sum()
            flat_action = np.random.choice(
                num_flat_actions,
                p=action_dist,
            )
            actions.append(flat_action)
            action = tuple(flat_actions[flat_action])
            (obs,), _, _, _ = env.step([action])
            if t < seq_len - 1:
                obs_list.append(obs)

    # Test that sampling actions from the model randomizes over the two components.
    obs_batch = np.stack(
        [preprocessor.transform(cast(Any, obs)) for obs in obs_list[::seq_len]]
    )
    state_in = [
        torch.stack([state_piece for _ in range(batch_size)])
        for state_piece in cast(List[torch.Tensor], model.get_initial_state())
    ]
    model_out, state_out = model(
        {SampleBatch.OBS: torch.from_numpy(obs_batch)},
        state_in,
        torch.tensor([1] * batch_size),
    )

    for i in range(batch_size):
        # model_out[0] is the log-prob distribution over mixture components
        component_index = int(torch.argmax(state_out[0][i]).item())
        assert state_out[0][i][component_index] == 0
        assert state_out[0][i][1 - component_index] == -torch.inf

        # Compare to just running that component alone.
        component_model = model.components[component_index]
        component_state_start = 1 + component_index * component_state_len
        component_state_end = component_state_start + component_state_len

        component_out, component_state_out = component_model(
            {SampleBatch.OBS: torch.from_numpy(obs_batch[i : i + 1])},
            [
                state_piece[i : i + 1]
                for state_piece in state_in[component_state_start:component_state_end]
            ],
            torch.tensor([1]),
        )
        np.testing.assert_allclose(
            torch.log_softmax(model_out[i], dim=0).cpu().detach().numpy(),
            torch.log_softmax(component_out[0], dim=0).cpu().detach().numpy(),
            rtol=1e-4,
            atol=1e-4,
        )
        for state_piece, component_state_piece in zip(
            state_out[component_state_start:component_state_end], component_state_out
        ):
            np.testing.assert_allclose(
                state_piece[i].cpu().detach().numpy(),
                component_state_piece[0].cpu().detach().numpy(),
                rtol=1e-4,
                atol=1e-4,
            )

    # Test that Bayesian inference over the mixture works.
    obs_batch = np.stack([preprocessor.transform(cast(Any, obs)) for obs in obs_list])
    state_in[0] = torch.tensor(
        [[-np.log(2), -np.log(2)] for _ in range(batch_size)], dtype=torch.float32
    )
    model_out, state_out = model(
        {
            SampleBatch.OBS: torch.from_numpy(obs_batch),
            SampleBatch.ACTIONS: torch.tensor(actions),
        },
        state_in,
        torch.tensor([seq_len] * batch_size),
    )
    for i in range(batch_size):
        seq_slice = slice(i * seq_len, (i + 1) * seq_len)
        component_0_out, component_0_state_out = model.components[0](
            {SampleBatch.OBS: torch.from_numpy(obs_batch[seq_slice])},
            [
                state_piece[None]
                for state_piece in model.components[0].get_initial_state()
            ],
            torch.tensor([seq_len]),
        )
        component_1_out, component_1_state_out = model.components[1](
            {SampleBatch.OBS: torch.from_numpy(obs_batch[seq_slice])},
            [
                state_piece[None]
                for state_piece in model.components[1].get_initial_state()
            ],
            torch.tensor([seq_len]),
        )

        action_seq = torch.tensor(actions)[seq_slice]
        component_0_probs = torch.softmax(component_0_out, dim=1)
        component_1_probs = torch.softmax(component_1_out, dim=1)
        component_0_action_probs = component_0_probs[torch.arange(seq_len), action_seq]
        component_1_action_probs = component_1_probs[torch.arange(seq_len), action_seq]

        component_0_cum_prod = torch.cumprod(
            torch.concatenate([torch.tensor([0.5]), component_0_action_probs], dim=0),
            dim=0,
        )
        component_1_cum_prod = torch.cumprod(
            torch.concatenate([torch.tensor([0.5]), component_1_action_probs], dim=0),
            dim=0,
        )

        component_0_weight = component_0_cum_prod / (
            component_0_cum_prod + component_1_cum_prod
        )
        component_1_weight = component_1_cum_prod / (
            component_0_cum_prod + component_1_cum_prod
        )

        np.testing.assert_allclose(
            torch.softmax(model_out[seq_slice], dim=1).cpu().detach().numpy(),
            (
                component_0_weight[:-1, None] * component_0_probs
                + component_1_weight[:-1, None] * component_1_probs
            )
            .cpu()
            .detach()
            .numpy(),
            rtol=1e-4,
            atol=1e-4,
        )

        np.testing.assert_allclose(
            state_out[0][i].exp().cpu().detach().numpy(),
            np.array(
                [
                    component_0_weight[seq_len].item(),
                    component_1_weight[seq_len].item(),
                ]
            ),
            rtol=1e-4,
            atol=1e-4,
        )

        for state_piece, component_state_piece in zip(
            state_out[1:], component_0_state_out + component_1_state_out
        ):
            np.testing.assert_allclose(
                state_piece[i].cpu().detach().numpy(),
                component_state_piece[0].cpu().detach().numpy(),
                rtol=1e-4,
                atol=1e-4,
            )


@pytest.mark.uses_rllib
def test_mixture_model_recurrent():
    test_mixture_model(recurrent=True)
