from typing import Dict, List, Tuple, cast, Any
import warnings
import torch
import numpy as np
import copy
from torch import nn
import torch.nn.functional as F  # noqa: N812
from abc import ABC, abstractmethod
from typing_extensions import TypedDict
from gym import spaces
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.algorithms.alpha_zero.models.custom_torch_models import ActorCriticModel
from ray.rllib.utils.torch_utils import convert_to_torch_tensor
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.policy.rnn_sequencing import add_time_dimension

from mbag.agents.action_distributions import MbagActionDistribution
from mbag.environment.blocks import MinecraftBlocks
from mbag.environment.mbag_env import (
    MbagConfigDict,
    DEFAULT_CONFIG as DEFAULT_ENV_CONFIG,
)
from mbag.environment.types import (
    CURRENT_BLOCKS,
    GOAL_BLOCKS,
    PLAYER_LOCATIONS,
    LAST_INTERACTED,
)


class MbagModel(ABC, TorchModelV2):
    """
    A model to be used with MbagAutoregressiveActionDistribution.
    """

    @abstractmethod
    def block_id_model(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        ...


class MbagModelConfig(TypedDict, total=False):
    env_config: MbagConfigDict
    """Environment configuration."""
    embedding_size: int
    """Block ID embedding size."""
    use_extra_features: bool
    """Use extra hand-designed features as input to the network."""
    mask_goal: bool
    """Remove goal information from observations before passing into the network."""
    fake_state: bool
    """Whether to add fake state to this model so that it's treated as recurrent."""
    hidden_size: int
    """Size of hidden layers."""
    num_action_layers: int
    """Number of extra layers for the action head."""
    num_value_layers: int
    """Number of extra layers for the value head."""
    use_per_location_lstm: bool
    """Include a LSTM operating per-location."""


DEFAULT_CONFIG: MbagModelConfig = {
    "env_config": DEFAULT_ENV_CONFIG,
    "embedding_size": 8,
    "use_extra_features": False,
    "mask_goal": False,
    "fake_state": False,
    "hidden_size": 16,
    "num_action_layers": 1,
    "num_value_layers": 1,
}


class MbagTorchModel(ActorCriticModel):
    """
    This base class implements common functionality for PyTorch MBAG models such
    as block type embedding, separate policy and value networks and the value head.
    """

    MASK_LOGIT = -1e8

    _logits: torch.Tensor

    def __init__(
        self,
        obs_space: spaces.Space,
        action_space: spaces.Space,
        num_outputs: int,
        model_config,
        name,
        **kwargs,
    ):
        ActorCriticModel.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )

        obs_space = cast(Any, obs_space).original_space
        if isinstance(obs_space, spaces.Dict):
            obs_space = obs_space.spaces["obs"]
        assert isinstance(obs_space, spaces.Tuple)
        self.world_obs_space: spaces.Box = obs_space[0]
        self.world_size = self.world_obs_space.shape[-3:]

        extra_config = copy.deepcopy(DEFAULT_CONFIG)
        extra_config.update(cast(MbagModelConfig, kwargs))
        self.env_config = extra_config["env_config"]
        self.embedding_size = extra_config["embedding_size"]
        self.use_extra_features = extra_config["use_extra_features"]
        self.mask_goal = extra_config["mask_goal"]
        self.fake_state: bool = extra_config["fake_state"]
        self.hidden_size = extra_config["hidden_size"]
        self.num_action_layers = extra_config["num_action_layers"]
        self.num_value_layers = extra_config["num_value_layers"]
        self.use_per_location_lstm = extra_config["use_per_location_lstm"]

        self.block_id_embedding = nn.Embedding(
            num_embeddings=len(MinecraftBlocks.ID2NAME),
            embedding_dim=self.embedding_size,
        )
        self.player_id_embedding = nn.Embedding(
            # Assume there are no more than 16 players, could be an issue down the line?
            num_embeddings=16,
            embedding_dim=self.embedding_size,
        )

        self.vf_share_layers: bool = model_config["vf_share_layers"]
        if self.vf_share_layers:
            self.backbone = self._construct_backbone()
        else:
            self.action_backbone = self._construct_backbone()
            self.value_backbone = self._construct_backbone(is_value_network=True)

        self.action_head = self._construct_action_head()
        self.value_head = self._construct_value_head()
        self.goal_head = self._construct_goal_head()

        if self.use_per_location_lstm:
            assert self.vf_share_layers
            assert not self.model_config.get("_time_major", False)
            self.per_location_lstm = nn.LSTM(
                input_size=self.hidden_size,
                hidden_size=self.hidden_size,
                num_layers=1,
                batch_first=True,
            )

    def _get_in_planes(self) -> int:
        """
        Return how many "planes" of data of size embedding_size are present in the
        embedded observation.
        """

        # We have in-planes for current blocks, player locations, and
        # goal blocks if mask_goal is False.
        return 3 if self.mask_goal else 4  # TODO: update if we add more

    def _get_in_channels(self) -> int:
        """Get the number of channels in the embedded observation."""
        in_channels = self._get_in_planes() * self.embedding_size
        # Add inventory observation as extra input channels.
        in_channels += MinecraftBlocks.NUM_BLOCKS
        # Timestep observation
        in_channels += 1
        if self.use_extra_features:
            in_channels += 1
        return in_channels

    def _get_head_in_channels(self) -> int:
        """
        Get the number of channels output from the backbone which are used as input to
        the value and block ID heads.
        """

        return self.hidden_size

    def _construct_backbone(self, is_value_network=False) -> nn.Module:
        """
        Construct the main backbone of the model. This takes as input the result of
        _get_embedded_obs and should output a tensor of shape
            (batch_size, self._get_head_in_channels()) + self.world_size.
        When vf_share_layers is True, this is called once; when vf_share_layers is
        False, it is called twice, and the value backbone is passed
        is_value_network=True.
        """

        raise NotImplementedError()

    def _construct_action_head(self) -> nn.Module:
        """
        Construct the head which outputs the action distribution logits.
        """

        action_head_layers: List[nn.Module] = []
        for layer_index in range(self.num_action_layers):
            action_head_layers.append(
                nn.Conv3d(
                    self._get_head_in_channels()
                    if layer_index == 0
                    else self.hidden_size,
                    MbagActionDistribution.NUM_CHANNELS
                    if layer_index == self.num_action_layers - 1
                    else self.hidden_size,
                    kernel_size=1,
                )
            )
            if layer_index < self.num_action_layers - 1:
                action_head_layers.append(nn.LeakyReLU())

        # Tamp down probabilities of actions which don't require a block location,
        # since otherwise these dominate the action distribution at the start of
        # training.
        logit_layer = cast(nn.Conv3d, action_head_layers[-1])
        assert logit_layer.bias is not None
        # for action_type in MbagAction.ACTION_TYPES:
        #     if action_type in MbagAction.BLOCK_ID_ACTION_TYPES:
        #         channel = MbagActionDistribution.ACTION_TYPE2CHANNEL[action_type]
        #         logit_layer.bias.data[channel] -= np.log(MinecraftBlocks.NUM_BLOCKS)

        return nn.Sequential(*action_head_layers)

    def _construct_value_head(self) -> nn.Module:
        """
        Construct the head which takes in the output of the value backbone and
        outputs a one-dimensional value estimate.
        """

        value_head_layers: List[nn.Module] = [
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
        ]
        for layer_index in range(self.num_value_layers):
            value_head_layers.append(
                nn.Linear(
                    self._get_head_in_channels()
                    if layer_index == 0
                    else self.hidden_size,
                    1 if layer_index == self.num_value_layers - 1 else self.hidden_size,
                )
            )
            if layer_index < self.num_value_layers - 1:
                value_head_layers.append(nn.LeakyReLU())
        return nn.Sequential(*value_head_layers)

    def _construct_goal_head(self) -> nn.Module:
        """
        Construct the head which takes in the output of the value backbone and
        outputs a goal estimate.
        """

        return nn.Sequential(
            nn.Conv3d(self._get_head_in_channels(), self.hidden_size, 1),
            nn.LeakyReLU(),
            nn.Conv3d(self.hidden_size, MinecraftBlocks.NUM_BLOCKS, 1),
        )

    def _get_embedded_obs(
        self,
        world_obs: torch.Tensor,
        inventory_obs: torch.Tensor,
        timestep: torch.Tensor,
    ):
        """
        Transform a raw observation into the input for the network backbone.
        """

        embedded_blocks = self.block_id_embedding(world_obs[:, CURRENT_BLOCKS])
        embedded_obs_pieces = [embedded_blocks]
        if not self.mask_goal:
            embedded_goal_blocks = self.block_id_embedding(world_obs[:, GOAL_BLOCKS])
            embedded_obs_pieces.append(embedded_goal_blocks)
        embedded_player_locations = self.player_id_embedding(
            world_obs[:, PLAYER_LOCATIONS]
        )
        embedded_obs_pieces.append(embedded_player_locations)
        embedded_obs_pieces.append(
            inventory_obs[:, None, None, None, :].expand(
                *embedded_obs_pieces[0].size()[:-1], -1
            )
        )
        embedded_obs_pieces.append(
            timestep[:, None, None, None, None].expand(
                *embedded_obs_pieces[0].size()[:-1], 1
            )
        )
        if self.use_extra_features:
            # Feature for if goal block is the same as the current block at each
            # location.
            embedded_obs_pieces.append(
                (world_obs[:, 0] == world_obs[:, 2]).float()[..., None]
            )
        last_interacted = self.player_id_embedding(world_obs[:, LAST_INTERACTED])
        embedded_obs_pieces.append(last_interacted)
        embedded_obs = torch.cat(embedded_obs_pieces, dim=-1)

        return embedded_obs.permute(0, 4, 1, 2, 3)

    def _run_lstm(
        self, backbone_out: torch.Tensor, state_in: List[torch.Tensor], seq_lens
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        flat_backbone_out = backbone_out.flatten(start_dim=1)
        flat_backone_out_with_time: torch.Tensor = add_time_dimension(
            flat_backbone_out,
            seq_lens=seq_lens,
            framework="torch",
            time_major=False,
        )
        # Should be of size (batch_size, max_seq_len, hidden_size, width, height, depth).
        backbone_out_with_time = flat_backone_out_with_time.reshape(
            *flat_backone_out_with_time.size()[:2],
            *backbone_out.size()[1:],
        )
        batch_size, max_seq_len = backbone_out_with_time.size()[:2]
        assert (
            backbone_out_with_time.size()[2:] == (self.hidden_size,) + self.world_size
        )
        backbone_out_per_location = backbone_out_with_time.permute(
            0, 3, 4, 5, 1, 2
        ).flatten(end_dim=3)
        # State in should be of size (batch_size, hidden_size, width, height, depth).
        state_in_per_location = tuple(
            state.permute(0, 2, 3, 4, 1).flatten(end_dim=3)[None].contiguous()
            for state in state_in
        )

        lstm_out_per_location: torch.Tensor
        state_out_per_location: torch.Tensor
        lstm_out_per_location, state_out_per_location = self.per_location_lstm(
            backbone_out_per_location, state_in_per_location
        )
        lstm_out_with_time = lstm_out_per_location.reshape(
            batch_size,
            *self.world_size,
            max_seq_len,
            self.hidden_size,
        ).permute(0, 4, 5, 1, 2, 3)
        state_out = [
            state.reshape(batch_size, *self.world_size, self.hidden_size).permute(
                0, 4, 1, 2, 3
            )
            for state in state_out_per_location
        ]

        lstm_out = lstm_out_with_time.flatten(end_dim=1)

        return lstm_out, state_out

    def forward(self, input_dict, state, seq_lens, mask_logits=True):
        # Seems like AlphaZero trainer likes to give just observations instead of
        # an input dict.
        try:
            obs = input_dict["obs"]
        except TypeError:
            obs = input_dict

        self._world_obs, self._inventory_obs, self._timestep = obs
        self._world_obs = self._world_obs.long()
        self._inventory_obs = self._inventory_obs.long()
        self._embedded_obs = self._get_embedded_obs(
            self._world_obs,
            self._inventory_obs,
            self._timestep,
        )
        if self._embedded_obs.requires_grad:
            self._embedded_obs.retain_grad()  # TODO: remove

        if self.vf_share_layers:
            self._backbone_out = self.backbone(self._embedded_obs)
        else:
            self._backbone_out = self.action_backbone(self._embedded_obs)
        self._backbone_out_shape = self._backbone_out.size()[1:]
        assert self._backbone_out_shape[0] == self._get_head_in_channels()

        if self.use_per_location_lstm:
            self._backbone_out, state = self._run_lstm(
                self._backbone_out, state, seq_lens
            )

        self._logits = self.action_head(self._backbone_out)
        self._flat_logits = MbagActionDistribution.to_flat_torch_logits(
            self.env_config, self._logits
        )

        if mask_logits:
            numpy_mask = MbagActionDistribution.get_mask_flat(
                self.env_config, convert_to_numpy(obs)
            )
            mask = torch.from_numpy(numpy_mask).to(self._flat_logits.device)
            self._flat_logits[~mask] = MbagTorchModel.MASK_LOGIT

        return self._flat_logits, state

    @property
    def logits(self) -> torch.Tensor:
        return self._flat_logits

    def block_id_model(self, head_input: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def value_function(self):
        if self.vf_share_layers:
            return self.value_head(self._backbone_out).squeeze(1)
        else:
            return self.value_head(self.value_backbone(self._embedded_obs)).squeeze(1)

    def goal_function(self):
        return self.goal_head(self._backbone_out)

    def get_initial_state(self):
        if self.use_per_location_lstm:
            return [torch.zeros((self.hidden_size, *self.world_size)) for _ in range(2)]
        else:
            if self.fake_state:
                return [np.zeros(1)]
            else:
                return super().get_initial_state()

    def compute_priors_and_value(self, input_dict):
        obs = convert_to_torch_tensor(
            self.preprocessor.transform(input_dict["obs"])[None]
        )
        input_dict = restore_original_dimensions(obs, self.obs_space, "torch")

        with torch.no_grad():
            model_out = self.forward(input_dict, None, [1], mask_logits=False)
            logits, _ = model_out
            value = self.value_function()
            logits, value = torch.squeeze(logits), torch.squeeze(value)
            priors = nn.Softmax(dim=-1)(logits)

            priors = priors.cpu().numpy()
            value = value.cpu().numpy()

            return priors, value


class ResidualBlock(nn.Module):
    """
    Implements a residual network block with two 3D convolutions, batch norm,
    and ReLU.
    """

    def __init__(
        self,
        channels: int,
        filter_size: int = 3,
        use_bn: bool = True,
        use_skip_connection: bool = True,
    ):
        super().__init__()

        self.conv1 = nn.Conv3d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=filter_size,
            stride=1,
            padding=(filter_size - 1) // 2,
        )
        self.bn1 = nn.BatchNorm3d(channels) if use_bn else nn.Identity()
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv3d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=filter_size,
            stride=1,
            padding=(filter_size - 1) // 2,
        )
        self.bn2 = nn.BatchNorm3d(channels) if use_bn else nn.Identity()
        self.relu2 = nn.ReLU()

        self.use_skip_connection = use_skip_connection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.use_skip_connection:
            out = out + x
        out = self.relu2(out)
        return out


class UNet3d(nn.Module):
    """
    Implements a model similar to U-Nets, but for 3d data.
    """

    def __init__(
        self,
        size: int,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        fc_layer: nn.Module = nn.Identity(),
        grow_factor: float = 2.0,
        use_bn: bool = False,
    ):
        """
        Expects inputs of shape
        (batch, in_channels, size, size, size)
        and produces outputs of shape
        (batch, out_channels, size, size, size)
        """
        super().__init__()

        self.size = size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.fc_layer = fc_layer

        self.down_layers: List[nn.Module] = []
        self.up_layers: List[nn.Module] = []
        layer_size = self.size
        for layer_index in range(self.num_layers):
            down_in_channels = self.in_channels * int(grow_factor**layer_index)
            down_out_channels = self.in_channels * int(grow_factor ** (layer_index + 1))
            down_layer = nn.Sequential(
                nn.Conv3d(
                    in_channels=down_in_channels,
                    out_channels=down_out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                *([nn.BatchNorm3d(down_out_channels)] if use_bn else []),
                nn.LeakyReLU(),
            )
            self.down_layers.append(down_layer)
            self.add_module(f"down_{layer_index}", down_layer)

            up_in_channels = (
                self.in_channels * 2 * int(grow_factor ** (layer_index + 1))
            )
            up_out_channels = self.in_channels * int(grow_factor**layer_index)
            up_layer = nn.Sequential(
                nn.ConvTranspose3d(
                    in_channels=up_in_channels,
                    out_channels=up_out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1 if layer_size % 2 == 0 else 0,
                ),
                *([nn.BatchNorm3d(up_out_channels)] if use_bn else []),
                nn.LeakyReLU(),
            )
            self.up_layers.append(up_layer)
            self.add_module(f"up_{layer_index}", up_layer)

            layer_size = (layer_size + 1) // 2

        self.fc_layer_size = (
            (layer_size**3) * self.in_channels * int(grow_factor**self.num_layers)
        )

        # Final 1x1x1 convolution to get the right number of out channels.
        self.final_layer = nn.Conv3d(
            in_channels=self.in_channels * 2,
            out_channels=self.out_channels,
            kernel_size=1,
        )

    def set_fc_layer(self, fc_layer: nn.Module):
        self.fc_layer = fc_layer

    def forward(self, inputs: torch.Tensor, *extra_fc_inputs):
        activations = [inputs]
        for down_layer in self.down_layers:
            activations.append(down_layer(activations[-1]))

        fc_layer_inputs = activations[-1].flatten(start_dim=1)
        outputs = self.fc_layer(fc_layer_inputs, *extra_fc_inputs)
        outputs = outputs.reshape(activations[-1].size())

        for layer_index, up_layer in reversed(list(enumerate(self.up_layers))):
            w, h, d = outputs.size()[-3:]
            layer_activations = F.pad(activations[layer_index + 1], (0, 2) * 3)[
                ..., :w, :h, :d
            ]
            layer_inputs = torch.cat([layer_activations, outputs], dim=1)
            outputs = up_layer(layer_inputs)

        w, h, d = outputs.size()[-3:]
        layer_activations = F.pad(activations[0], (0, 2) * 3)[..., :w, :h, :d]
        final_layer_inputs = torch.cat([layer_activations, outputs], dim=1)

        w, h, d = inputs.size()[-3:]
        final_layer_inputs = final_layer_inputs[..., :w, :h, :d]

        return self.final_layer(final_layer_inputs)


class MbagConvolutionalModelConfig(MbagModelConfig, total=False):
    num_conv_1_layers: int
    """Number of 1x1x1 convolutions before the main backbone."""
    num_layers: int
    use_resnet: bool
    filter_size: int
    hidden_channels: int
    num_unet_layers: int
    """Number of layers to include in a UNet3d, if any."""
    unet_grow_factor: float
    unet_use_bn: bool


CONV_DEFAULT_CONFIG: MbagConvolutionalModelConfig = {
    "env_config": DEFAULT_CONFIG["env_config"],
    "embedding_size": DEFAULT_CONFIG["embedding_size"],
    "use_extra_features": DEFAULT_CONFIG["use_extra_features"],
    "mask_goal": DEFAULT_CONFIG["mask_goal"],
    "hidden_size": DEFAULT_CONFIG["hidden_size"],
    "num_action_layers": DEFAULT_CONFIG["num_action_layers"],
    "num_value_layers": DEFAULT_CONFIG["num_value_layers"],
    "fake_state": DEFAULT_CONFIG["fake_state"],
    "num_conv_1_layers": 0,
    "num_layers": 3,
    "use_resnet": False,
    "filter_size": 3,
    "hidden_channels": 32,
    "num_unet_layers": 0,
    "unet_grow_factor": 2.0,
    "unet_use_bn": False,
}


class MbagConvolutionalModel(MbagTorchModel):
    """
    Has an all-convolutional backbone and separate heads for each part of the
    autoregressive action distribution.
    """

    def __init__(
        self,
        obs_space: spaces.Space,
        action_space: spaces.Space,
        num_outputs: int,
        model_config,
        name,
        **kwargs,
    ):
        extra_config: MbagConvolutionalModelConfig = copy.deepcopy(CONV_DEFAULT_CONFIG)
        extra_config.update(cast(MbagConvolutionalModelConfig, kwargs))
        self.num_conv_1_layers = extra_config["num_conv_1_layers"]
        self.num_layers = extra_config["num_layers"]
        self.use_resnet = extra_config["use_resnet"]
        self.filter_size = extra_config["filter_size"]
        self.hidden_channels = extra_config["hidden_channels"]
        self.num_unet_layers = extra_config["num_unet_layers"]
        self.unet_grow_factor = extra_config["unet_grow_factor"]
        self.unet_use_bn = extra_config["unet_use_bn"]

        super().__init__(
            obs_space, action_space, num_outputs, model_config, name, **kwargs
        )

    def _construct_backbone(self, is_value_network=False) -> nn.Module:
        include_unet = not is_value_network
        backbone_layers: List[nn.Module] = []
        layer_index = 0
        num_layers = self.num_conv_1_layers + self.num_layers
        while layer_index < num_layers:
            if layer_index < self.num_conv_1_layers:
                filter_size = 1
            else:
                filter_size = self.filter_size
            if layer_index == 0:
                in_channels = self._get_in_channels()
            else:
                in_channels = self.hidden_channels

            if (
                self.use_resnet
                and in_channels == self.hidden_channels
                and layer_index + 2 < num_layers
            ):
                backbone_layers.append(ResidualBlock(channels=self.hidden_channels))
                layer_index += 2
            else:
                backbone_layers.append(
                    nn.Conv3d(
                        in_channels=in_channels,
                        out_channels=self.hidden_channels,
                        kernel_size=filter_size,
                        stride=1,
                        padding=(filter_size - 1) // 2,
                    )
                )
                backbone_layers.append(nn.LeakyReLU())
                layer_index += 1
        if include_unet and self.num_unet_layers > 0:
            self.unet = UNet3d(
                self.world_obs_space.shape[1],
                self.hidden_channels,
                self.hidden_channels,
                self.num_unet_layers,
                grow_factor=self.unet_grow_factor,
                use_bn=self.unet_use_bn,
            )
            backbone_layers.append(self.unet)
        else:
            backbone_layers = backbone_layers[:-1]  # Remove last ReLU.
        return nn.Sequential(*backbone_layers)


ModelCatalog.register_custom_model("mbag_convolutional_model", MbagConvolutionalModel)


class MbagRecurrentConvolutionalModelConfig(MbagConvolutionalModelConfig):
    pass


RECURRENT_CONV_DEFAULT_CONFIG: MbagRecurrentConvolutionalModelConfig = {
    **CONV_DEFAULT_CONFIG,  # type: ignore
}


class AddTimeDimRNN(nn.Module):
    _rnn_state: Tuple[torch.Tensor, torch.Tensor]
    _outputs: torch.Tensor
    _new_state: List[torch.Tensor]

    def __init__(self, rnn: nn.Module):
        super().__init__()
        self.rnn = rnn

    def set_state_seq_lens(self, rnn_state, seq_lens):
        if isinstance(seq_lens, np.ndarray):
            seq_lens = torch.Tensor(seq_lens).int()
        self._seq_lens = seq_lens
        self._rnn_state = rnn_state

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        max_seq_len = inputs.shape[0] // self._seq_lens.shape[0]
        input_shape = inputs.size()[1:]
        inputs = inputs.reshape(-1, max_seq_len, *input_shape)
        outputs, new_state = self.rnn(
            inputs,
            [self._rnn_state[0].unsqueeze(0), self._rnn_state[1].unsqueeze(0)],
        )
        self._outputs = outputs.reshape(-1, *input_shape)
        self._new_state = [new_state[0].squeeze(0), new_state[1].squeeze(0)]
        return self._outputs

    def get_outputs(self) -> torch.Tensor:
        return self._outputs

    def get_new_state(self):
        return self._new_state


class MbagRecurrentConvolutionalModel(MbagModel, nn.Module):
    _logits: torch.Tensor

    def __init__(
        self,
        obs_space: spaces.Space,
        action_space: spaces.Space,
        num_outputs: int,
        model_config,
        name,
        **kwargs,
    ):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        extra_config = RECURRENT_CONV_DEFAULT_CONFIG
        extra_config.update(kwargs)  # type: ignore

        self.conv_model = MbagConvolutionalModel(
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name=f"{name}.conv_model",
            **extra_config,
        )
        assert hasattr(self.conv_model, "unet")
        unet: UNet3d = self.conv_model.unet
        self.rnn_hidden_dim = unet.fc_layer_size

        self.lstm = nn.LSTM(self.rnn_hidden_dim, self.rnn_hidden_dim, batch_first=True)
        self.rnn = AddTimeDimRNN(self.lstm)
        unet.set_fc_layer(self.rnn)

        if self.model_config["vf_share_layers"]:
            value_layers: List[nn.Module] = []
            hidden_channels = extra_config["hidden_channels"]
            for layer_index in range(extra_config["num_value_layers"]):
                value_layers.append(
                    nn.Linear(
                        self.rnn_hidden_dim if layer_index == 0 else hidden_channels,
                        hidden_channels,
                    )
                )
                value_layers.append(nn.LeakyReLU())
            self.value_head = nn.Sequential(
                *value_layers,
                nn.Linear(
                    self.rnn_hidden_dim if len(value_layers) == 0 else hidden_channels,
                    1,
                    bias=True,
                ),
            )
        else:
            warnings.warn(
                "without vf_share_layers, the value function will not be recurrent"
            )

    def get_initial_state(self):
        # Place hidden states on same device as model.
        param = next(iter(self.lstm.parameters()))
        h = [
            param.new(1, self.rnn_hidden_dim).zero_().squeeze(0),
            param.new(1, self.rnn_hidden_dim).zero_().squeeze(0),
        ]
        return h

    def value_function(self):
        if self.model_config["vf_share_layers"]:
            return self.value_head(self.rnn.get_outputs()).squeeze(1)
        else:
            return self.conv_model.value_function()

    def forward(
        self,
        input_dict: Dict[str, torch.Tensor],
        state: List[torch.Tensor],
        seq_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        self.rnn.set_state_seq_lens(state, seq_lens)
        self._logits, _ = self.conv_model.forward(input_dict, state, seq_lens)
        new_state = self.rnn.get_new_state()

        return self._logits, new_state

    @property
    def logits(self) -> torch.Tensor:
        return self._logits

    def block_id_model(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.conv_model.block_id_model(inputs)


ModelCatalog.register_custom_model(
    "mbag_recurrent_convolutional_model", MbagRecurrentConvolutionalModel
)


class View(nn.Module):
    def __init__(self, *shape: int):
        super().__init__()
        self.shape = shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(*self.shape)


class Permute(nn.Module):
    def __init__(self, *dims: int):
        super().__init__()
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(*self.dims)


class SeparatedTransformerEncoder(nn.Module):
    def __init__(self, num_layers: int, d_model: int, nhead: int, dim_feedforward):
        super().__init__()

        self.layers: List[nn.TransformerEncoderLayer] = []
        for layer_index in range(num_layers):
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                batch_first=True,
            )
            self.add_module(f"layer_{layer_index}", layer)
            self.layers.append(layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input shape: (batch_size, channels, spatial_dim_1, spatial_dim_2...)
        n_spatial_dims = len(x.size()) - 2
        for layer_index, layer in enumerate(self.layers):
            spatial_dim = layer_index % n_spatial_dims
            permutation = (
                (0,)
                + tuple(
                    other_spatial_dim + 2
                    for other_spatial_dim in range(n_spatial_dims)
                    if other_spatial_dim != spatial_dim
                )
                + (spatial_dim + 2, 1)
            )
            inverse_permutation = tuple(
                permutation.index(dim) for dim in range(len(x.size()))
            )
            x_permuted = x.permute(*permutation)
            layer_input = x_permuted.flatten(end_dim=-3)
            layer_output = layer(layer_input)
            x = layer_output.reshape(x_permuted.size()).permute(*inverse_permutation)

        return x


class MbagTransformerModelConfig(MbagModelConfig, total=False):
    position_embedding_size: int
    num_layers: int
    num_heads: int
    use_separated_transformer: bool


TRANSFORMER_DEFAULT_CONFIG: MbagTransformerModelConfig = {
    "env_config": DEFAULT_CONFIG["env_config"],
    "embedding_size": DEFAULT_CONFIG["embedding_size"],
    "use_extra_features": DEFAULT_CONFIG["use_extra_features"],
    "mask_goal": DEFAULT_CONFIG["mask_goal"],
    "hidden_size": DEFAULT_CONFIG["hidden_size"],
    "num_action_layers": DEFAULT_CONFIG["num_action_layers"],
    "num_value_layers": DEFAULT_CONFIG["num_value_layers"],
    "fake_state": DEFAULT_CONFIG["fake_state"],
    "position_embedding_size": 12,
    "num_layers": 3,
    "num_heads": 2,
    "use_separated_transformer": False,
}


class MbagTransformerModel(MbagTorchModel):
    """
    Model which uses a transformer encoder as the backbone.
    """

    def __init__(
        self,
        obs_space: spaces.Space,
        action_space: spaces.Space,
        num_outputs: int,
        model_config,
        name,
        **kwargs,
    ):
        extra_config: MbagTransformerModelConfig = copy.deepcopy(
            TRANSFORMER_DEFAULT_CONFIG
        )
        extra_config.update(cast(MbagTransformerModelConfig, kwargs))
        self.position_embedding_size = extra_config["position_embedding_size"]
        self.num_layers = extra_config["num_layers"]
        self.num_heads = extra_config["num_heads"]
        self.use_separated_transformer = extra_config["use_separated_transformer"]

        super().__init__(
            obs_space, action_space, num_outputs, model_config, name, **kwargs
        )

        assert self._get_in_channels() <= self.hidden_size

        # Initialize positional embeddings along each dimension.
        self.position_embedding = nn.Parameter(
            torch.zeros(self.world_size + (self.position_embedding_size,))
        )
        dim_embedding_size = self.position_embedding_size // 6 * 2
        self.position_embedding.data[
            ..., :dim_embedding_size
        ] = self._get_position_embedding(
            self.position_embedding.size()[0],
            dim_embedding_size,
        )[
            :, None, None
        ]
        self.position_embedding.data[
            ..., dim_embedding_size : dim_embedding_size * 2
        ] = self._get_position_embedding(
            self.position_embedding.size()[1],
            dim_embedding_size,
        )[
            None, :, None
        ]
        self.position_embedding.data[
            ..., dim_embedding_size * 2 : dim_embedding_size * 3
        ] = self._get_position_embedding(
            self.position_embedding.size()[2],
            dim_embedding_size,
        )[
            None, None, :
        ]

    def _get_in_channels(self) -> int:
        return super()._get_in_channels() + self.position_embedding_size

    def _get_position_embedding(self, seq_len: int, size: int) -> torch.Tensor:
        """
        Get an initial positional embedding of shape (seq_len, size) by using
        the sin/cos embedding.
        """

        embedding = torch.zeros(seq_len, size)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, size, 2).float() * (-np.log(10000.0) / size)
        )
        embedding[:, 0::2] = torch.sin(position * div_term)
        embedding[:, 1::2] = torch.cos(position * div_term)
        return embedding

    def _pad_to_hidden_size(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pads the last dimension of the given tensor so that it is self.hidden_size.
        """

        return F.pad(x, [0, self.hidden_size - x.size()[-1]])

    def _construct_backbone(self, is_value_network=False) -> nn.Module:
        if self.use_separated_transformer:
            return SeparatedTransformerEncoder(
                num_layers=self.num_layers,
                d_model=self.hidden_size,
                nhead=self.num_heads,
                dim_feedforward=self.hidden_size,
            )
        else:
            return nn.Sequential(
                nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=self.hidden_size,
                        nhead=self.num_heads,
                        dim_feedforward=self.hidden_size,
                        batch_first=True,
                    ),
                    self.num_layers,
                ),
                View(-1, *self.world_size, self.hidden_size),
                Permute(0, 4, 1, 2, 3),
            )

    def _get_embedded_obs(
        self,
        world_obs: torch.Tensor,
        inventory_obs: torch.Tensor,
        timestep: torch.Tensor,
    ):
        embedded_obs = super()._get_embedded_obs(world_obs, inventory_obs, timestep)
        batch_size = embedded_obs.size()[0]
        embedded_obs = self._pad_to_hidden_size(
            torch.cat(
                [
                    embedded_obs.permute(0, 2, 3, 4, 1),
                    self.position_embedding[None].expand(batch_size, -1, -1, -1, -1),
                ],
                dim=4,
            )
        ).permute(0, 4, 1, 2, 3)

        if self.use_separated_transformer:
            return embedded_obs
        else:
            return embedded_obs.flatten(start_dim=2).transpose(1, 2)


ModelCatalog.register_custom_model("mbag_transformer_model", MbagTransformerModel)


class OtherAgentActionPredictorMixin(MbagTorchModel):
    def __init__(
        self,
        obs_space: spaces.Space,
        action_space: spaces.Space,
        num_outputs: int,
        model_config,
        name,
        **kwargs,
    ):
        super().__init__(
            obs_space, action_space, num_outputs, model_config, name, **kwargs
        )

        self.other_agent_action_prediction_head = self._construct_action_head()

    def predict_other_agent_action(self) -> torch.Tensor:
        logits: torch.Tensor = self.other_agent_action_prediction_head(
            self._backbone_out
        )
        flat_logits = MbagActionDistribution.to_flat_torch_logits(
            self.env_config, logits
        )
        return flat_logits


class MbagTransformerAlphaZeroModel(
    MbagTransformerModel, OtherAgentActionPredictorMixin
):
    pass


ModelCatalog.register_custom_model(
    "mbag_transformer_alpha_zero_model", MbagTransformerAlphaZeroModel
)
