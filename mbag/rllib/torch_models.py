from typing import Dict, List, Tuple, cast
import warnings
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F  # noqa: N812
from abc import ABC, abstractmethod
from typing_extensions import TypedDict
from gym import spaces
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

from mbag.environment.blocks import MinecraftBlocks
from mbag.environment.types import CURRENT_BLOCKS, GOAL_BLOCKS, PLAYER_LOCATIONS


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


class MbagConvolutionalModelConfig(TypedDict, total=False):
    embedding_size: int
    use_extra_features: bool
    """Use extra hand-designed features as input to the network."""
    mask_goal: bool
    """Remove goal information from observations before passing into the network."""
    num_conv_1_layers: int
    """Number of 1x1x1 convolutions before the main backbone."""
    num_layers: int
    use_resnet: bool
    filter_size: int
    hidden_channels: int
    num_block_id_layers: int
    """Number of extra layers for the block ID head."""
    num_unet_layers: int
    """Number of layers to include in a UNet3d, if any."""
    unet_grow_factor: float
    unet_use_bn: bool
    fake_state: bool
    """Whether to add fake state to this model so that it's treated as recurrent."""


CONV_DEFAULT_CONFIG: MbagConvolutionalModelConfig = {
    "embedding_size": 8,
    "use_extra_features": False,
    "mask_goal": False,
    "num_conv_1_layers": 0,
    "num_layers": 3,
    "use_resnet": False,
    "filter_size": 3,
    "hidden_channels": 32,
    "num_block_id_layers": 1,
    "num_unet_layers": 0,
    "unet_grow_factor": 2.0,
    "unet_use_bn": False,
    "fake_state": False,
}


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
            layer_inputs = torch.cat([activations[layer_index + 1], outputs], dim=1)
            outputs = up_layer(layer_inputs)
        final_layer_inputs = torch.cat([activations[0], outputs], dim=1)
        return self.final_layer(final_layer_inputs)


class MbagConvolutionalModel(MbagModel, nn.Module):
    """
    Has an all-convolutional backbone and separate heads for each part of the
    autoregressive action distribution.
    """

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
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        obs_space = obs_space.original_space
        assert isinstance(obs_space, spaces.Tuple)
        self.world_obs_space: spaces.Box = obs_space[0]

        assert isinstance(action_space, spaces.Tuple)
        self.action_type_space: spaces.Discrete = action_space[0]
        self.block_location_space: spaces.Discrete = action_space[1]
        self.block_id_space: spaces.Discrete = action_space[2]

        assert self.block_location_space.n == np.prod(self.world_obs_space.shape[-3:])

        extra_config = CONV_DEFAULT_CONFIG
        extra_config.update(cast(MbagConvolutionalModelConfig, kwargs))
        self.embedding_size = extra_config["embedding_size"]
        self.use_extra_features = extra_config["use_extra_features"]
        self.mask_goal = extra_config["mask_goal"]
        self.num_conv_1_layers = extra_config["num_conv_1_layers"]
        self.num_layers = extra_config["num_layers"]
        self.use_resnet = extra_config["use_resnet"]
        self.filter_size = extra_config["filter_size"]
        self.hidden_channels = extra_config["hidden_channels"]
        self.num_block_id_layers = extra_config["num_block_id_layers"]
        self.num_unet_layers = extra_config["num_unet_layers"]
        self.unet_grow_factor = extra_config["unet_grow_factor"]
        self.unet_use_bn = extra_config["unet_use_bn"]
        self.fake_state: bool = extra_config["fake_state"]

        # We have in-planes for current blocks, player locations, and
        # goal blocks if mask_goal is False.
        self.in_planes = 2 if self.mask_goal else 3  # TODO: update if we add more
        self.in_channels = self.in_planes * self.embedding_size
        if self.use_extra_features:
            self.in_channels += 1

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
            self.backbone = self._construct_backbone(include_unet=True)
        else:
            self.action_backbone = self._construct_backbone(include_unet=True)
            self.value_backbone = self._construct_backbone()

        block_id_in_channels = self.action_type_space.n + self.hidden_channels
        block_id_layers: List[nn.Module] = []
        for layer_index in range(self.num_block_id_layers):
            block_id_layers.append(
                nn.Linear(
                    block_id_in_channels if layer_index == 0 else self.hidden_channels,
                    self.block_id_space.n
                    if layer_index == self.num_block_id_layers - 1
                    else self.hidden_channels,
                )
            )
            if layer_index < self.num_block_id_layers - 1:
                block_id_layers.append(nn.LeakyReLU())
        self.block_id_head = nn.Sequential(*block_id_layers)

        self.value_head = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(self.action_type_space.n + self.hidden_channels, 1, bias=True),
        )

    def _construct_backbone(self, include_unet=False):
        backbone_layers: List[nn.Module] = []
        layer_index = 0
        num_layers = self.num_conv_1_layers + self.num_layers
        while layer_index < num_layers:
            if layer_index < self.num_conv_1_layers:
                filter_size = 1
            else:
                filter_size = self.filter_size
            if layer_index == 0:
                in_channels = self.in_channels
            else:
                in_channels = self.hidden_channels
            if layer_index == self.num_conv_1_layers + self.num_layers - 1 and (
                self.num_unet_layers == 0 or not include_unet
            ):
                out_channels = self.action_type_space.n + self.hidden_channels
            else:
                out_channels = self.hidden_channels

            if (
                self.use_resnet
                and in_channels == self.hidden_channels
                and out_channels == self.hidden_channels
                and layer_index + 2 < num_layers
            ):
                backbone_layers.append(ResidualBlock(channels=self.hidden_channels))
                layer_index += 2
            else:
                backbone_layers.append(
                    nn.Conv3d(
                        in_channels=in_channels,
                        out_channels=out_channels,
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
                self.action_type_space.n + self.hidden_channels,
                self.num_unet_layers,
                grow_factor=self.unet_grow_factor,
                use_bn=self.unet_use_bn,
            )
            backbone_layers.append(self.unet)
        else:
            backbone_layers = backbone_layers[:-1]  # Remove last ReLU.
        return nn.Sequential(*backbone_layers)

    def forward(self, input_dict, state, seq_lens):
        (self._world_obs,) = input_dict["obs"]

        # TODO: embed other block info?
        self._world_obs = self._world_obs.long()
        embedded_blocks = self.block_id_embedding(self._world_obs[:, CURRENT_BLOCKS])
        embedded_obs_pieces = [embedded_blocks]
        if not self.mask_goal:
            embedded_goal_blocks = self.block_id_embedding(
                self._world_obs[:, GOAL_BLOCKS]
            )
            embedded_obs_pieces.append(embedded_goal_blocks)
        embedded_player_locations = self.player_id_embedding(
            self._world_obs[:, PLAYER_LOCATIONS]
        )
        embedded_obs_pieces.append(embedded_player_locations)
        if self.use_extra_features:
            # Feature for if goal block is the same as the current block at each
            # location.
            embedded_obs_pieces.append(
                (self._world_obs[:, 0] == self._world_obs[:, 2]).float()[..., None]
            )
        embedded_obs = torch.cat(embedded_obs_pieces, dim=-1)

        self._embedded_obs = embedded_obs.permute(0, 4, 1, 2, 3)

        with torch.autocast(self._device.type):
            if self.vf_share_layers:
                self._backbone_out = self.backbone(self._embedded_obs)
                self._backbone_out_shape = self._backbone_out.size()[1:]
                self._logits = self._backbone_out.flatten(start_dim=1)
            else:
                backbone_out = self.action_backbone(self._embedded_obs)
                self._backbone_out_shape = backbone_out.size()[1:]
                self._logits = backbone_out.flatten(start_dim=1)
        return self._logits.float(), state

    @property
    def logits(self) -> torch.Tensor:
        return self._logits

    @property
    def _device(self) -> torch.device:
        return self._embedded_obs.device

    def block_id_model(self, head_input: torch.Tensor) -> torch.Tensor:
        with torch.autocast(self._device.type):
            block_id_logits: torch.Tensor = self.block_id_head(head_input)
        return block_id_logits.float()

    def value_function(self):
        with torch.autocast(self._device.type):
            if self.vf_share_layers:
                vf = self.value_head(self._backbone_out).squeeze(1)
            else:
                vf = self.value_head(self.value_backbone(self._embedded_obs)).squeeze(1)
        return vf.float()

    def get_initial_state(self):
        if self.fake_state:
            return [np.zeros(1)]
        else:
            return super().get_initial_state()


ModelCatalog.register_custom_model("mbag_convolutional_model", MbagConvolutionalModel)


class MbagRecurrentConvolutionalModelConfig(MbagConvolutionalModelConfig):
    num_value_layers: int


RECURRENT_CONV_DEFAULT_CONFIG: MbagRecurrentConvolutionalModelConfig = {
    **CONV_DEFAULT_CONFIG,  # type: ignore
    "num_value_layers": 0,
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


class MbagTransformerModelConfig(TypedDict, total=False):
    embedding_size: int
    position_embedding_size: int
    num_layers: int
    num_heads: int
    hidden_size: int
    num_block_id_layers: int


TRANSFORMER_DEFAULT_CONFIG: MbagTransformerModelConfig = {
    "embedding_size": 10,
    "position_embedding_size": 12,
    "num_layers": 3,
    "num_heads": 2,
    "hidden_size": 32,
    "num_block_id_layers": 2,
}


class MbagTransformerModel(MbagModel, nn.Module):
    """
    Concatenates all observation parts into one long sequence, with different encodings
    for each part, and then applies a transformer to the result. There are special
    positions in the sequence for the action type output and the block ID output.
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
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        obs_space = obs_space.original_space
        assert isinstance(obs_space, spaces.Tuple)
        self.world_obs_space: spaces.Box = obs_space[0]
        self.world_size = self.world_obs_space.shape[-3:]

        assert isinstance(action_space, spaces.Tuple)
        self.action_type_space: spaces.Discrete = action_space[0]
        self.block_location_space: spaces.Discrete = action_space[1]
        self.block_id_space: spaces.Discrete = action_space[2]

        assert self.block_location_space.n == np.prod(self.world_size)

        extra_config = TRANSFORMER_DEFAULT_CONFIG
        extra_config.update(cast(MbagTransformerModelConfig, kwargs))
        self.embedding_size = extra_config["embedding_size"]
        self.position_embedding_size = extra_config["position_embedding_size"]
        self.num_layers = extra_config["num_layers"]
        self.num_heads = extra_config["num_heads"]
        self.hidden_size = extra_config["hidden_size"]
        self.num_block_id_layers = extra_config["num_block_id_layers"]
        assert (
            self.embedding_size * 2 + self.position_embedding_size <= self.hidden_size
        )

        self.block_id_embedding = nn.Embedding(
            num_embeddings=len(MinecraftBlocks.ID2NAME),
            embedding_dim=self.embedding_size,
        )
        self.position_embedding = nn.Parameter(
            torch.zeros(self.world_size + (self.position_embedding_size,))
        )

        # Special embeddings for the sequence elements for value function, action_type,
        # block_id, and block_location outputs.
        self.vf_embedding = nn.Parameter(torch.zeros((1, self.position_embedding_size)))

        # Initialize positional embeddings along each dimension.
        dim_embedding_size = self.position_embedding_size // 3
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
            self.position_embedding.size()[0],
            dim_embedding_size,
        )[
            None, :, None
        ]
        self.position_embedding.data[
            ..., dim_embedding_size * 2 : dim_embedding_size * 3
        ] = self._get_position_embedding(
            self.position_embedding.size()[0],
            dim_embedding_size,
        )[
            None, None, :
        ]

        self.vf_embedding.data.normal_()

        self.vf_share_layers: bool = model_config["vf_share_layers"]
        if not self.vf_share_layers:
            raise ValueError(
                "MbagTransformerModel does not support vf_share_layers=False"
            )

        self.backbone_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.hidden_size,
                nhead=self.num_heads,
                dim_feedforward=self.hidden_size,
                batch_first=True,
            ),
            self.num_layers,
        )

        self.action_type_location_head = nn.Linear(
            self.hidden_size, self.action_type_space.n + self.hidden_size, bias=True
        )

        block_id_in_size = self.action_type_space.n + self.hidden_size
        block_id_layers: List[nn.Module] = []
        for layer_index in range(self.num_block_id_layers):
            block_id_layers.append(
                nn.Linear(
                    block_id_in_size if layer_index == 0 else self.hidden_size,
                    self.block_id_space.n
                    if layer_index == self.num_block_id_layers - 1
                    else self.hidden_channels,
                )
            )
            if layer_index < self.num_block_id_layers - 1:
                block_id_layers.append(nn.LeakyReLU())
        self.block_id_head = nn.Sequential(*block_id_layers)

        self.value_head = nn.Linear(self.hidden_size, 1, bias=True)

    def _get_position_embedding(self, seq_len: int, size: int) -> torch.Tensor:
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

    def forward(self, input_dict, state, seq_lens):
        (self._world_obs,) = input_dict["obs"]
        batch_size = self._world_obs.size()[0]

        # TODO: embed other block info?
        self._world_obs = self._world_obs.long()
        embedded_blocks = self.block_id_embedding(self._world_obs[:, 0])
        embedded_goal_blocks = self.block_id_embedding(self._world_obs[:, 2])

        embedded_world_obs = torch.cat(
            [
                self.position_embedding[None].expand(batch_size, -1, -1, -1, -1),
                embedded_blocks,
                embedded_goal_blocks,
            ],
            dim=-1,
        )
        flattened_world_obs = (
            embedded_world_obs.permute(0, 4, 1, 2, 3)
            .flatten(start_dim=2)
            .transpose(1, 2)
        )
        encoder_input = torch.cat(
            [
                self._pad_to_hidden_size(
                    self.vf_embedding[None].expand(batch_size, -1, -1)
                ),
                self._pad_to_hidden_size(flattened_world_obs),
            ],
            dim=1,
        )

        backbone_out = self.backbone_encoder(encoder_input)
        self._value_out = backbone_out[:, 0]
        backbone_out = backbone_out[:, 1:]

        dist_inputs = (
            self.action_type_location_head(backbone_out)
            .transpose(1, 2)
            .reshape(
                batch_size,
                self.action_type_space.n + self.hidden_size,
                *self.world_size,
            )
        )

        return dist_inputs, []

    def block_id_model(self, head_input: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.block_id_head(head_input))

    def value_function(self):
        return self.value_head(self._value_out)[:, 0]


ModelCatalog.register_custom_model("mbag_transformer_model", MbagTransformerModel)
