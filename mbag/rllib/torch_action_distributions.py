from typing import Any, Optional, cast

import torch
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.torch.torch_action_dist import (
    ActionDistribution,
    TorchCategorical,
)
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import TensorType
from torch.distributions import Categorical

from mbag.agents.action_distributions import MbagActionDistribution
from mbag.environment.actions import MbagAction

from .torch_models import MbagTorchModel


def kl_categorical_categorical_no_inf(p: Categorical, q: Categorical) -> torch.Tensor:
    t: torch.Tensor = p.probs * (p.logits - q.logits)
    # t[(q.probs == 0).expand_as(t)] = inf
    t[(p.probs == 0).expand_as(t)] = 0
    return t.sum(-1)


class TorchCategoricalNoInf(TorchCategorical):
    def entropy(self) -> TensorType:
        t: torch.Tensor = self.dist.probs * self.dist.logits
        t[(self.dist.probs == 0).expand_as(t)] = 0
        return -t.sum(-1)

    def kl(self, other: ActionDistribution) -> TensorType:
        assert isinstance(other, TorchCategorical)
        return kl_categorical_categorical_no_inf(self.dist, other.dist)


ModelCatalog.register_custom_action_dist("categorical_no_inf", TorchCategoricalNoInf)


class MbagBilevelCategorical(TorchCategoricalNoInf):
    def __init__(self, inputs, model: TorchModelV2):
        super().__init__(inputs, model=model)
        self._action_mapping: Optional[torch.Tensor] = None

        if isinstance(self.model, MbagTorchModel) and inputs is self.model.logits:
            cast(Any, self.model).action_dist = self

    def _action_type_probs(self) -> torch.Tensor:
        assert self._action_mapping is not None
        batch_size = self.dist.probs.size()[0]
        action_type_probs = torch.zeros(
            (batch_size, MbagAction.NUM_ACTION_TYPES),
            device=self.dist.probs.device,
            dtype=self.dist.probs.dtype,
        )
        action_type_probs.index_add_(1, self._action_mapping[:, 0], self.dist.probs)
        return action_type_probs

    def entropy(self):
        assert hasattr(self.model, "env_config")
        if self._action_mapping is None:
            self._action_mapping = torch.from_numpy(
                MbagActionDistribution.get_action_mapping(self.model.env_config)
            ).to(self.dist.probs.device)
            self._action_types = MbagActionDistribution.get_valid_action_types(
                self.model.env_config
            )

        action_type_probs = self._action_type_probs()
        self._action_type_entropy = Categorical(probs=action_type_probs).entropy()

        entropy = self._action_type_entropy

        for action_type in self._action_types:
            action_type_prob = action_type_probs[:, action_type]
            sub_distribution_logits = self.dist.logits[
                :, self._action_mapping[:, 0] == action_type
            ]
            sub_distribution_entropy = Categorical(
                logits=sub_distribution_logits
            ).entropy()
            entropy = entropy + action_type_prob.detach() * sub_distribution_entropy

        return entropy

    def action_type_entropy(self):
        return self._action_type_entropy


ModelCatalog.register_custom_action_dist(
    "mbag_bilevel_categorical", MbagBilevelCategorical
)
