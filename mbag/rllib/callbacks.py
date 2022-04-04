from typing import Dict, Optional
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.typing import PolicyID

from mbag.environment.types import MbagInfoDict
from mbag.rllib.rllib_env import MbagMultiAgentEnv


class MbagCallbacks(DefaultCallbacks):
    def on_episode_start(
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: Episode,
        **kwargs,
    ) -> None:
        def update_env_global_timestep(env: MbagMultiAgentEnv):
            if worker.global_vars is not None:
                env.wrapped_env.update_global_timestep(worker.global_vars["timestep"])

        worker.foreach_env(update_env_global_timestep)

    def on_episode_step(
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        policies: Optional[Dict[PolicyID, Policy]] = None,
        episode: Episode,
        **kwargs,
    ) -> None:
        assert policies is not None
        for agent_id in episode.get_agents():
            policy_id = worker.policy_mapping_fn(agent_id, episode, worker)
            own_reward_key = f"{policy_id}/own_reward"
            if own_reward_key not in episode.custom_metrics:
                episode.custom_metrics[own_reward_key] = 0
            info_dict: MbagInfoDict = episode.last_info_for(agent_id)
            episode.custom_metrics[own_reward_key] += info_dict["own_reward"]

    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: Episode,
        env_index: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().on_episode_end(
            worker=worker,
            base_env=base_env,
            policies=policies,
            episode=episode,
            env_index=env_index,
            **kwargs,
        )

        info_dict: MbagInfoDict = episode.last_info_for("player_0")
        episode.custom_metrics["goal_similarity"] = info_dict["goal_similarity"]

        for agent_id in episode.get_agents():
            policy_id = worker.policy_mapping_fn(agent_id, episode, worker)
            info_dict = episode.last_info_for(agent_id)
            episode.custom_metrics[f"{policy_id}/own_reward_prop"] = info_dict[
                "own_reward_prop"
            ]
