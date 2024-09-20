import copy
import logging
import traceback
from typing import List, Optional, Tuple, Type

from mbag.agents.mbag_agent import MbagAgent
from mbag.environment.actions import MbagActionTuple
from mbag.environment.mbag_env import MbagConfigDict, MbagEnv
from mbag.environment.types import MbagInfoDict

from .episode import MbagEpisode

logger = logging.getLogger(__name__)

MbagAgentConfig = Tuple[Type[MbagAgent], dict]
"""
An MbagAgent subclass together with the agent config for that agent.
"""


EpisodeInfo = MbagEpisode  # For backwards compatibility.


class MbagEvaluator(object):
    """
    Used to evaluate a (set of) MBAG agent(s).
    """

    def __init__(
        self,
        env_config: MbagConfigDict,
        agent_configs: List[MbagAgentConfig],
        *,
        force_get_set_state=False,
        return_on_exception=False,
    ):
        if len(agent_configs) != env_config["num_players"]:
            raise ValueError(
                f"not enough agent_configs (expected {env_config['num_players']})"
            )

        self.env = MbagEnv(env_config)
        env_config = self.env.config
        self.agents = [
            agent_class(agent_config, env_config)
            for agent_class, agent_config in agent_configs
        ]
        self.force_get_set_state = force_get_set_state
        self.return_on_exception = return_on_exception

    def rollout(self, *, agent_seed: Optional[int] = None) -> MbagEpisode:
        """
        Run a single episode.
        """

        for agent in self.agents:
            agent.reset(seed=agent_seed)
        all_obs, all_infos = self.env.reset()
        previous_infos = all_infos
        done = False
        timestep = 0
        if self.force_get_set_state:
            agent_states = [agent.get_state() for agent in self.agents]

        reward_history = []
        obs_history = [all_obs]
        info_history: List[List[MbagInfoDict]] = []

        try:
            while not done:
                if self.force_get_set_state:
                    for agent, state in zip(self.agents, agent_states):
                        agent.reset()
                        agent.set_state(state)

                env_state = self.env.get_state()

                all_actions: List[MbagActionTuple] = []
                for agent_index, agent in enumerate(self.agents):
                    previous_info = previous_infos[agent_index]
                    obs = all_obs[agent_index]
                    action = agent.get_action_with_info_and_env_state(
                        obs, previous_info, env_state
                    )
                    all_actions.append(action)

                all_obs, all_rewards, all_done, all_infos = self.env.step(all_actions)
                done = all_done[0]
                reward_history.append(all_rewards[0])
                obs_history.append(all_obs)
                info_history.append(all_infos)
                timestep += 1

                if self.force_get_set_state:
                    agent_states = [agent.get_state() for agent in self.agents]
                previous_infos = all_infos
        except (Exception, KeyboardInterrupt) as exception:
            if self.return_on_exception:
                logger.error(exception)
                traceback.print_exc()
            else:
                raise exception

        if self.env.config["malmo"]["use_malmo"]:
            self.env.malmo_interface.end_episode()

        episode_info = MbagEpisode(
            env_config=copy.deepcopy(self.env.config),
            reward_history=reward_history,
            cumulative_reward=sum(reward_history),
            length=timestep,
            last_obs=all_obs,
            last_infos=all_infos,
            obs_history=obs_history,
            info_history=info_history,
        )
        return episode_info
