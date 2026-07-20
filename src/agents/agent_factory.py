from copy import deepcopy

from agents.las_lyapunov_agent import LAS_LyapunovAgent
from agents.las_td3_agent import LAS_TD3Agent
from agents.lqr_agent import LQRAgent
from agents.lyapunov_agent import LyapunovAgent
from agents.td3_agent import TD3Agent


class AgentFactory:
    @staticmethod
    def create_agent(config, env=None):
        """Create an isolated agent instance from a configuration mapping."""
        config = deepcopy(config)
        if env is not None:
            config["state_space"] = env.observation_space
            config["action_space"] = env.action_space

        configured_type = config.get("agent_str")
        if not configured_type:
            raise ValueError("Configuration requires agent_str")
        agent_type = str(configured_type).upper()
        constructors = {
            "LQR": LQRAgent,
            "LYAPUNOV-AC": LyapunovAgent,
            "TD3": TD3Agent,
            "LAS_TD3": LAS_TD3Agent,
            "LDP": LAS_LyapunovAgent,
        }
        try:
            constructor = constructors[agent_type]
        except KeyError as exc:
            raise ValueError(f"Unknown agent type: {agent_type}") from exc
        return constructor(config)
