# from .model_v1 import ppo_v1
# from .model_v3 import dqn
# from .model_v4 import dqn_multi
# from .model_v5 import dqn_multi_full
# from .model_v6 import dqn_multi_full_split
# from .model_v7 import ddqn_multi_full
# from .model_v8 import dqn_multi_full_dropout


import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.dqn.dqn import DQN

def model_selector(cfg, env):
	if cfg.model.model.name == "ppo_v1":
		#return ppo_v1(cfg, env)
		return  DQN(
        env=env,
        policy="MlpPolicy",
        learning_rate=0.001,
        learning_starts=0,
        train_freq=1,
        target_update_interval=500,
        exploration_initial_eps=0.05,
        exploration_final_eps=0.01,
        verbose=1,
    )
	elif cfg.model.model.name == "dqn_v1":
		return dqn(cfg, env)
	elif cfg.model.model.name == "multihead_dqn":
		return dqn_multi(cfg, env)
	elif cfg.model.model.name == "multihead_dqn_full":
		return dqn_multi_full(cfg, env)
	elif cfg.model.model.name == "multihead_dqn_full_split":
		return dqn_multi_full_split(cfg, env)
	elif cfg.model.model.name == "multihead_ddqn_full":
		return ddqn_multi_full(cfg, env)
	elif cfg.model.model.name == "multihead_dqn_full_dropout":
		return dqn_multi_full_dropout(cfg, env)