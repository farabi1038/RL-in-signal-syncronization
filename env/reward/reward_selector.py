from .reward_v1 import reward_v1

def reward_selector(cfg):
	if cfg.reward.reward.name == "v1":
		return reward_v1
