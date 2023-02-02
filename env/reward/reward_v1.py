import numpy as np

def reward_v1(state, speeds, cfg, normalize=False):
	resolution = cfg.env.local_resolution
	if cfg.reward.slowness:
		if cfg.reward.normalize:
			return -speeds / resolution
		else:
			return -speeds
	elif state.edge_full_side:
		total = (state.main * .8) + (state.side * .2)
		if cfg.reward.normalize:
			return np.sum((total * -1) / resolution)
		else:
			return np.sum(total * -1)
	else:
		if cfg.reward.normalize:
			return np.sum(np.array(state.temp).astype(np.float32) * -1) / resolution
		else:
			return np.sum(np.array(state.temp).astype(np.float32) * -1)

