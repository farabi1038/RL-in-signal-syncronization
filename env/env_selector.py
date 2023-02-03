from .env_v1 import foothill_v1
from .env_v2 import foothill_v2

def env_selector(cfg, logger):
	if cfg.env.env.name == "foothill_v1":
		env = foothill_v1(cfg, logger)
		logger.info(f"action space: {env.action.space()}")
		logger.info(f"observation space: {len(env.state)}")
		env.seed(cfg.env.env.seed)

		if cfg.env.env.render:
			env.render()

		env.init()
		
		return env

	elif cfg.env.name == "foothill_v2":
		env = foothill_v2(cfg, logger)
		logger.info(f"action space: {env.action.space()}")
		logger.info(f"observation space: {len(env.state)}")
		env.seed(cfg.env.env.seed)

		if cfg.env.env.render:
			env.render()

		env.init()
		
		return env
