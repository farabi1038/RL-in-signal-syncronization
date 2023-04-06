#from .env_v1 import foothill_v1
#from .env_v2 import foothill_v2
import gym
from sumo_rl import SumoEnvironment
def env_selector(cfg, logger):


	'''
	if cfg.env.name == "foothill_v1":
		env = foothill_v1(cfg, logger)
		logger.info(f"action space: {env.action.space()}")
		logger.info(f"observation space: {len(env.state)}")
		env.seed(cfg.env.seed)

		if cfg.env.render:
			env.render()

		env.init()
		
		return env

	elif cfg.env.name == "foothill_v2":
		env = foothill_v2(cfg, logger)
		logger.info(f"action space: {env.action.space()}")
		logger.info(f"observation space: {len(env.state)}")
		env.seed(cfg.env.seed)

		if cfg.env.render:
			env.render()

		env.init()
	'''	
	env = SumoEnvironment(net_file='/Users/ibnefarabishihab/Documents/GitHub/RL-in-signal-syncronization/sumo_files/foothill.net.xml',
                route_file='/Users/ibnefarabishihab/Documents/GitHub/RL-in-signal-syncronization/sumo_files/route.sample.xml',
                out_csv_name='path_to_output.csv',
                use_gui=True,
                num_seconds=100000)


	return env
