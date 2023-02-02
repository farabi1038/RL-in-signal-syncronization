import os, sys
import hydra
import logging

### import selectors and trainer
from env.env_selector import env_selector
from model.model_selector import model_selector
from training_module import trainer

### standard sumo code to ensure sumo_home is inside path
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

### setting up logger
logger = logging.getLogger(__name__)

### attach hydra to main function
@hydra.main(config_path="conf/config.yaml")
def main(cfg):
	# set config status to True (trigger specific if elses code blocks inside environment file)
	cfg.train.status = True
	logger.info(f"Training with the following config:\n{cfg.pretty()}")

	# set respective env and agent from selector, then pass them through trainer
	env = env_selector(cfg, logger)
	agent = model_selector(cfg, env)
	trainer(cfg, env, agent, logger)

if __name__ == "__main__":
	main()