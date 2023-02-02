import os, sys
import hydra
import logging

from env.env_selector import env_selector
from model.model_selector import model_selector
from benchmark_module import benchmark

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

logger = logging.getLogger(__name__)

@hydra.main(config_path="conf/config.yaml")
def main(cfg):
	cfg.benchmark.status = True
	logger.info(f"Benchmark with the following config:\n{cfg.pretty()}")

	env = env_selector(cfg, logger)
	agent = model_selector(cfg, env)
	benchmark(cfg, env, agent, logger)

if __name__ == "__main__":
	main()