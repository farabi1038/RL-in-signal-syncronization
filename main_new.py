import gymnasium as gym
import sumo_rl
import hydra
import logging
from omegaconf import DictConfig, OmegaConf

### import selectors and trainer
from env.env_selector import env_selector
from model.model_selector import model_selector
from training_module import trainer
logger = logging.getLogger(__name__)
env = gym.make('foothill_v1',
                net_file='/Users/ibnefarabishihab/Documents/GitHub/RL-in-signal-syncronization/sumo_files/foothill.net.xml',
                route_file='/Users/ibnefarabishihab/Documents/GitHub/RL-in-signal-syncronization/sumo_files/route.sample.xml',
                out_csv_name='path_to_output.csv',
                use_gui=True,
                num_seconds=100000)
obs, info = env.reset()
done = False
while not done:
    next_obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    done = terminated or truncated