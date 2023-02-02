import os
import pickle
import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
import numpy as np
import pandas as pd
from utils.functions import signal_status
np.set_printoptions(suppress=True)

def benchmark(cfg, env, agent, logger):
    n_episodes = cfg.benchmark.n_episodes
    max_t = cfg.benchmark.max_t

    for i in range(1, n_episodes):
        obs = env.reset()
        reward = 0
        R = 0
        while env.t < max_t:
            obs, reward = env.step_local(0)
            R += reward

        logger.info(f"Episode: {i}, reward: {R}")
        
    logger.info(f"Benchmark completed.")

    signal_status(cfg)
