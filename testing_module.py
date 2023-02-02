import os
import pickle
import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
import numpy as np
from utils.functions import signal_status
np.set_printoptions(suppress=True)

def tester(cfg, env, agent, logger):
    n_episodes = cfg.test.n_episodes
    max_t = cfg.test.max_t
    head = os.path.split(os.getcwd())[0]
    head = os.path.split(head)[0]
    agent.load(os.path.join(head, cfg.test.load))

    for i in range(1, n_episodes):
        obs = env.reset()
        reward = 0
        R = 0
        while env.t < max_t:
            action = agent.act(obs)
            obs, reward = env.step_local(action)
            R += reward

        agent.stop_episode()
        logger.info(f"Episode: {i}, reward: {R}")
        
    logger.info(f"Testing completed.")

    signal_status(cfg)
