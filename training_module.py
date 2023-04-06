import os
import pickle
import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
import numpy as np
import traci

from tqdm import trange
from stable_baselines3.common.evaluation import evaluate_policy
import shutil
import subprocess
from stable_baselines3.dqn.dqn import DQN

def trainer(cfg, env, agent, logger):
    '''
    n_episodes = cfg.train.train.n_episodes
    max_t = cfg.train.train.max_t
    best_agent = -999999999999

    dic = {}
    for i in range(1, n_episodes):
        obs = env.reset()
        reward = 0
        R = 0
        while env.t < max_t:
            action = agent.act_and_train(obs, reward)
            obs, reward = env.step_local(action)
            R += reward

        agent.stop_episode_and_train(obs, reward)
        logger.info(f"Episode: {i}, reward: {R}, {agent.get_statistics()}")
        
        dic[i] = {}
        dic[i]['episode'] = i
        dic[i]['reward'] = R
        dic[i]['average_q'] = agent.get_statistics()[0][1]
        dic[i]['average_loss'] = agent.get_statistics()[1][1]
        dic[i]['n_updates'] = agent.get_statistics()[2][1]

        if i % 50 == 0:
            eval_obs = env.reset()
            eval_reward = 0
            eval_R = 0
            while env.t < max_t:
                eval_action = agent.act(eval_obs)
                eval_obs, eval_reward = env.step_local(eval_action)
                eval_R += eval_reward
            prev_agent = eval_R
            logger.info(f" ")
            logger.info(f"Evaluation phase")
            if prev_agent > best_agent:
                logger.info(f"Latest agent is the best agent at iteration {i}.")
                logger.info(f"Current agent: {prev_agent} > best agent: {best_agent}")
                best_agent = prev_agent
            else:
                logger.info(f"Latest agent is not the best agent at iteration {i}.")
                logger.info(f"Current agent: {prev_agent} <= best agent: {best_agent}")
            logger.info(f"Resume training")
            logger.info(f" ")
            agent.save(os.path.join(os.getcwd(), f"{i}_agent"))

    with open('dic.pkl', 'wb') as handle:
        pickle.dump(dic, handle)
    agent.save(os.path.join(os.getcwd(), "latest_agent"))
    logger.info(f"Training completed.")
    traci.close()
    '''
    model = DQN(
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
    model.learn(total_timesteps=100000)
