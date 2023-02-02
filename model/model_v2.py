import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
import numpy as np
import chainerrl.misc.random_seed as chainerseed

def dqn(cfg, env):

    class QFunction(chainer.Chain):

        def __init__(self, obs_size, n_actions):
            super().__init__()
            with self.init_scope():
                self.l0 = L.Linear(obs_size, cfg.model.L1)
                self.l1 = L.Linear(cfg.model.L1, cfg.model.L2)
                self.l2 = L.Linear(cfg.model.L2, n_actions)

        def __call__(self, x):
            h = F.relu(self.l0(x))
            h = F.relu(self.l1(h))
            return chainerrl.action_value.DiscreteActionValue(self.l2(h))
    
    chainerseed.set_random_seed(cfg.model.cseed)
    n_actions = env.action.space()
    obs_size = len(env.observation)
        
    q_func = QFunction(obs_size, n_actions)

    optimizer = chainer.optimizers.Adam(eps=cfg.model.epsilon)
    optimizer.setup(q_func)

    gamma = cfg.model.gamma
    explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(
            cfg.model.decay_start, cfg.model.decay_end, cfg.model.decay_total_steps,
            lambda: np.random.randint(n_actions))
    replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=cfg.model.replay_buffer)
    phi = lambda x: x.astype(np.float32, copy=False)
    agent = chainerrl.agents.DoubleDQN(q_func, optimizer, 
        replay_buffer, gamma, explorer,
        minibatch_size=cfg.model.batchsize,
        replay_start_size=cfg.model.replay, update_interval=cfg.model.update_interval,
        target_update_interval=cfg.model.target_update, phi=phi)

    return agent