import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
import numpy as np
import chainerrl.misc.random_seed as chainerseed
from chainerrl import q_functions
from chainerrl import replay_buffer
from chainerrl import explorers
from chainer import optimizers
# from chainerrl.agents.dqn import DQN
from gym import spaces
from .DQN_modified import DQN
from .Double_DQN_modified import DoubleDQN
from .explorer_modified import LinearDecayEpsilonGreedy

class QFunction(chainer.Chain):

    def __init__(self, obs_size, n_actions, cfg):
        super().__init__()
        self.n_actions = len(n_actions)
        winit = chainerrl.initializers.Orthogonal(1.)
        with self.init_scope():
            print(f"ob size {obs_size}, actuin {n_actions}")
            hidden_layers = []
            for a_len in range(self.n_actions):
                hidden_layers.append(L.Linear(obs_size, cfg.model.n_hidden_channels, initialW=winit))
                hidden_layers.append(L.Linear(cfg.model.n_hidden_channels, cfg.model.n_hidden_channels, initialW=winit))
                hidden_layers.append(L.Linear(cfg.model.n_hidden_channels, n_actions[a_len]))
            self.hidden_layers = chainer.ChainList(*hidden_layers)

    def __call__(self, x, test=False):

        action_list = []
        for p_len in range(self.n_actions):
            h = F.relu(self.hidden_layers[p_len*3](x))
            h = F.dropout(h, ratio=0.2)
            h = F.relu(self.hidden_layers[(p_len*3)+1](h))
            y = chainerrl.action_value.DiscreteActionValue(self.hidden_layers[(p_len*3)+2](h))
            action_list.append(y)

        return action_list

def dqn_multi_full_dropout(cfg, env):
    chainerseed.set_random_seed(cfg.model.cseed)
    action_space = env.action.space()
    obs_size = len(env.state)

    if isinstance(action_space, spaces.Box):
        action_size = action_space.low.size
        # Use NAF to apply DQN to continuous action spaces
        q_func = q_functions.FCQuadraticStateQFunction(
            obs_size, action_size,
            n_hidden_channels=128,
            n_hidden_layers=2,
            action_space=action_space)
        # Use the Ornstein-Uhlenbeck process for exploration
        ou_sigma = (action_space.high - action_space.low) * 0.2
        explorer = explorers.AdditiveOU(sigma=ou_sigma)
    else:
        n_actions = [i.n for i in action_space]
        q_func = QFunction(obs_size, n_actions, cfg)
        # Use epsilon-greedy for exploration
        explorer = LinearDecayEpsilonGreedy(
            1.0, 0.1, cfg.model.lindecaysteps, {i:j.sample for (i,j) in zip(range(len(n_actions)),action_space)})

    if cfg.model.opt_type == "adam":
        opt = optimizers.Adam()
    elif cfg.model.opt_type == "rmsprop":
        opt = optimizers.RMSpropGraves(
                lr=cfg.model.rms_lr, alpha=0.95, momentum=0.0, eps=1e-2)
    opt.setup(q_func)

    rbuf_capacity = 5 * 10 ** 5
    if cfg.model.replay_type == "prioritized":
        betasteps = ((cfg.train.max_t * cfg.train.n_episodes) - cfg.model.replay_start_size) \
            // cfg.model.update_interval
        rbuf = replay_buffer.PrioritizedReplayBuffer(
            rbuf_capacity, betasteps=betasteps)
    elif cfg.model.replay_type == "normal":
        rbuf = replay_buffer.ReplayBuffer(rbuf_capacity)

    agent = DQN(q_func, opt, rbuf, gpu=cfg.model.gpu, gamma=0.99,
                explorer=explorer, replay_start_size=cfg.model.replay_start_size,
                target_update_interval=cfg.model.target_update_interval,
                update_interval=cfg.model.update_interval,
                minibatch_size=cfg.model.batchsize,
                target_update_method='hard',
                soft_update_tau=1e-2,
                )

    return agent