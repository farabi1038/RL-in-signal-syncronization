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
from .explorer_modified import LinearDecayEpsilonGreedy

class QFunction(chainer.Chain):

    def __init__(self, obs_size, n_actions, n_hidden_channels=128):
        super().__init__()
        self.n_actions = len(n_actions)
        with self.init_scope():
            print(f"ob size {obs_size}, actuin {n_actions}")
            self.l0_0 = L.Linear(obs_size, n_hidden_channels)
            self.l0_1 = L.Linear(n_hidden_channels, n_hidden_channels)
            self.l0_2 = L.Linear(n_hidden_channels, n_actions[0])

            self.l1_0 = L.Linear(obs_size, n_hidden_channels)
            self.l1_1 = L.Linear(n_hidden_channels, n_hidden_channels)
            self.l1_2 = L.Linear(n_hidden_channels, n_actions[1])

            self.l2_0 = L.Linear(obs_size, n_hidden_channels)
            self.l2_1 = L.Linear(n_hidden_channels, n_hidden_channels)
            self.l2_2 = L.Linear(n_hidden_channels, n_actions[2])

            self.l3_0 = L.Linear(obs_size, n_hidden_channels)
            self.l3_1 = L.Linear(n_hidden_channels, n_hidden_channels)
            self.l3_2 = L.Linear(n_hidden_channels, n_actions[3])

            self.l4_0 = L.Linear(obs_size, n_hidden_channels)
            self.l4_1 = L.Linear(n_hidden_channels, n_hidden_channels)
            self.l4_2 = L.Linear(n_hidden_channels, n_actions[4])

            self.l5_0 = L.Linear(obs_size, n_hidden_channels)
            self.l5_1 = L.Linear(n_hidden_channels, n_hidden_channels)
            self.l5_2 = L.Linear(n_hidden_channels, n_actions[5])

            self.l6_0 = L.Linear(obs_size, n_hidden_channels)
            self.l6_1 = L.Linear(n_hidden_channels, n_hidden_channels)
            self.l6_2 = L.Linear(n_hidden_channels, n_actions[6])

            self.l7_0 = L.Linear(obs_size, n_hidden_channels)
            self.l7_1 = L.Linear(n_hidden_channels, n_hidden_channels)
            self.l7_2 = L.Linear(n_hidden_channels, n_actions[7])

    def __call__(self, x, test=False):

        h0 = F.relu(self.l0_0(x))
        h0 = F.relu(self.l0_1(h0))
        y0 = chainerrl.action_value.DiscreteActionValue(self.l0_2(h0))

        h1 = F.relu(self.l1_0(x))
        h1 = F.relu(self.l1_1(h1))
        y1 = chainerrl.action_value.DiscreteActionValue(self.l1_2(h1))

        h2 = F.relu(self.l2_0(x))
        h2 = F.relu(self.l2_1(h2))
        y2 = chainerrl.action_value.DiscreteActionValue(self.l2_2(h2))

        h3 = F.relu(self.l3_0(x))
        h3 = F.relu(self.l3_1(h3))
        y3 = chainerrl.action_value.DiscreteActionValue(self.l3_2(h3))

        h4 = F.relu(self.l4_0(x))
        h4 = F.relu(self.l4_1(h4))
        y4 = chainerrl.action_value.DiscreteActionValue(self.l4_2(h4))

        h5 = F.relu(self.l5_0(x))
        h5 = F.relu(self.l5_1(h5))
        y5 = chainerrl.action_value.DiscreteActionValue(self.l5_2(h5))

        h6 = F.relu(self.l6_0(x))
        h6 = F.relu(self.l6_1(h6))
        y6 = chainerrl.action_value.DiscreteActionValue(self.l6_2(h6))

        h7 = F.relu(self.l7_0(x))
        h7 = F.relu(self.l7_1(h7))
        y7 = chainerrl.action_value.DiscreteActionValue(self.l7_2(h7))
        return [y0, y1, y2, y3, y4, y5, y6, y7]

def dqn_multi(cfg, env):
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
        # q_func= q_functions.FCStateQFunctionWithDiscreteAction(
        #     obs_size, n_actions,
        #     n_hidden_channels=32, n_hidden_layers=2)
        q_func = QFunction(obs_size, n_actions, cfg.model.hidden_channels)
        # Use epsilon-greedy for exploration
        explorer = LinearDecayEpsilonGreedy(
            1.0, 0.1, cfg.model.lindecaysteps, {i:j.sample for (i,j) in zip(range(len(n_actions)),action_space)})


    opt = optimizers.Adam()
    opt.setup(q_func)

    rbuf_capacity = 5 * 10 ** 5
    if cfg.model.replay_type == "prioritized":
        betasteps = ((cfg.train.max_t * cfg.train.n_episodes) - cfg.model.replay_start_size) \
            // cfg.model.update_interval
        rbuf = replay_buffer.PrioritizedReplayBuffer(
            rbuf_capacity, betasteps=betasteps)
    elif cfg.model.replay_type == "normal":
        rbuf = replay_buffer.ReplayBuffer(rbuf_capacity)

    agent = DQN(q_func, opt, rbuf, gpu=-1, gamma=0.99,
                explorer=explorer, replay_start_size=cfg.model.replay_start_size,
                target_update_interval=cfg.model.target_update_interval,
                update_interval=cfg.model.update_interval,
                minibatch_size=cfg.model.batchsize,
                target_update_method='hard',
                soft_update_tau=1e-2,
                )

    return agent