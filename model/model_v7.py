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
from .Double_DQN_modified import DoubleDQN
from .explorer_modified import LinearDecayEpsilonGreedy

class QFunction(chainer.Chain):

    def __init__(self, obs_size, n_actions, n_hidden_channels=128):
        super().__init__()
        self.n_actions = len(n_actions)
        winit = chainerrl.initializers.Orthogonal(1.)
        with self.init_scope():
            print(f"ob size {obs_size}, actuin {n_actions}")
            self.l0_0 = L.Linear(obs_size, n_hidden_channels, initialW=winit)
            self.l0_1 = L.Linear(n_hidden_channels, n_hidden_channels, initialW=winit)
            self.l0_2 = L.Linear(n_hidden_channels, n_actions[0])

            self.l1_0 = L.Linear(obs_size, n_hidden_channels, initialW=winit)
            self.l1_1 = L.Linear(n_hidden_channels, n_hidden_channels, initialW=winit)
            self.l1_2 = L.Linear(n_hidden_channels, n_actions[1])

            self.l2_0 = L.Linear(obs_size, n_hidden_channels, initialW=winit)
            self.l2_1 = L.Linear(n_hidden_channels, n_hidden_channels, initialW=winit)
            self.l2_2 = L.Linear(n_hidden_channels, n_actions[2])

            self.l3_0 = L.Linear(obs_size, n_hidden_channels, initialW=winit)
            self.l3_1 = L.Linear(n_hidden_channels, n_hidden_channels, initialW=winit)
            self.l3_2 = L.Linear(n_hidden_channels, n_actions[3])

            self.l4_0 = L.Linear(obs_size, n_hidden_channels, initialW=winit)
            self.l4_1 = L.Linear(n_hidden_channels, n_hidden_channels, initialW=winit)
            self.l4_2 = L.Linear(n_hidden_channels, n_actions[4])

            self.l5_0 = L.Linear(obs_size, n_hidden_channels, initialW=winit)
            self.l5_1 = L.Linear(n_hidden_channels, n_hidden_channels, initialW=winit)
            self.l5_2 = L.Linear(n_hidden_channels, n_actions[5])

            self.l6_0 = L.Linear(obs_size, n_hidden_channels, initialW=winit)
            self.l6_1 = L.Linear(n_hidden_channels, n_hidden_channels, initialW=winit)
            self.l6_2 = L.Linear(n_hidden_channels, n_actions[6])

            self.l7_0 = L.Linear(obs_size, n_hidden_channels, initialW=winit)
            self.l7_1 = L.Linear(n_hidden_channels, n_hidden_channels, initialW=winit)
            self.l7_2 = L.Linear(n_hidden_channels, n_actions[7])

            self.l8_0 = L.Linear(obs_size, n_hidden_channels, initialW=winit)
            self.l8_1 = L.Linear(n_hidden_channels, n_hidden_channels, initialW=winit)
            self.l8_2 = L.Linear(n_hidden_channels, n_actions[8])

            self.l9_0 = L.Linear(obs_size, n_hidden_channels, initialW=winit)
            self.l9_1 = L.Linear(n_hidden_channels, n_hidden_channels, initialW=winit)
            self.l9_2 = L.Linear(n_hidden_channels, n_actions[9])

            self.l10_0 = L.Linear(obs_size, n_hidden_channels, initialW=winit)
            self.l10_1 = L.Linear(n_hidden_channels, n_hidden_channels, initialW=winit)
            self.l10_2 = L.Linear(n_hidden_channels, n_actions[10])

            self.l11_0 = L.Linear(obs_size, n_hidden_channels, initialW=winit)
            self.l11_1 = L.Linear(n_hidden_channels, n_hidden_channels, initialW=winit)
            self.l11_2 = L.Linear(n_hidden_channels, n_actions[11])

            self.l12_0 = L.Linear(obs_size, n_hidden_channels, initialW=winit)
            self.l12_1 = L.Linear(n_hidden_channels, n_hidden_channels, initialW=winit)
            self.l12_2 = L.Linear(n_hidden_channels, n_actions[12])

            self.l13_0 = L.Linear(obs_size, n_hidden_channels, initialW=winit)
            self.l13_1 = L.Linear(n_hidden_channels, n_hidden_channels, initialW=winit)
            self.l13_2 = L.Linear(n_hidden_channels, n_actions[13])

            self.l14_0 = L.Linear(obs_size, n_hidden_channels, initialW=winit)
            self.l14_1 = L.Linear(n_hidden_channels, n_hidden_channels, initialW=winit)
            self.l14_2 = L.Linear(n_hidden_channels, n_actions[14])

            self.l15_0 = L.Linear(obs_size, n_hidden_channels, initialW=winit)
            self.l15_1 = L.Linear(n_hidden_channels, n_hidden_channels, initialW=winit)
            self.l15_2 = L.Linear(n_hidden_channels, n_actions[15])

            self.l16_0 = L.Linear(obs_size, n_hidden_channels, initialW=winit)
            self.l16_1 = L.Linear(n_hidden_channels, n_hidden_channels, initialW=winit)
            self.l16_2 = L.Linear(n_hidden_channels, n_actions[16])

            self.l17_0 = L.Linear(obs_size, n_hidden_channels, initialW=winit)
            self.l17_1 = L.Linear(n_hidden_channels, n_hidden_channels, initialW=winit)
            self.l17_2 = L.Linear(n_hidden_channels, n_actions[17])

            self.l18_0 = L.Linear(obs_size, n_hidden_channels, initialW=winit)
            self.l18_1 = L.Linear(n_hidden_channels, n_hidden_channels, initialW=winit)
            self.l18_2 = L.Linear(n_hidden_channels, n_actions[18])

            self.l19_0 = L.Linear(obs_size, n_hidden_channels, initialW=winit)
            self.l19_1 = L.Linear(n_hidden_channels, n_hidden_channels, initialW=winit)
            self.l19_2 = L.Linear(n_hidden_channels, n_actions[19])

            self.l20_0 = L.Linear(obs_size, n_hidden_channels, initialW=winit)
            self.l20_1 = L.Linear(n_hidden_channels, n_hidden_channels, initialW=winit)
            self.l20_2 = L.Linear(n_hidden_channels, n_actions[20])

            self.l21_0 = L.Linear(obs_size, n_hidden_channels, initialW=winit)
            self.l21_1 = L.Linear(n_hidden_channels, n_hidden_channels, initialW=winit)
            self.l21_2 = L.Linear(n_hidden_channels, n_actions[21])

            self.l22_0 = L.Linear(obs_size, n_hidden_channels, initialW=winit)
            self.l22_1 = L.Linear(n_hidden_channels, n_hidden_channels, initialW=winit)
            self.l22_2 = L.Linear(n_hidden_channels, n_actions[22])

            self.l23_0 = L.Linear(obs_size, n_hidden_channels, initialW=winit)
            self.l23_1 = L.Linear(n_hidden_channels, n_hidden_channels, initialW=winit)
            self.l23_2 = L.Linear(n_hidden_channels, n_actions[23])

            self.l24_0 = L.Linear(obs_size, n_hidden_channels, initialW=winit)
            self.l24_1 = L.Linear(n_hidden_channels, n_hidden_channels, initialW=winit)
            self.l24_2 = L.Linear(n_hidden_channels, n_actions[24])

            self.l25_0 = L.Linear(obs_size, n_hidden_channels, initialW=winit)
            self.l25_1 = L.Linear(n_hidden_channels, n_hidden_channels, initialW=winit)
            self.l25_2 = L.Linear(n_hidden_channels, n_actions[25])

            self.l26_0 = L.Linear(obs_size, n_hidden_channels, initialW=winit)
            self.l26_1 = L.Linear(n_hidden_channels, n_hidden_channels, initialW=winit)
            self.l26_2 = L.Linear(n_hidden_channels, n_actions[26])

            self.l27_0 = L.Linear(obs_size, n_hidden_channels, initialW=winit)
            self.l27_1 = L.Linear(n_hidden_channels, n_hidden_channels, initialW=winit)
            self.l27_2 = L.Linear(n_hidden_channels, n_actions[27])

            self.l28_0 = L.Linear(obs_size, n_hidden_channels, initialW=winit)
            self.l28_1 = L.Linear(n_hidden_channels, n_hidden_channels, initialW=winit)
            self.l28_2 = L.Linear(n_hidden_channels, n_actions[28])

            self.l29_0 = L.Linear(obs_size, n_hidden_channels, initialW=winit)
            self.l29_1 = L.Linear(n_hidden_channels, n_hidden_channels, initialW=winit)
            self.l29_2 = L.Linear(n_hidden_channels, n_actions[29])

            self.l30_0 = L.Linear(obs_size, n_hidden_channels, initialW=winit)
            self.l30_1 = L.Linear(n_hidden_channels, n_hidden_channels, initialW=winit)
            self.l30_2 = L.Linear(n_hidden_channels, n_actions[30])

            self.l31_0 = L.Linear(obs_size, n_hidden_channels, initialW=winit)
            self.l31_1 = L.Linear(n_hidden_channels, n_hidden_channels, initialW=winit)
            self.l31_2 = L.Linear(n_hidden_channels, n_actions[31])

            self.l32_0 = L.Linear(obs_size, n_hidden_channels, initialW=winit)
            self.l32_1 = L.Linear(n_hidden_channels, n_hidden_channels, initialW=winit)
            self.l32_2 = L.Linear(n_hidden_channels, n_actions[32])

            self.l33_0 = L.Linear(obs_size, n_hidden_channels, initialW=winit)
            self.l33_1 = L.Linear(n_hidden_channels, n_hidden_channels, initialW=winit)
            self.l33_2 = L.Linear(n_hidden_channels, n_actions[33])

            self.l34_0 = L.Linear(obs_size, n_hidden_channels, initialW=winit)
            self.l34_1 = L.Linear(n_hidden_channels, n_hidden_channels, initialW=winit)
            self.l34_2 = L.Linear(n_hidden_channels, n_actions[34])

            self.l35_0 = L.Linear(obs_size, n_hidden_channels, initialW=winit)
            self.l35_1 = L.Linear(n_hidden_channels, n_hidden_channels, initialW=winit)
            self.l35_2 = L.Linear(n_hidden_channels, n_actions[35])

            self.l36_0 = L.Linear(obs_size, n_hidden_channels, initialW=winit)
            self.l36_1 = L.Linear(n_hidden_channels, n_hidden_channels, initialW=winit)
            self.l36_2 = L.Linear(n_hidden_channels, n_actions[36])

            self.l37_0 = L.Linear(obs_size, n_hidden_channels, initialW=winit)
            self.l37_1 = L.Linear(n_hidden_channels, n_hidden_channels, initialW=winit)
            self.l37_2 = L.Linear(n_hidden_channels, n_actions[37])

            self.l38_0 = L.Linear(obs_size, n_hidden_channels, initialW=winit)
            self.l38_1 = L.Linear(n_hidden_channels, n_hidden_channels, initialW=winit)
            self.l38_2 = L.Linear(n_hidden_channels, n_actions[38])

            self.l39_0 = L.Linear(obs_size, n_hidden_channels, initialW=winit)
            self.l39_1 = L.Linear(n_hidden_channels, n_hidden_channels, initialW=winit)
            self.l39_2 = L.Linear(n_hidden_channels, n_actions[39])


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


        h8 = F.relu(self.l8_0(x))
        h8 = F.relu(self.l8_1(h8))
        y8 = chainerrl.action_value.DiscreteActionValue(self.l8_2(h8))

        h9 = F.relu(self.l9_0(x))
        h9 = F.relu(self.l9_1(h9))
        y9 = chainerrl.action_value.DiscreteActionValue(self.l9_2(h9))

        h10 = F.relu(self.l10_0(x))
        h10 = F.relu(self.l10_1(h10))
        y10 = chainerrl.action_value.DiscreteActionValue(self.l10_2(h10))

        h11 = F.relu(self.l11_0(x))
        h11 = F.relu(self.l11_1(h11))
        y11 = chainerrl.action_value.DiscreteActionValue(self.l11_2(h11))

        h12 = F.relu(self.l12_0(x))
        h12 = F.relu(self.l12_1(h12))
        y12 = chainerrl.action_value.DiscreteActionValue(self.l12_2(h12))

        h13 = F.relu(self.l13_0(x))
        h13 = F.relu(self.l13_1(h13))
        y13 = chainerrl.action_value.DiscreteActionValue(self.l13_2(h13))

        h14 = F.relu(self.l14_0(x))
        h14 = F.relu(self.l14_1(h14))
        y14 = chainerrl.action_value.DiscreteActionValue(self.l14_2(h14))

        h15 = F.relu(self.l15_0(x))
        h15 = F.relu(self.l15_1(h15))
        y15 = chainerrl.action_value.DiscreteActionValue(self.l15_2(h15))

        h16 = F.relu(self.l16_0(x))
        h16 = F.relu(self.l16_1(h16))
        y16 = chainerrl.action_value.DiscreteActionValue(self.l16_2(h16))

        h17 = F.relu(self.l17_0(x))
        h17 = F.relu(self.l17_1(h17))
        y17 = chainerrl.action_value.DiscreteActionValue(self.l17_2(h17))

        h18 = F.relu(self.l18_0(x))
        h18 = F.relu(self.l18_1(h18))
        y18 = chainerrl.action_value.DiscreteActionValue(self.l18_2(h18))

        h19 = F.relu(self.l19_0(x))
        h19 = F.relu(self.l19_1(h19))
        y19 = chainerrl.action_value.DiscreteActionValue(self.l19_2(h19))

        h20 = F.relu(self.l20_0(x))
        h20 = F.relu(self.l20_1(h20))
        y20 = chainerrl.action_value.DiscreteActionValue(self.l20_2(h20))

        h21 = F.relu(self.l21_0(x))
        h21 = F.relu(self.l21_1(h21))
        y21 = chainerrl.action_value.DiscreteActionValue(self.l21_2(h21))

        h22 = F.relu(self.l22_0(x))
        h22 = F.relu(self.l22_1(h22))
        y22 = chainerrl.action_value.DiscreteActionValue(self.l22_2(h22))

        h23 = F.relu(self.l23_0(x))
        h23 = F.relu(self.l23_1(h23))
        y23 = chainerrl.action_value.DiscreteActionValue(self.l23_2(h23))

        h24 = F.relu(self.l24_0(x))
        h24 = F.relu(self.l24_1(h24))
        y24 = chainerrl.action_value.DiscreteActionValue(self.l24_2(h24))

        h25 = F.relu(self.l25_0(x))
        h25 = F.relu(self.l25_1(h25))
        y25 = chainerrl.action_value.DiscreteActionValue(self.l25_2(h25))

        h26 = F.relu(self.l26_0(x))
        h26 = F.relu(self.l26_1(h26))
        y26 = chainerrl.action_value.DiscreteActionValue(self.l26_2(h26))

        h27 = F.relu(self.l27_0(x))
        h27 = F.relu(self.l27_1(h27))
        y27 = chainerrl.action_value.DiscreteActionValue(self.l27_2(h27))

        h28 = F.relu(self.l28_0(x))
        h28 = F.relu(self.l28_1(h28))
        y28 = chainerrl.action_value.DiscreteActionValue(self.l28_2(h28))

        h29 = F.relu(self.l29_0(x))
        h29 = F.relu(self.l29_1(h29))
        y29 = chainerrl.action_value.DiscreteActionValue(self.l29_2(h29))

        h30 = F.relu(self.l30_0(x))
        h30 = F.relu(self.l30_1(h30))
        y30 = chainerrl.action_value.DiscreteActionValue(self.l30_2(h30))

        h31 = F.relu(self.l31_0(x))
        h31 = F.relu(self.l31_1(h31))
        y31 = chainerrl.action_value.DiscreteActionValue(self.l31_2(h31))

        h32 = F.relu(self.l32_0(x))
        h32 = F.relu(self.l32_1(h32))
        y32 = chainerrl.action_value.DiscreteActionValue(self.l32_2(h32))

        h33 = F.relu(self.l33_0(x))
        h33 = F.relu(self.l33_1(h33))
        y33 = chainerrl.action_value.DiscreteActionValue(self.l33_2(h33))

        h34 = F.relu(self.l34_0(x))
        h34 = F.relu(self.l34_1(h34))
        y34 = chainerrl.action_value.DiscreteActionValue(self.l34_2(h34))

        h35 = F.relu(self.l35_0(x))
        h35 = F.relu(self.l35_1(h35))
        y35 = chainerrl.action_value.DiscreteActionValue(self.l35_2(h35))

        h36 = F.relu(self.l36_0(x))
        h36 = F.relu(self.l36_1(h36))
        y36 = chainerrl.action_value.DiscreteActionValue(self.l36_2(h36))

        h37 = F.relu(self.l37_0(x))
        h37 = F.relu(self.l37_1(h37))
        y37 = chainerrl.action_value.DiscreteActionValue(self.l37_2(h37))

        h38 = F.relu(self.l38_0(x))
        h38 = F.relu(self.l38_1(h38))
        y38 = chainerrl.action_value.DiscreteActionValue(self.l38_2(h38))

        h39 = F.relu(self.l39_0(x))
        h39 = F.relu(self.l39_1(h39))
        y39 = chainerrl.action_value.DiscreteActionValue(self.l39_2(h39))

        return [y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, \
                y11, y12, y13, y14, y15, y16, y17, y18, y19, y20, \
                y21, y22, y23, y24, y25, y26, y27, y28, y29, y30, \
                y31, y32, y33, y34, y35, y36, y37, y38, y39]

def ddqn_multi_full(cfg, env):
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
        q_func = QFunction(obs_size, n_actions)
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

    agent = DoubleDQN(q_func, opt, rbuf, gpu=cfg.model.gpu, gamma=0.99,
                explorer=explorer, replay_start_size=cfg.model.replay_start_size,
                target_update_interval=cfg.model.target_update_interval,
                update_interval=cfg.model.update_interval,
                minibatch_size=cfg.model.batchsize,
                target_update_method='hard',
                soft_update_tau=1e-2,
                )

    return agent