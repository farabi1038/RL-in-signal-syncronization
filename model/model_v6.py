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

            self.time_input_0 = L.Linear(41, n_hidden_channels//2)
            self.time_input_1 = L.Linear(n_hidden_channels//2, n_hidden_channels//2)
            self.time_input_2 = L.Linear(n_hidden_channels//2, n_hidden_channels//2)

            self.l0_0 = L.Linear(obs_size - 41, n_hidden_channels//2)
            self.l0_1 = L.Linear(n_hidden_channels//2, n_hidden_channels//2)
            self.l0_2 = L.Linear(n_hidden_channels, n_actions[0])

            self.l1_0 = L.Linear(obs_size - 41, n_hidden_channels//2)
            self.l1_1 = L.Linear(n_hidden_channels//2, n_hidden_channels//2)
            self.l1_2 = L.Linear(n_hidden_channels, n_actions[1])

            self.l2_0 = L.Linear(obs_size - 41, n_hidden_channels//2)
            self.l2_1 = L.Linear(n_hidden_channels//2, n_hidden_channels//2)
            self.l2_2 = L.Linear(n_hidden_channels, n_actions[2])

            self.l3_0 = L.Linear(obs_size - 41, n_hidden_channels//2)
            self.l3_1 = L.Linear(n_hidden_channels//2, n_hidden_channels//2)
            self.l3_2 = L.Linear(n_hidden_channels, n_actions[3])

            self.l4_0 = L.Linear(obs_size - 41, n_hidden_channels//2)
            self.l4_1 = L.Linear(n_hidden_channels//2, n_hidden_channels//2)
            self.l4_2 = L.Linear(n_hidden_channels, n_actions[4])

            self.l5_0 = L.Linear(obs_size - 41, n_hidden_channels//2)
            self.l5_1 = L.Linear(n_hidden_channels//2, n_hidden_channels//2)
            self.l5_2 = L.Linear(n_hidden_channels, n_actions[5])

            self.l6_0 = L.Linear(obs_size - 41, n_hidden_channels//2)
            self.l6_1 = L.Linear(n_hidden_channels//2, n_hidden_channels//2)
            self.l6_2 = L.Linear(n_hidden_channels, n_actions[6])

            self.l7_0 = L.Linear(obs_size - 41, n_hidden_channels//2)
            self.l7_1 = L.Linear(n_hidden_channels//2, n_hidden_channels//2)
            self.l7_2 = L.Linear(n_hidden_channels, n_actions[7])

            self.l8_0 = L.Linear(obs_size - 41, n_hidden_channels//2)
            self.l8_1 = L.Linear(n_hidden_channels//2, n_hidden_channels//2)
            self.l8_2 = L.Linear(n_hidden_channels, n_actions[8])

            self.l9_0 = L.Linear(obs_size - 41, n_hidden_channels//2)
            self.l9_1 = L.Linear(n_hidden_channels//2, n_hidden_channels//2)
            self.l9_2 = L.Linear(n_hidden_channels, n_actions[9])

            self.l10_0 = L.Linear(obs_size - 41, n_hidden_channels//2)
            self.l10_1 = L.Linear(n_hidden_channels//2, n_hidden_channels//2)
            self.l10_2 = L.Linear(n_hidden_channels, n_actions[10])

            self.l11_0 = L.Linear(obs_size - 41, n_hidden_channels//2)
            self.l11_1 = L.Linear(n_hidden_channels//2, n_hidden_channels//2)
            self.l11_2 = L.Linear(n_hidden_channels, n_actions[11])

            self.l12_0 = L.Linear(obs_size - 41, n_hidden_channels//2)
            self.l12_1 = L.Linear(n_hidden_channels//2, n_hidden_channels//2)
            self.l12_2 = L.Linear(n_hidden_channels, n_actions[12])

            self.l13_0 = L.Linear(obs_size - 41, n_hidden_channels//2)
            self.l13_1 = L.Linear(n_hidden_channels//2, n_hidden_channels//2)
            self.l13_2 = L.Linear(n_hidden_channels, n_actions[13])

            self.l14_0 = L.Linear(obs_size - 41, n_hidden_channels//2)
            self.l14_1 = L.Linear(n_hidden_channels//2, n_hidden_channels//2)
            self.l14_2 = L.Linear(n_hidden_channels, n_actions[14])

            self.l15_0 = L.Linear(obs_size - 41, n_hidden_channels//2)
            self.l15_1 = L.Linear(n_hidden_channels//2, n_hidden_channels//2)
            self.l15_2 = L.Linear(n_hidden_channels, n_actions[15])

            self.l16_0 = L.Linear(obs_size - 41, n_hidden_channels//2)
            self.l16_1 = L.Linear(n_hidden_channels//2, n_hidden_channels//2)
            self.l16_2 = L.Linear(n_hidden_channels, n_actions[16])

            self.l17_0 = L.Linear(obs_size - 41, n_hidden_channels//2)
            self.l17_1 = L.Linear(n_hidden_channels//2, n_hidden_channels//2)
            self.l17_2 = L.Linear(n_hidden_channels, n_actions[17])

            self.l18_0 = L.Linear(obs_size - 41, n_hidden_channels//2)
            self.l18_1 = L.Linear(n_hidden_channels//2, n_hidden_channels//2)
            self.l18_2 = L.Linear(n_hidden_channels, n_actions[18])

            self.l19_0 = L.Linear(obs_size - 41, n_hidden_channels//2)
            self.l19_1 = L.Linear(n_hidden_channels//2, n_hidden_channels//2)
            self.l19_2 = L.Linear(n_hidden_channels, n_actions[19])

            self.l20_0 = L.Linear(obs_size - 41, n_hidden_channels//2)
            self.l20_1 = L.Linear(n_hidden_channels//2, n_hidden_channels//2)
            self.l20_2 = L.Linear(n_hidden_channels, n_actions[20])

            self.l21_0 = L.Linear(obs_size - 41, n_hidden_channels//2)
            self.l21_1 = L.Linear(n_hidden_channels//2, n_hidden_channels//2)
            self.l21_2 = L.Linear(n_hidden_channels, n_actions[21])

            self.l22_0 = L.Linear(obs_size - 41, n_hidden_channels//2)
            self.l22_1 = L.Linear(n_hidden_channels//2, n_hidden_channels//2)
            self.l22_2 = L.Linear(n_hidden_channels, n_actions[22])

            self.l23_0 = L.Linear(obs_size - 41, n_hidden_channels//2)
            self.l23_1 = L.Linear(n_hidden_channels//2, n_hidden_channels//2)
            self.l23_2 = L.Linear(n_hidden_channels, n_actions[23])

            self.l24_0 = L.Linear(obs_size - 41, n_hidden_channels//2)
            self.l24_1 = L.Linear(n_hidden_channels//2, n_hidden_channels//2)
            self.l24_2 = L.Linear(n_hidden_channels, n_actions[24])

            self.l25_0 = L.Linear(obs_size - 41, n_hidden_channels//2)
            self.l25_1 = L.Linear(n_hidden_channels//2, n_hidden_channels//2)
            self.l25_2 = L.Linear(n_hidden_channels, n_actions[25])

            self.l26_0 = L.Linear(obs_size - 41, n_hidden_channels//2)
            self.l26_1 = L.Linear(n_hidden_channels//2, n_hidden_channels//2)
            self.l26_2 = L.Linear(n_hidden_channels, n_actions[26])

            self.l27_0 = L.Linear(obs_size - 41, n_hidden_channels//2)
            self.l27_1 = L.Linear(n_hidden_channels//2, n_hidden_channels//2)
            self.l27_2 = L.Linear(n_hidden_channels, n_actions[27])

            self.l28_0 = L.Linear(obs_size - 41, n_hidden_channels//2)
            self.l28_1 = L.Linear(n_hidden_channels//2, n_hidden_channels//2)
            self.l28_2 = L.Linear(n_hidden_channels, n_actions[28])

            self.l29_0 = L.Linear(obs_size - 41, n_hidden_channels//2)
            self.l29_1 = L.Linear(n_hidden_channels//2, n_hidden_channels//2)
            self.l29_2 = L.Linear(n_hidden_channels, n_actions[29])

            self.l30_0 = L.Linear(obs_size - 41, n_hidden_channels//2)
            self.l30_1 = L.Linear(n_hidden_channels//2, n_hidden_channels//2)
            self.l30_2 = L.Linear(n_hidden_channels, n_actions[30])

            self.l31_0 = L.Linear(obs_size - 41, n_hidden_channels//2)
            self.l31_1 = L.Linear(n_hidden_channels//2, n_hidden_channels//2)
            self.l31_2 = L.Linear(n_hidden_channels, n_actions[31])

            self.l32_0 = L.Linear(obs_size - 41, n_hidden_channels//2)
            self.l32_1 = L.Linear(n_hidden_channels//2, n_hidden_channels//2)
            self.l32_2 = L.Linear(n_hidden_channels, n_actions[32])

            self.l33_0 = L.Linear(obs_size - 41, n_hidden_channels//2)
            self.l33_1 = L.Linear(n_hidden_channels//2, n_hidden_channels//2)
            self.l33_2 = L.Linear(n_hidden_channels, n_actions[33])

            self.l34_0 = L.Linear(obs_size - 41, n_hidden_channels//2)
            self.l34_1 = L.Linear(n_hidden_channels//2, n_hidden_channels//2)
            self.l34_2 = L.Linear(n_hidden_channels, n_actions[34])

            self.l35_0 = L.Linear(obs_size - 41, n_hidden_channels//2)
            self.l35_1 = L.Linear(n_hidden_channels//2, n_hidden_channels//2)
            self.l35_2 = L.Linear(n_hidden_channels, n_actions[35])

            self.l36_0 = L.Linear(obs_size - 41, n_hidden_channels//2)
            self.l36_1 = L.Linear(n_hidden_channels//2, n_hidden_channels//2)
            self.l36_2 = L.Linear(n_hidden_channels, n_actions[36])

            self.l37_0 = L.Linear(obs_size - 41, n_hidden_channels//2)
            self.l37_1 = L.Linear(n_hidden_channels//2, n_hidden_channels//2)
            self.l37_2 = L.Linear(n_hidden_channels, n_actions[37])

            self.l38_0 = L.Linear(obs_size - 41, n_hidden_channels//2)
            self.l38_1 = L.Linear(n_hidden_channels//2, n_hidden_channels//2)
            self.l38_2 = L.Linear(n_hidden_channels, n_actions[38])

            self.l39_0 = L.Linear(obs_size - 41, n_hidden_channels//2)
            self.l39_1 = L.Linear(n_hidden_channels//2, n_hidden_channels//2)
            self.l39_2 = L.Linear(n_hidden_channels, n_actions[39])


    def __call__(self, x, test=False):

        length = len(x[0]) - 41
        obs, timer_input = x[:,:length], x[:,length:]

        t0 = F.relu(self.time_input_0(timer_input))
        t1 = F.relu(self.time_input_1(t0))
        t2 = F.relu(self.time_input_2(t1))

        h0 = F.relu(self.l0_0(obs))
        h0 = F.relu(self.l0_1(h0))
        y0 = chainerrl.action_value.DiscreteActionValue(self.l0_2(F.concat((h0,t2), axis=1)))

        h1 = F.relu(self.l1_0(obs))
        h1 = F.relu(self.l1_1(h1))
        y1 = chainerrl.action_value.DiscreteActionValue(self.l1_2(F.concat((h1,t2), axis=1)))

        h2 = F.relu(self.l2_0(obs))
        h2 = F.relu(self.l2_1(h2))
        y2 = chainerrl.action_value.DiscreteActionValue(self.l2_2(F.concat((h2,t2), axis=1)))

        h3 = F.relu(self.l3_0(obs))
        h3 = F.relu(self.l3_1(h3))
        y3 = chainerrl.action_value.DiscreteActionValue(self.l3_2(F.concat((h3,t2), axis=1)))

        h4 = F.relu(self.l4_0(obs))
        h4 = F.relu(self.l4_1(h4))
        y4 = chainerrl.action_value.DiscreteActionValue(self.l4_2(F.concat((h4,t2), axis=1)))

        h5 = F.relu(self.l5_0(obs))
        h5 = F.relu(self.l5_1(h5))
        y5 = chainerrl.action_value.DiscreteActionValue(self.l5_2(F.concat((h5,t2), axis=1)))

        h6 = F.relu(self.l6_0(obs))
        h6 = F.relu(self.l6_1(h6))
        y6 = chainerrl.action_value.DiscreteActionValue(self.l6_2(F.concat((h6,t2), axis=1)))

        h7 = F.relu(self.l7_0(obs))
        h7 = F.relu(self.l7_1(h7))
        y7 = chainerrl.action_value.DiscreteActionValue(self.l7_2(F.concat((h7,t2), axis=1)))

        h8 = F.relu(self.l8_0(obs))
        h8 = F.relu(self.l8_1(h8))
        y8 = chainerrl.action_value.DiscreteActionValue(self.l8_2(F.concat((h8,t2), axis=1)))

        h9 = F.relu(self.l9_0(obs))
        h9 = F.relu(self.l9_1(h9))
        y9 = chainerrl.action_value.DiscreteActionValue(self.l9_2(F.concat((h9,t2), axis=1)))

        h10 = F.relu(self.l10_0(obs))
        h10 = F.relu(self.l10_1(h10))
        y10 = chainerrl.action_value.DiscreteActionValue(self.l10_2(F.concat((h10,t2), axis=1)))

        h11 = F.relu(self.l11_0(obs))
        h11 = F.relu(self.l11_1(h11))
        y11 = chainerrl.action_value.DiscreteActionValue(self.l11_2(F.concat((h11,t2), axis=1)))

        h12 = F.relu(self.l12_0(obs))
        h12 = F.relu(self.l12_1(h12))
        y12 = chainerrl.action_value.DiscreteActionValue(self.l12_2(F.concat((h12,t2), axis=1)))

        h13 = F.relu(self.l13_0(obs))
        h13 = F.relu(self.l13_1(h13))
        y13 = chainerrl.action_value.DiscreteActionValue(self.l13_2(F.concat((h13,t2), axis=1)))

        h14 = F.relu(self.l14_0(obs))
        h14 = F.relu(self.l14_1(h14))
        y14 = chainerrl.action_value.DiscreteActionValue(self.l14_2(F.concat((h14,t2), axis=1)))

        h15 = F.relu(self.l15_0(obs))
        h15 = F.relu(self.l15_1(h15))
        y15 = chainerrl.action_value.DiscreteActionValue(self.l15_2(F.concat((h15,t2), axis=1)))

        h16 = F.relu(self.l16_0(obs))
        h16 = F.relu(self.l16_1(h16))
        y16 = chainerrl.action_value.DiscreteActionValue(self.l16_2(F.concat((h16,t2), axis=1)))

        h17 = F.relu(self.l17_0(obs))
        h17 = F.relu(self.l17_1(h17))
        y17 = chainerrl.action_value.DiscreteActionValue(self.l17_2(F.concat((h17,t2), axis=1)))

        h18 = F.relu(self.l18_0(obs))
        h18 = F.relu(self.l18_1(h18))
        y18 = chainerrl.action_value.DiscreteActionValue(self.l18_2(F.concat((h18,t2), axis=1)))

        h19 = F.relu(self.l19_0(obs))
        h19 = F.relu(self.l19_1(h19))
        y19 = chainerrl.action_value.DiscreteActionValue(self.l19_2(F.concat((h19,t2), axis=1)))

        h20 = F.relu(self.l20_0(obs))
        h20 = F.relu(self.l20_1(h20))
        y20 = chainerrl.action_value.DiscreteActionValue(self.l20_2(F.concat((h20,t2), axis=1)))

        h21 = F.relu(self.l21_0(obs))
        h21 = F.relu(self.l21_1(h21))
        y21 = chainerrl.action_value.DiscreteActionValue(self.l21_2(F.concat((h21,t2), axis=1)))

        h22 = F.relu(self.l22_0(obs))
        h22 = F.relu(self.l22_1(h22))
        y22 = chainerrl.action_value.DiscreteActionValue(self.l22_2(F.concat((h22,t2), axis=1)))

        h23 = F.relu(self.l23_0(obs))
        h23 = F.relu(self.l23_1(h23))
        y23 = chainerrl.action_value.DiscreteActionValue(self.l23_2(F.concat((h23,t2), axis=1)))

        h24 = F.relu(self.l24_0(obs))
        h24 = F.relu(self.l24_1(h24))
        y24 = chainerrl.action_value.DiscreteActionValue(self.l24_2(F.concat((h24,t2), axis=1)))

        h25 = F.relu(self.l25_0(obs))
        h25 = F.relu(self.l25_1(h25))
        y25 = chainerrl.action_value.DiscreteActionValue(self.l25_2(F.concat((h25,t2), axis=1)))

        h26 = F.relu(self.l26_0(obs))
        h26 = F.relu(self.l26_1(h26))
        y26 = chainerrl.action_value.DiscreteActionValue(self.l26_2(F.concat((h26,t2), axis=1)))

        h27 = F.relu(self.l27_0(obs))
        h27 = F.relu(self.l27_1(h27))
        y27 = chainerrl.action_value.DiscreteActionValue(self.l27_2(F.concat((h27,t2), axis=1)))

        h28 = F.relu(self.l28_0(obs))
        h28 = F.relu(self.l28_1(h28))
        y28 = chainerrl.action_value.DiscreteActionValue(self.l28_2(F.concat((h28,t2), axis=1)))

        h29 = F.relu(self.l29_0(obs))
        h29 = F.relu(self.l29_1(h29))
        y29 = chainerrl.action_value.DiscreteActionValue(self.l29_2(F.concat((h29,t2), axis=1)))

        h30 = F.relu(self.l30_0(obs))
        h30 = F.relu(self.l30_1(h30))
        y30 = chainerrl.action_value.DiscreteActionValue(self.l30_2(F.concat((h30,t2), axis=1)))

        h31 = F.relu(self.l31_0(obs))
        h31 = F.relu(self.l31_1(h31))
        y31 = chainerrl.action_value.DiscreteActionValue(self.l31_2(F.concat((h31,t2), axis=1)))

        h32 = F.relu(self.l32_0(obs))
        h32 = F.relu(self.l32_1(h32))
        y32 = chainerrl.action_value.DiscreteActionValue(self.l32_2(F.concat((h32,t2), axis=1)))

        h33 = F.relu(self.l33_0(obs))
        h33 = F.relu(self.l33_1(h33))
        y33 = chainerrl.action_value.DiscreteActionValue(self.l33_2(F.concat((h33,t2), axis=1)))

        h34 = F.relu(self.l34_0(obs))
        h34 = F.relu(self.l34_1(h34))
        y34 = chainerrl.action_value.DiscreteActionValue(self.l34_2(F.concat((h34,t2), axis=1)))

        h35 = F.relu(self.l35_0(obs))
        h35 = F.relu(self.l35_1(h35))
        y35 = chainerrl.action_value.DiscreteActionValue(self.l35_2(F.concat((h35,t2), axis=1)))

        h36 = F.relu(self.l36_0(obs))
        h36 = F.relu(self.l36_1(h36))
        y36 = chainerrl.action_value.DiscreteActionValue(self.l36_2(F.concat((h36,t2), axis=1)))

        h37 = F.relu(self.l37_0(obs))
        h37 = F.relu(self.l37_1(h37))
        y37 = chainerrl.action_value.DiscreteActionValue(self.l37_2(F.concat((h37,t2), axis=1)))

        h38 = F.relu(self.l38_0(obs))
        h38 = F.relu(self.l38_1(h38))
        y38 = chainerrl.action_value.DiscreteActionValue(self.l38_2(F.concat((h38,t2), axis=1)))

        h39 = F.relu(self.l39_0(obs))
        h39 = F.relu(self.l39_1(h39))
        y39 = chainerrl.action_value.DiscreteActionValue(self.l39_2(F.concat((h39,t2), axis=1)))

        return [y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, \
                y11, y12, y13, y14, y15, y16, y17, y18, y19, y20, \
                y21, y22, y23, y24, y25, y26, y27, y28, y29, y30, \
                y31, y32, y33, y34, y35, y36, y37, y38, y39]

def dqn_multi_full_split(cfg, env):
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

    agent = DQN(q_func, opt, rbuf, gpu=cfg.model.gpu, gamma=0.99,
                explorer=explorer, replay_start_size=cfg.model.replay_start_size,
                target_update_interval=cfg.model.target_update_interval,
                update_interval=cfg.model.update_interval,
                minibatch_size=cfg.model.batchsize,
                target_update_method='hard',
                soft_update_tau=1e-2,
                )

    return agent