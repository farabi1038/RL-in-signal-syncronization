import chainer
from chainerrl import policies
from chainerrl.agents import a3c
import chainer.functions as F
import chainer.links as L
from chainerrl import links
from chainerrl.agents import PPO
import chainerrl
import numpy as np
import chainerrl.misc.random_seed as chainerseed
from chainerrl.optimizers.nonbias_weight_decay import NonbiasWeightDecay

def ppo_v1(cfg, env):
    class A3CFFGaussian(chainer.Chain, a3c.A3CModel):

        def __init__(self, obs_size, action_space,
                     n_hidden_layers=2, n_hidden_channels=64,
                     bound_mean=None):
            assert bound_mean in [False, True]
            super().__init__()
            hidden_sizes = (n_hidden_channels,) * n_hidden_layers
            with self.init_scope():
                self.pi = policies.FCGaussianPolicyWithStateIndependentCovariance(
                    obs_size, action_space.low.size,
                    n_hidden_layers, n_hidden_channels,
                    var_type='diagonal', nonlinearity=F.tanh,
                    bound_mean=bound_mean,
                    min_action=action_space.low, max_action=action_space.high,
                    mean_wscale=1e-2)
                self.v = links.MLP(obs_size, 1, hidden_sizes=hidden_sizes)

        def pi_and_v(self, state):
            return self.pi(state), self.v(state)


    chainerseed.set_random_seed(cfg.model.cseed)
    n_actions = env.action.space()
    obs_size = len(env.state)

    obs_normalizer = links.EmpiricalNormalization(obs_size, clip_threshold=cfg.model.clip_threshold)
    model = A3CFFGaussian(obs_size, n_actions, bound_mean=cfg.model.bound_mean)
    opt = chainer.optimizers.Adam(alpha=cfg.model.lr, eps=1e-5)
    opt.setup(model)

    if cfg.model.weight_decay > 0:
        opt.add_hook(NonbiasWeightDecay(cfg.model.weight_decay))
    
    agent = PPO(model, opt,
                obs_normalizer=obs_normalizer,
                gpu=cfg.model.gpu,
                update_interval=cfg.model.update_interval,
                minibatch_size=cfg.model.batchsize, epochs=cfg.model.epochs,
                clip_eps_vf=None, entropy_coef=cfg.model.entropy_coef,
                standardize_advantages=cfg.model.standardize_advantages,
                )

    return agent