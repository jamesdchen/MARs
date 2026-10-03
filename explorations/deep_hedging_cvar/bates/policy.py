"""Actor-critic for the Bates book: (stock target, stock signal, swap target,
swap signal) actions.

Same contract as ../impact/policy.py and ../policy.py, which is PufferLib's:
forward_eval(obs, state) returns (torch.distributions.Normal, value), and the
value is forced to zero on the post-expiry observation (time fraction 1),
which has no future reward. The actor mean is also the network trained by
the pathwise baselines (train_pathwise.py); they train actor only, and the
critic and logstd exist for PPO.

Initial actions: the zero last layer makes the actor start from init_mean
on every state. The book is short a 100 call and a 95 put, whose combined
delta is about -0.3, so the stock target starts at 0.3; the swap target
starts flat. Both signals start slightly positive, so the initial policy
trades on every date: a target only gets a pathwise gradient on dates where
its instrument trades, and the fee gradient then teaches the signal when not
to. logstd holds one log standard deviation for each action dimension.
"""

import torch
import torch.nn as nn

from market import OBS_DIM, ACT_DIM


def mlp(inp, hidden, out, depth=2):
    layers, d = [], inp
    for _ in range(depth):
        layers += [nn.Linear(d, hidden), nn.Tanh()]
        d = hidden
    layers.append(nn.Linear(d, out))
    return nn.Sequential(*layers)


class HedgePolicy(nn.Module):
    def __init__(self, hidden=64, init_logstd=(-2.0, -0.7, -1.3, -0.7),
                 init_mean=(0.3, 0.25, 0.0, 0.25)):
        super().__init__()
        self.hidden_size = hidden
        self.is_continuous = True
        self.actor = mlp(OBS_DIM, hidden, ACT_DIM)
        self.critic = mlp(OBS_DIM, hidden, 1)
        nn.init.zeros_(self.actor[-1].weight)
        with torch.no_grad():
            self.actor[-1].bias.copy_(torch.tensor(init_mean))
        self.logstd = nn.Parameter(torch.tensor([init_logstd]))

    def mean_action(self, obs):
        return self.actor(obs.float())

    def forward_eval(self, obs, state=None):
        obs = obs.float()
        mean = self.actor(obs)
        dist = torch.distributions.Normal(mean, self.logstd.exp().expand_as(mean))
        alive = (obs[:, :1] < 1).float()
        return dist, self.critic(obs) * alive

    def forward(self, obs, state=None):
        return self.forward_eval(obs, state)
