"""Actor-critic used by PPO; the actor mean doubles as the pathwise network."""

import torch
import torch.nn as nn

from hedging import OBS_DIM


def mlp(inp, hidden, out, depth=2):
    layers, d = [], inp
    for _ in range(depth):
        layers += [nn.Linear(d, hidden), nn.Tanh()]
        d = hidden
    layers.append(nn.Linear(d, out))
    return nn.Sequential(*layers)


class HedgePolicy(nn.Module):
    """Gaussian hedge ratio policy with a separate critic.

    Follows PufferLib's policy contract: forward_eval(obs, state) returns
    (torch.distributions.Normal, value). The value is forced to zero at the
    post-expiry observation (time fraction 1), which has no future reward.
    """

    def __init__(self, hidden=64, init_logstd=-2.0):
        super().__init__()
        self.hidden_size = hidden
        self.is_continuous = True
        self.actor = mlp(OBS_DIM, hidden, 1)
        self.critic = mlp(OBS_DIM, hidden, 1)
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.constant_(self.actor[-1].bias, 0.5)
        self.logstd = nn.Parameter(torch.full((1, 1), init_logstd))

    def mean_action(self, obs):
        return self.actor(obs.float()).squeeze(-1)

    def forward_eval(self, obs, state=None):
        obs = obs.float()
        mean = self.actor(obs)
        dist = torch.distributions.Normal(mean, self.logstd.exp().expand_as(mean))
        alive = (obs[:, :1] < 1).float()
        return dist, self.critic(obs) * alive

    def forward(self, obs, state=None):
        return self.forward_eval(obs, state)
