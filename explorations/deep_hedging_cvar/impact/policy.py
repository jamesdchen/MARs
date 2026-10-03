"""Actor-critic for (target position, trade signal) actions.

Same contract as ../policy.py: forward_eval returns (Normal, value), and the
value is masked to 0 on the post-expiry observation. The actor mean is also
the network trained by the pathwise baselines.
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


class ImpactHedgePolicy(nn.Module):
    def __init__(self, hidden=64, init_logstd=(-2.0, -0.7), init_mean=(0.5, 0.25)):
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
