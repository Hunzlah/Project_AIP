import numpy as np
import torch.nn as nn


class PPOModel(nn.Module):
    """Basic PPO model."""

    def __init__(self, input_size, output_size):

        super(PPOModel, self).__init__()
        self.input_size = input_size

        # shared network (CNN Part)
        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=(8, 8),
                      stride=(4, 4)),
            nn.ELU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4),
                      stride=(2, 2)),
            nn.ELU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3),
                      stride=(1, 1)),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(in_features=7 * 7 * 64, out_features=256),
            nn.ELU(),
            nn.Linear(in_features=256, out_features=448),
            nn.ELU()
        )

        # actor head
        self.actor = nn.Sequential(
            nn.Linear(in_features=448, out_features=448),
            nn.ELU(),
            nn.Linear(in_features=448, out_features=output_size)
        )

        # the layer before having 2 value heads
        self.common_critic_layer = nn.Sequential(
            nn.Linear(in_features=448, out_features=448),
            nn.ELU()
        )

        # extrinsic and intrinsic reward heads
        self.critic_ext = nn.Linear(in_features=448, out_features=1)
        self.critic_int = nn.Linear(in_features=448, out_features=1)

        # initialize the weights
        for p in self.modules():

            # we need to do that in order to initialize the weights
            # otherwise it returns an error saying that ELU
            # (activation function) does not have weights

            # first initialize the nn.Conv2d and nn.Linear
            if isinstance(p, nn.Conv2d):
                nn.init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                nn.init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

        # initialize critics
        nn.init.orthogonal_(self.critic_ext.weight, 0.01)
        self.critic_ext.bias.data.zero_()

        nn.init.orthogonal_(self.critic_int.weight, 0.01)
        self.critic_int.bias.data.zero_()

        # initialize actor
        for i in range(len(self.actor)):
            if type(self.actor[i]) == nn.Linear:
                nn.init.orthogonal_(self.actor[i].weight, 0.01)
                self.actor[i].bias.data.zero_()

        # init value common layer
        for i in range(len(self.common_critic_layer)):
            if type(self.common_critic_layer[i]) == nn.Linear:
                nn.init.orthogonal_(self.common_critic_layer[i].weight, 0.1)
                self.common_critic_layer[i].bias.data.zero_()

    def forward(self, state):
        """Forward function."""
        x = self.feature(state)
        policy = self.actor(x)
        value_ext = self.critic_ext(self.common_critic_layer(x) + x)
        value_int = self.critic_int(self.common_critic_layer(x) + x)
        return policy, value_ext, value_int
