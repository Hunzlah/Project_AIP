import torch
import numpy as np
import torch.nn as nn


class RNDModel(nn.Module):
    """Random Network Distillation Model."""

    def __init__(self, input_size, output_size, device, num_networks=3):
        super(RNDModel, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.num_networks = num_networks
        self.device = device

        feature_output = 7 * 7 * 64

        # prediction networks
        self.predictors = [
            self._create_predictor_network(feature_output)
            for _ in range(num_networks)
        ]

        # target networks
        self.targets = [
            self._create_target_network(feature_output)
            for _ in range(num_networks)
        ]

    def _create_predictor_network(self, feature_output):
        """Creates, initializes and returns a Sequential Predictor Network."""
        network = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(8, 8),
                      stride=(4, 4)),
            nn.ELU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4),
                      stride=(2, 2)),
            nn.ELU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3),
                      stride=(1, 1)),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(in_features=feature_output, out_features=512),
            nn.ELU(),
            nn.Linear(in_features=512, out_features=512),
            nn.ELU(),
            nn.Linear(in_features=512, out_features=512)
        )
        self._init_weights(network)
        return network.to(self.device)

    def _create_target_network(self, feature_output):
        """Creates, initializes and returns a nn.Sequential Target Network."""
        network = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(8, 8),
                      stride=(4, 4)),
            nn.ELU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4),
                      stride=(2, 2)),
            nn.ELU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3),
                      stride=(1, 1)),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(in_features=feature_output, out_features=512)
        )
        self._init_weights(network)
        self._freeze_model(network)
        return network.to(self.device)

    @staticmethod
    def _init_weights(network):
        """Initializes weights and biases of a nn.Module network."""
        for p in network.modules():
            if isinstance(p, nn.Conv2d):
                nn.init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                nn.init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

    @staticmethod
    def _freeze_model(network):
        for param in network.parameters():
            param.requires_grad = False
        network.eval()

    def predictors_parameters(self):
        """Returns a list with all the Predictor Networks parameters."""
        return [
            param
            for predictor in self.predictors
            for param in list(predictor.parameters())
        ]

    def forward(self, next_obs):
        """Forward function."""
        targets_outs = [
            target_network(next_obs) for target_network in self.targets
        ]
        targets_features = torch.stack(targets_outs, dim=1)

        predictors_outs = [
            predictor_network(next_obs) for predictor_network in self.predictors
        ]
        predictors_features = torch.stack(predictors_outs, dim=1)

        return predictors_features, targets_features
