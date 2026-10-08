"""Define a small multilayer perceptron for ground-state energy regression."""

from numbers import Integral

from torch import nn


class EnergyMLP(nn.Module):
    """Map standardized potentials to standardized ground-state energies.

    The architecture is N -> 32 -> 16 -> 1 with Tanh hidden activations.
    Recover physical energies externally as prediction * y_std + y_mean.

    Attributes:
        network: Sequential fully connected regression network.
    """

    def __init__(self, N=200):
        """Initialize the regression network.

        Args:
            N: Number of potential values in each input sample.

        Raises:
            ValueError: If N is not a positive integer.
        """
        super().__init__()
        if isinstance(N, bool) or not isinstance(N, Integral) or N <= 0:
            raise ValueError('N must be a positive integer.')
        # Use Tanh in both hidden layers for a smooth nonlinear mapping.
        # Leave the output linear to permit both positive and negative energies.
        self.network = nn.Sequential(
            nn.Linear(N, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
        )

    def forward(self, V):
        """Predict standardized energies.

        Args:
            V: Standardized potential tensor of shape (batch_size, N).

        Returns:
            Energy tensor of shape (batch_size, 1).
        """
        return self.network(V)
