"""ODE dynamics functions for Neural ODE models.

IMPORTANT: ODE functions must use smooth activations (Tanh, Softplus, ELU)
for proper gradient flow through the ODE solver. Avoid ReLU/LeakyReLU
as they are not Lipschitz continuous.
"""

import torch
import torch.nn as nn


class ODEFunc(nn.Module):
    """Standard ODE dynamics function f(t, z) for dz/dt = f(t, z).

    Uses a multi-layer MLP with smooth activations.

    Args:
        latent_dim: Dimension of the latent state.
        hidden_dim: Hidden layer dimension.
        n_layers: Number of hidden layers.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 2,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Build MLP with smooth activations (CRITICAL: use Tanh, not ReLU)
        layers = []
        layers.append(nn.Linear(latent_dim, hidden_dim))
        layers.append(nn.Tanh())

        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(hidden_dim, latent_dim))

        self.net = nn.Sequential(*layers)

        # Track number of function evaluations (for debugging)
        self.nfe = 0

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Compute dz/dt at time t given state z.

        Args:
            t: Current time (scalar or tensor).
            z: Current state [batch, latent_dim].

        Returns:
            Time derivative dz/dt [batch, latent_dim].
        """
        self.nfe += 1
        return self.net(z)

    def reset_nfe(self) -> None:
        """Reset the function evaluation counter."""
        self.nfe = 0


class GRUODEFunc(nn.Module):
    """GRU-based ODE dynamics for continuous-time hidden state evolution.

    Implements continuous-time GRU dynamics following:
    dh/dt = (1 - z) * (h_tilde - h)

    where z is the update gate and h_tilde is the candidate state.

    Args:
        latent_dim: Dimension of the hidden state.
        hidden_dim: Dimension for computing gates.
    """

    def __init__(self, latent_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim

        # Update gate
        self.W_z = nn.Linear(latent_dim, latent_dim)
        # Reset gate
        self.W_r = nn.Linear(latent_dim, latent_dim)
        # Candidate hidden state
        self.W_h = nn.Linear(latent_dim, latent_dim)

        self.nfe = 0

    def forward(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Compute continuous GRU dynamics.

        Args:
            t: Current time.
            h: Current hidden state [batch, latent_dim].

        Returns:
            Time derivative dh/dt.
        """
        self.nfe += 1

        # Gates (using sigmoid for smooth transitions)
        z = torch.sigmoid(self.W_z(h))
        r = torch.sigmoid(self.W_r(h))

        # Candidate state (using Tanh for smooth activation)
        h_tilde = torch.tanh(self.W_h(r * h))

        # Continuous dynamics: dh/dt = (1 - z) * (h_tilde - h)
        dh_dt = (1 - z) * (h_tilde - h)

        return dh_dt

    def reset_nfe(self) -> None:
        """Reset the function evaluation counter."""
        self.nfe = 0


class AugmentedODEFunc(nn.Module):
    """Augmented Neural ODE for increased expressiveness.

    Augments the state space with additional dimensions initialized to zero,
    allowing the ODE to learn more complex trajectories.

    Args:
        input_dim: Input state dimension.
        augment_dim: Number of augmented dimensions.
        hidden_dim: Hidden layer dimension.
        n_layers: Number of hidden layers.
    """

    def __init__(
        self,
        input_dim: int,
        augment_dim: int = 0,
        hidden_dim: int = 128,
        n_layers: int = 2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.augment_dim = augment_dim
        self.total_dim = input_dim + augment_dim

        # Build MLP for augmented space
        layers = []
        layers.append(nn.Linear(self.total_dim, hidden_dim))
        layers.append(nn.Tanh())

        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(hidden_dim, self.total_dim))

        self.net = nn.Sequential(*layers)
        self.nfe = 0

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Compute dynamics in augmented space.

        Args:
            t: Current time.
            z: Current state [batch, total_dim].

        Returns:
            Time derivative [batch, total_dim].
        """
        self.nfe += 1
        return self.net(z)

    def augment(self, x: torch.Tensor) -> torch.Tensor:
        """Augment input with zeros.

        Args:
            x: Input tensor [batch, input_dim].

        Returns:
            Augmented tensor [batch, total_dim].
        """
        if self.augment_dim == 0:
            return x

        zeros = torch.zeros(
            x.shape[0], self.augment_dim,
            device=x.device, dtype=x.dtype
        )
        return torch.cat([x, zeros], dim=-1)

    def unaugment(self, z: torch.Tensor) -> torch.Tensor:
        """Extract original dimensions from augmented state.

        Args:
            z: Augmented state [batch, total_dim].

        Returns:
            Original dimensions [batch, input_dim].
        """
        return z[..., :self.input_dim]

    def reset_nfe(self) -> None:
        """Reset the function evaluation counter."""
        self.nfe = 0
