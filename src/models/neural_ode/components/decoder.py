"""Decoder networks for Neural ODE models."""

import torch
import torch.nn as nn


class MLPDecoder(nn.Module):
    """MLP decoder for mapping latent states to observations.

    Args:
        latent_dim: Input latent dimension.
        output_dim: Output observation dimension.
        hidden_dim: Hidden layer dimension.
        n_layers: Number of hidden layers.
    """

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_dim: int = 64,
        n_layers: int = 2,
    ):
        super().__init__()

        layers = []
        layers.append(nn.Linear(latent_dim, hidden_dim))
        layers.append(nn.ReLU())  # ReLU is fine in decoder (not in ODE dynamics)

        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent state to observation.

        Args:
            z: Latent state [batch, latent_dim] or [times, batch, latent_dim].

        Returns:
            Decoded observation.
        """
        return self.net(z)


class SequenceDecoder(nn.Module):
    """Decoder that outputs a sequence from initial latent state.

    Uses an RNN to generate sequential outputs from an initial state.

    Args:
        latent_dim: Input latent dimension.
        output_dim: Output observation dimension.
        hidden_dim: RNN hidden dimension.
        seq_len: Output sequence length.
    """

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_dim: int = 64,
        seq_len: int = 96,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

        # Project latent to initial hidden state
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)

        # GRU for sequence generation
        self.gru = nn.GRU(output_dim, hidden_dim, batch_first=True)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        z0: torch.Tensor,
        x_init: torch.Tensor = None,
    ) -> torch.Tensor:
        """Generate sequence from initial latent state.

        Args:
            z0: Initial latent state [batch, latent_dim].
            x_init: Optional initial input [batch, 1, output_dim].

        Returns:
            Generated sequence [batch, seq_len, output_dim].
        """
        batch_size = z0.size(0)
        device = z0.device

        # Initialize hidden state from latent
        h = self.latent_to_hidden(z0).unsqueeze(0)  # [1, batch, hidden]

        # Initialize first input
        if x_init is None:
            x = torch.zeros(batch_size, 1, self.output_proj.out_features, device=device)
        else:
            x = x_init

        # Generate sequence autoregressively
        outputs = []
        for _ in range(self.seq_len):
            out, h = self.gru(x, h)
            y = self.output_proj(out)
            outputs.append(y)
            x = y  # Use output as next input

        return torch.cat(outputs, dim=1)


class ODEDecoder(nn.Module):
    """Decoder using ODE dynamics to generate trajectory.

    Integrates ODE from initial latent state over specified time points.

    Args:
        latent_dim: Latent state dimension.
        output_dim: Output observation dimension.
        ode_func: ODE dynamics function.
        use_adjoint: Whether to use adjoint method.
    """

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        ode_func: nn.Module,
        use_adjoint: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.ode_func = ode_func
        self.use_adjoint = use_adjoint

        # Map latent trajectory to observations
        self.output_net = MLPDecoder(latent_dim, output_dim)

    def forward(
        self,
        z0: torch.Tensor,
        times: torch.Tensor,
    ) -> torch.Tensor:
        """Generate trajectory from initial state.

        Args:
            z0: Initial latent state [batch, latent_dim].
            times: Time points to evaluate [num_times].

        Returns:
            Trajectory [batch, num_times, output_dim].
        """
        # Get ODE integrator
        if self.use_adjoint:
            from torchdiffeq import odeint_adjoint as odeint
        else:
            from torchdiffeq import odeint

        # Integrate ODE
        # z_traj: [num_times, batch, latent_dim]
        z_traj = odeint(self.ode_func, z0, times, method="dopri5")

        # Map to observations: [num_times, batch, output_dim]
        y_traj = self.output_net(z_traj)

        # Transpose to [batch, num_times, output_dim]
        y_traj = y_traj.transpose(0, 1)

        return y_traj
