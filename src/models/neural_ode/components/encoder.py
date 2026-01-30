"""Encoder networks for Neural ODE models."""

from typing import Optional, Tuple

import torch
import torch.nn as nn


class RNNEncoder(nn.Module):
    """Standard RNN encoder for time series.

    Processes input sequence and outputs a latent representation.

    Args:
        input_dim: Input feature dimension.
        hidden_dim: RNN hidden dimension.
        latent_dim: Output latent dimension.
        n_layers: Number of RNN layers.
        rnn_type: Type of RNN ('gru' or 'lstm').
        bidirectional: Whether to use bidirectional RNN.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        latent_dim: int = 64,
        n_layers: int = 1,
        rnn_type: str = "gru",
        bidirectional: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.n_directions = 2 if bidirectional else 1

        # RNN layer
        if rnn_type.lower() == "gru":
            self.rnn = nn.GRU(
                input_dim,
                hidden_dim,
                n_layers,
                batch_first=True,
                bidirectional=bidirectional,
            )
        elif rnn_type.lower() == "lstm":
            self.rnn = nn.LSTM(
                input_dim,
                hidden_dim,
                n_layers,
                batch_first=True,
                bidirectional=bidirectional,
            )
        else:
            raise ValueError(f"Unknown RNN type: {rnn_type}")

        self.rnn_type = rnn_type.lower()

        # Project to latent space (mean and logvar for VAE)
        rnn_output_dim = hidden_dim * self.n_directions
        self.fc_mean = nn.Linear(rnn_output_dim, latent_dim)
        self.fc_logvar = nn.Linear(rnn_output_dim, latent_dim)

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input sequence to latent distribution parameters.

        Args:
            x: Input sequence [batch, seq_len, input_dim].
            lengths: Optional sequence lengths for packing.

        Returns:
            Tuple of (mean, logvar) for latent distribution.
        """
        batch_size = x.size(0)

        # Run RNN
        if lengths is not None:
            # Pack for variable length sequences
            x_packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, hidden = self.rnn(x_packed)
        else:
            _, hidden = self.rnn(x)

        # Extract final hidden state
        if self.rnn_type == "lstm":
            hidden = hidden[0]  # Use hidden state, not cell state

        # hidden: [n_layers * n_directions, batch, hidden_dim]
        # Take last layer
        if self.bidirectional:
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        else:
            hidden = hidden[-1]

        # Project to latent distribution parameters
        mean = self.fc_mean(hidden)
        logvar = self.fc_logvar(hidden)

        return mean, logvar


class ODERNNEncoder(nn.Module):
    """ODE-RNN encoder for irregularly sampled time series.

    Combines RNN updates at observation times with ODE evolution
    between observations.

    Args:
        input_dim: Input feature dimension.
        hidden_dim: Hidden state dimension.
        latent_dim: Output latent dimension.
        ode_func: ODE dynamics function.
        use_adjoint: Whether to use adjoint method.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        latent_dim: int = 64,
        ode_func: Optional[nn.Module] = None,
        use_adjoint: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.use_adjoint = use_adjoint

        # GRU cell for updates at observation times
        self.gru_cell = nn.GRUCell(input_dim, hidden_dim)

        # ODE function for evolution between observations
        if ode_func is None:
            from .ode_func import ODEFunc
            self.ode_func = ODEFunc(hidden_dim, hidden_dim)
        else:
            self.ode_func = ode_func

        # Project to latent distribution
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(
        self,
        x: torch.Tensor,
        times: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode irregularly sampled sequence.

        Args:
            x: Input observations [batch, seq_len, input_dim].
            times: Observation times [batch, seq_len] or [seq_len].

        Returns:
            Tuple of (mean, logvar) for latent distribution.
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        # Initialize hidden state
        h = torch.zeros(batch_size, self.hidden_dim, device=device)

        # Get ODE integrator
        if self.use_adjoint:
            from torchdiffeq import odeint_adjoint as odeint
        else:
            from torchdiffeq import odeint

        # Handle time tensor shape
        if times.dim() == 1:
            times = times.unsqueeze(0).expand(batch_size, -1)

        # Process sequence backwards (for encoding)
        for i in range(seq_len - 1, -1, -1):
            # ODE evolution from current time to previous time (if not first step)
            if i < seq_len - 1:
                t_span = torch.tensor(
                    [times[0, i + 1].item(), times[0, i].item()],
                    device=device
                )
                h = odeint(self.ode_func, h, t_span, method="dopri5")[-1]

            # GRU update at observation time
            h = self.gru_cell(x[:, i, :], h)

        # Project to latent distribution
        mean = self.fc_mean(h)
        logvar = self.fc_logvar(h)

        return mean, logvar
