"""ODE-RNN model for time series forecasting.

Combines RNN updates at observation times with continuous ODE dynamics
between observations. Simpler than Latent ODE but effective for regular
time series.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn

from ..base import BaseODEModel
from ..registry import register_model
from .components.ode_func import ODEFunc
from .components.decoder import MLPDecoder


@register_model("ode_rnn")
class ODERNN(BaseODEModel):
    """ODE-RNN model for time series forecasting.

    Uses ODE dynamics to evolve hidden state with GRU updates
    at observation times.

    Args:
        config: Configuration dictionary.
    """

    def __init__(self, config: Dict):
        super().__init__(config)

        model_config = config.get("model", config)
        ode_config = config.get("ode", {})

        # Dimensions
        self.input_dim = config.get("enc_in", 7)
        self.output_dim = config.get("c_out", 7)
        self.hidden_dim = model_config.get("hidden_dim", 64)

        # ODE settings
        self.solver = ode_config.get("solver", "dopri5")
        self.rtol = ode_config.get("rtol", 1e-4)
        self.atol = ode_config.get("atol", 1e-5)
        self.use_adjoint = ode_config.get("use_adjoint", True)

        # Input projection
        self.input_proj = nn.Linear(self.input_dim, self.hidden_dim)

        # GRU cell for updates
        self.gru_cell = nn.GRUCell(self.hidden_dim, self.hidden_dim)

        # ODE dynamics
        self.ode_func = ODEFunc(
            latent_dim=self.hidden_dim,
            hidden_dim=model_config.get("ode_dims", 128),
            n_layers=model_config.get("ode_layers", 2),
        )

        # Output decoder
        self.decoder = MLPDecoder(
            latent_dim=self.hidden_dim,
            output_dim=self.output_dim,
            hidden_dim=model_config.get("dec_dims", 64),
            n_layers=model_config.get("dec_layers", 2),
        )

    def get_ode_func(self) -> nn.Module:
        """Return the ODE dynamics function."""
        return self.ode_func

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor,
        x_dec: torch.Tensor,
        x_mark_dec: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for time series forecasting.

        Args:
            x_enc: Encoder input [batch, seq_len, enc_in].
            x_mark_enc: Encoder time features (unused).
            x_dec: Decoder input (unused).
            x_mark_dec: Decoder time features (unused).

        Returns:
            Predictions [batch, pred_len, c_out].
        """
        batch_size, seq_len, _ = x_enc.shape
        device = x_enc.device

        # Initialize hidden state
        h = torch.zeros(batch_size, self.hidden_dim, device=device)

        # Get ODE integrator
        odeint = self.get_ode_integrator()

        # Process encoder sequence
        x_proj = self.input_proj(x_enc)  # [batch, seq_len, hidden_dim]

        for t in range(seq_len):
            # GRU update with observation
            h = self.gru_cell(x_proj[:, t, :], h)

            # ODE evolution to next time step (if not last step)
            if t < seq_len - 1:
                t_span = torch.tensor([0.0, 1.0], device=device)
                h = odeint(
                    self.ode_func,
                    h,
                    t_span,
                    method=self.solver,
                    rtol=self.rtol,
                    atol=self.atol,
                )[-1]

        # Generate predictions by evolving ODE
        t_pred = torch.linspace(0, 1, self.pred_len + 1, device=device)
        h_traj = odeint(
            self.ode_func,
            h,
            t_pred,
            method=self.solver,
            rtol=self.rtol,
            atol=self.atol,
        )  # [pred_len + 1, batch, hidden_dim]

        # Decode predictions (skip initial state)
        h_pred = h_traj[1:]  # [pred_len, batch, hidden_dim]
        y_pred = self.decoder(h_pred)  # [pred_len, batch, output_dim]

        # Transpose to [batch, pred_len, output_dim]
        return y_pred.transpose(0, 1)
