"""Latent ODE model for time series forecasting.

Paper: "Latent ODEs for Irregularly-Sampled Time Series" (NeurIPS 2019)

Key features:
- VAE-style architecture with ODE dynamics in latent space
- Handles both regular and irregular time series
- Memory-efficient training via adjoint method
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from ..base import BaseODEModel
from ..registry import register_model
from .components.ode_func import ODEFunc
from .components.encoder import RNNEncoder
from .components.decoder import MLPDecoder


@register_model("latent_ode")
class LatentODE(BaseODEModel):
    """Latent ODE model for time series forecasting.

    Encodes input sequence to initial latent state, evolves it through
    ODE dynamics, and decodes trajectory to predictions.

    Args:
        config: Configuration dictionary containing:
            - enc_in: Input dimension
            - c_out: Output dimension
            - seq_len: Input sequence length
            - pred_len: Prediction horizon
            - latent_dim: Latent space dimension (default: 64)
            - rec_dims: Recognition network hidden dimension (default: 64)
            - ode_dims: ODE function hidden dimension (default: 128)
            - dec_dims: Decoder hidden dimension (default: 64)
            - ode_solver: ODE solver method (default: 'dopri5')
            - ode_rtol: Relative tolerance (default: 1e-4)
            - ode_atol: Absolute tolerance (default: 1e-5)
            - use_adjoint: Use adjoint method (default: True)
    """

    def __init__(self, config: Dict):
        super().__init__(config)

        model_config = config.get("model", config)
        ode_config = config.get("ode", {})

        # Dimensions
        self.input_dim = config.get("enc_in", 7)
        self.output_dim = config.get("c_out", 7)
        self.latent_dim = model_config.get("latent_dim", 64)
        self.rec_dims = model_config.get("rec_dims", 64)
        self.ode_dims = model_config.get("ode_dims", 128)
        self.dec_dims = model_config.get("dec_dims", 64)

        # ODE settings
        self.solver = ode_config.get("solver", "dopri5")
        self.rtol = ode_config.get("rtol", 1e-4)
        self.atol = ode_config.get("atol", 1e-5)
        self.use_adjoint = ode_config.get("use_adjoint", True)

        # Encoder (recognition network)
        self.encoder = RNNEncoder(
            input_dim=self.input_dim,
            hidden_dim=self.rec_dims,
            latent_dim=self.latent_dim,
            n_layers=model_config.get("rec_layers", 2),
            rnn_type="gru",
        )

        # ODE dynamics in latent space
        self.ode_func = ODEFunc(
            latent_dim=self.latent_dim,
            hidden_dim=self.ode_dims,
            n_layers=model_config.get("ode_layers", 2),
        )

        # Decoder
        self.decoder = MLPDecoder(
            latent_dim=self.latent_dim,
            output_dim=self.output_dim,
            hidden_dim=self.dec_dims,
            n_layers=model_config.get("dec_layers", 2),
        )

    def get_ode_func(self) -> nn.Module:
        """Return the ODE dynamics function."""
        return self.ode_func

    def reparameterize(
        self,
        mean: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """Reparameterization trick for VAE.

        Args:
            mean: Mean of the latent distribution.
            logvar: Log variance of the latent distribution.

        Returns:
            Sampled latent vector.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor,
        x_dec: torch.Tensor,
        x_mark_dec: torch.Tensor,
        return_latent: bool = False,
    ) -> torch.Tensor:
        """Forward pass for time series forecasting.

        Args:
            x_enc: Encoder input [batch, seq_len, enc_in].
            x_mark_enc: Encoder time features (unused in this model).
            x_dec: Decoder input (unused, for API compatibility).
            x_mark_dec: Decoder time features (unused in this model).
            return_latent: Whether to return latent distribution params.

        Returns:
            If training: Tuple of (predictions, mean, logvar)
            If inference: predictions [batch, pred_len, c_out]
        """
        batch_size = x_enc.size(0)
        device = x_enc.device

        # Encode input sequence to latent distribution
        mean, logvar = self.encoder(x_enc)

        # Sample initial latent state
        if self.training:
            z0 = self.reparameterize(mean, logvar)
        else:
            z0 = mean  # Use mean for inference

        # Create time points for prediction
        # Normalize to [0, 1] range for numerical stability
        t = torch.linspace(0, 1, self.pred_len + 1, device=device)

        # Integrate ODE from initial state
        odeint = self.get_ode_integrator()
        z_traj = odeint(
            self.ode_func,
            z0,
            t,
            method=self.solver,
            rtol=self.rtol,
            atol=self.atol,
        )  # [pred_len + 1, batch, latent_dim]

        # Take prediction steps (skip initial state)
        z_pred = z_traj[1:]  # [pred_len, batch, latent_dim]

        # Decode to observations
        y_pred = self.decoder(z_pred)  # [pred_len, batch, output_dim]

        # Transpose to [batch, pred_len, output_dim]
        y_pred = y_pred.transpose(0, 1)

        if self.training or return_latent:
            return y_pred, mean, logvar
        return y_pred

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution.

        Args:
            x: Input sequence [batch, seq_len, input_dim].

        Returns:
            Tuple of (mean, logvar).
        """
        return self.encoder(x)

    def decode(
        self,
        z0: torch.Tensor,
        num_steps: int,
    ) -> torch.Tensor:
        """Decode from latent state.

        Args:
            z0: Initial latent state [batch, latent_dim].
            num_steps: Number of prediction steps.

        Returns:
            Predictions [batch, num_steps, output_dim].
        """
        device = z0.device
        t = torch.linspace(0, 1, num_steps + 1, device=device)

        odeint = self.get_ode_integrator()
        z_traj = odeint(
            self.ode_func,
            z0,
            t,
            method=self.solver,
            rtol=self.rtol,
            atol=self.atol,
        )

        z_pred = z_traj[1:]
        y_pred = self.decoder(z_pred)
        return y_pred.transpose(0, 1)
