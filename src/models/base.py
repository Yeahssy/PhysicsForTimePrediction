"""Base model classes for time series forecasting."""

from abc import ABC, abstractmethod
from typing import Dict, Optional

import torch
import torch.nn as nn


class BaseModel(ABC, nn.Module):
    """Abstract base class for all time series forecasting models.

    All models must implement the forward method with the standardized
    signature for encoder-decoder style forecasting.

    Args:
        config: Configuration dictionary containing model hyperparameters.
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        # Sequence parameters
        self.seq_len = config.get("seq_len", 96)
        self.label_len = config.get("label_len", 48)
        self.pred_len = config.get("pred_len", 96)

        # Feature dimensions
        self.enc_in = config.get("enc_in", 7)
        self.dec_in = config.get("dec_in", 7)
        self.c_out = config.get("c_out", 7)

        # Model dimensions
        self.d_model = config.get("d_model", 512)
        self.dropout = config.get("dropout", 0.1)

    @abstractmethod
    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor,
        x_dec: torch.Tensor,
        x_mark_dec: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for time series forecasting.

        Args:
            x_enc: Encoder input sequence [batch, seq_len, enc_in]
            x_mark_enc: Encoder temporal features [batch, seq_len, time_features]
            x_dec: Decoder input sequence [batch, label_len + pred_len, dec_in]
            x_mark_dec: Decoder temporal features [batch, label_len + pred_len, time_features]

        Returns:
            Predictions [batch, pred_len, c_out]
        """
        pass

    @property
    def model_type(self) -> str:
        """Return model type for training loop selection."""
        return "standard"

    def get_parameter_count(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_parameter_summary(self) -> Dict[str, int]:
        """Return parameter counts by module."""
        summary = {}
        for name, module in self.named_children():
            count = sum(p.numel() for p in module.parameters() if p.requires_grad)
            summary[name] = count
        return summary


class BaseODEModel(BaseModel):
    """Base class for ODE-based models requiring adjoint training.

    Extends BaseModel with ODE-specific configurations for solver settings
    and adjoint method support.

    Important:
        - ODE dynamics functions must use smooth activations (Tanh, Softplus)
        - Avoid ReLU/LeakyReLU for Lipschitz continuity
        - Use odeint_adjoint for O(1) memory during training

    Args:
        config: Configuration dictionary containing model and ODE hyperparameters.
    """

    def __init__(self, config: Dict):
        super().__init__(config)

        # ODE solver settings
        self.solver = config.get("ode_solver", "dopri5")
        self.rtol = config.get("ode_rtol", 1e-4)
        self.atol = config.get("ode_atol", 1e-5)
        self.use_adjoint = config.get("use_adjoint", True)

        # Latent space dimensions
        self.latent_dim = config.get("latent_dim", 64)

    @abstractmethod
    def get_ode_func(self) -> nn.Module:
        """Return the ODE dynamics function module.

        Required for adjoint method to collect parameters.
        The returned module must be an nn.Module subclass.

        Returns:
            ODE function f(t, z) where dz/dt = f(t, z)
        """
        pass

    @property
    def model_type(self) -> str:
        """Return model type for training loop selection."""
        return "ode"

    def get_ode_integrator(self):
        """Return the appropriate ODE integrator based on adjoint setting."""
        if self.use_adjoint:
            from torchdiffeq import odeint_adjoint as odeint
        else:
            from torchdiffeq import odeint
        return odeint

    def integrate(
        self,
        func: nn.Module,
        y0: torch.Tensor,
        t: torch.Tensor,
        method: Optional[str] = None,
    ) -> torch.Tensor:
        """Integrate ODE from initial state y0 over time points t.

        Args:
            func: ODE dynamics function f(t, y)
            y0: Initial state [batch, latent_dim]
            t: Time points to evaluate [num_times]
            method: ODE solver method (defaults to self.solver)

        Returns:
            Solution at time points [num_times, batch, latent_dim]
        """
        odeint = self.get_ode_integrator()
        return odeint(
            func,
            y0,
            t,
            method=method or self.solver,
            rtol=self.rtol,
            atol=self.atol,
        )
