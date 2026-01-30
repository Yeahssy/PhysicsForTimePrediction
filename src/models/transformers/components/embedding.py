"""Embedding layers for time series transformers."""

import math
from typing import Optional

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformers.

    Args:
        d_model: Model dimension.
        max_len: Maximum sequence length.
        dropout: Dropout probability.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.

        Args:
            x: Input tensor [batch, seq_len, d_model].

        Returns:
            Tensor with positional encoding added.
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TokenEmbedding(nn.Module):
    """Embed input tokens using 1D convolution.

    Args:
        c_in: Number of input channels (features).
        d_model: Model dimension.
    """

    def __init__(self, c_in: int, d_model: int):
        super().__init__()
        padding = 1
        self.conv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=padding,
            padding_mode="circular",
            bias=False,
        )

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed input tokens.

        Args:
            x: Input tensor [batch, seq_len, c_in].

        Returns:
            Embedded tensor [batch, seq_len, d_model].
        """
        # Conv1d expects [batch, channels, seq_len]
        x = self.conv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class TemporalEmbedding(nn.Module):
    """Embed temporal features (hour, day, week, month).

    Args:
        d_model: Model dimension.
        embed_type: Embedding type ('fixed' or 'timeF').
        freq: Time frequency for determining feature set.
    """

    def __init__(self, d_model: int, embed_type: str = "fixed", freq: str = "h"):
        super().__init__()
        self.embed_type = embed_type

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        if embed_type == "fixed":
            # Fixed embeddings for each temporal component
            if freq in ["t", "min"]:
                self.minute_embed = nn.Embedding(minute_size, d_model)
            self.hour_embed = nn.Embedding(hour_size, d_model)
            self.weekday_embed = nn.Embedding(weekday_size, d_model)
            self.day_embed = nn.Embedding(day_size, d_model)
            self.month_embed = nn.Embedding(month_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed temporal features.

        Args:
            x: Time features [batch, seq_len, num_features].

        Returns:
            Temporal embedding [batch, seq_len, d_model].
        """
        if self.embed_type == "fixed":
            # Assume x contains [hour, weekday, day, month, ...]
            x = x.long()
            hour_x = self.hour_embed(x[:, :, 0])
            weekday_x = self.weekday_embed(x[:, :, 1])
            day_x = self.day_embed(x[:, :, 2])
            month_x = self.month_embed(x[:, :, 3])
            return hour_x + weekday_x + day_x + month_x
        else:
            # For timeF, x is already continuous features
            return x


class TimeFeatureEmbedding(nn.Module):
    """Linear projection of time features.

    Args:
        d_inp: Input dimension (number of time features).
        d_model: Model dimension.
    """

    def __init__(self, d_inp: int, d_model: int):
        super().__init__()
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project time features.

        Args:
            x: Time features [batch, seq_len, d_inp].

        Returns:
            Projected features [batch, seq_len, d_model].
        """
        return self.embed(x)


class DataEmbedding(nn.Module):
    """Combined embedding for time series data.

    Combines token embedding, positional encoding, and temporal embedding.

    Args:
        c_in: Number of input features.
        d_model: Model dimension.
        embed_type: Temporal embedding type ('fixed', 'timeF').
        freq: Time frequency.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        c_in: int,
        d_model: int,
        embed_type: str = "timeF",
        freq: str = "h",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.value_embedding = TokenEmbedding(c_in, d_model)
        self.position_embedding = PositionalEncoding(d_model, dropout=dropout)

        if embed_type == "timeF":
            # Number of time features depends on frequency
            freq_map = {"h": 4, "d": 3, "w": 2, "m": 1, "t": 5, "min": 5}
            d_inp = freq_map.get(freq, 4)
            self.temporal_embedding = TimeFeatureEmbedding(d_inp, d_model)
        else:
            self.temporal_embedding = TemporalEmbedding(d_model, embed_type, freq)

        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        x_mark: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Embed input data.

        Args:
            x: Input values [batch, seq_len, c_in].
            x_mark: Time features [batch, seq_len, time_features].

        Returns:
            Combined embedding [batch, seq_len, d_model].
        """
        # Value embedding
        x_embed = self.value_embedding(x)

        # Add positional encoding
        x_embed = self.position_embedding(x_embed)

        # Add temporal embedding if provided
        if x_mark is not None:
            x_embed = x_embed + self.temporal_embedding(x_mark)

        return self.dropout(x_embed)


class DataEmbeddingWithoutPosition(nn.Module):
    """Data embedding without positional encoding.

    Useful for models that handle position differently (e.g., through attention).

    Args:
        c_in: Number of input features.
        d_model: Model dimension.
        embed_type: Temporal embedding type.
        freq: Time frequency.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        c_in: int,
        d_model: int,
        embed_type: str = "timeF",
        freq: str = "h",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.value_embedding = TokenEmbedding(c_in, d_model)

        if embed_type == "timeF":
            freq_map = {"h": 4, "d": 3, "w": 2, "m": 1, "t": 5, "min": 5}
            d_inp = freq_map.get(freq, 4)
            self.temporal_embedding = TimeFeatureEmbedding(d_inp, d_model)
        else:
            self.temporal_embedding = TemporalEmbedding(d_model, embed_type, freq)

        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        x_mark: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Embed input data without positional encoding.

        Args:
            x: Input values [batch, seq_len, c_in].
            x_mark: Time features [batch, seq_len, time_features].

        Returns:
            Combined embedding [batch, seq_len, d_model].
        """
        x_embed = self.value_embedding(x)

        if x_mark is not None:
            x_embed = x_embed + self.temporal_embedding(x_mark)

        return self.dropout(x_embed)
