"""Series decomposition components for Autoformer."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MovingAverage(nn.Module):
    """Moving average for trend extraction.

    Args:
        kernel_size: Size of the moving average kernel.
        stride: Stride for the convolution.
    """

    def __init__(self, kernel_size: int, stride: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply moving average.

        Args:
            x: Input tensor [batch, seq_len, features].

        Returns:
            Smoothed tensor [batch, seq_len, features].
        """
        # Pad to maintain sequence length
        front = x[:, :1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)

        # Apply average pooling (expects [batch, features, seq_len])
        x = x.permute(0, 2, 1)
        x = self.avg(x)
        x = x.permute(0, 2, 1)

        return x


class SeriesDecomposition(nn.Module):
    """Series decomposition into trend and seasonal components.

    Used in Autoformer for progressive decomposition during forecasting.

    Args:
        kernel_size: Kernel size for moving average (trend extraction).
    """

    def __init__(self, kernel_size: int = 25):
        super().__init__()
        self.moving_avg = MovingAverage(kernel_size, stride=1)

    def forward(self, x: torch.Tensor) -> tuple:
        """Decompose series into seasonal and trend components.

        Args:
            x: Input tensor [batch, seq_len, features].

        Returns:
            Tuple of (seasonal, trend) components.
        """
        # Extract trend using moving average
        trend = self.moving_avg(x)

        # Seasonal = original - trend
        seasonal = x - trend

        return seasonal, trend


class FeedForward(nn.Module):
    """Position-wise feed-forward network.

    Args:
        d_model: Model dimension.
        d_ff: Feed-forward dimension.
        dropout: Dropout probability.
        activation: Activation function ('relu' or 'gelu').
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            self.activation = F.relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feed-forward transformation.

        Args:
            x: Input tensor [batch, seq_len, d_model].

        Returns:
            Transformed tensor [batch, seq_len, d_model].
        """
        # Conv1d expects [batch, channels, seq_len]
        x = x.transpose(1, 2)
        x = self.dropout(self.activation(self.conv1(x)))
        x = self.conv2(x)
        x = x.transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
    """Transformer encoder layer.

    Args:
        attention: Attention mechanism.
        d_model: Model dimension.
        d_ff: Feed-forward dimension.
        dropout: Dropout probability.
        activation: Activation function.
    """

    def __init__(
        self,
        attention: nn.Module,
        d_model: int,
        d_ff: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()
        self.attention = attention
        self.feed_forward = FeedForward(d_model, d_ff, dropout, activation)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor = None,
    ) -> tuple:
        """Apply encoder layer.

        Args:
            x: Input tensor [batch, seq_len, d_model].
            attn_mask: Optional attention mask.

        Returns:
            Tuple of (output, attention_weights).
        """
        # Self-attention with residual
        attn_out, attn = self.attention(x, x, x, attn_mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # Feed-forward with residual
        ff_out = self.feed_forward(x)
        x = x + self.dropout(ff_out)
        x = self.norm2(x)

        return x, attn


class DecoderLayer(nn.Module):
    """Transformer decoder layer.

    Args:
        self_attention: Self-attention mechanism.
        cross_attention: Cross-attention mechanism.
        d_model: Model dimension.
        d_ff: Feed-forward dimension.
        dropout: Dropout probability.
        activation: Activation function.
    """

    def __init__(
        self,
        self_attention: nn.Module,
        cross_attention: nn.Module,
        d_model: int,
        d_ff: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.feed_forward = FeedForward(d_model, d_ff, dropout, activation)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        self_attn_mask: torch.Tensor = None,
        cross_attn_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Apply decoder layer.

        Args:
            x: Decoder input [batch, seq_len, d_model].
            memory: Encoder output [batch, seq_len, d_model].
            self_attn_mask: Mask for self-attention.
            cross_attn_mask: Mask for cross-attention.

        Returns:
            Decoder output.
        """
        # Self-attention
        self_attn_out, _ = self.self_attention(x, x, x, self_attn_mask)
        x = x + self.dropout(self_attn_out)
        x = self.norm1(x)

        # Cross-attention
        cross_attn_out, _ = self.cross_attention(x, memory, memory, cross_attn_mask)
        x = x + self.dropout(cross_attn_out)
        x = self.norm2(x)

        # Feed-forward
        ff_out = self.feed_forward(x)
        x = x + self.dropout(ff_out)
        x = self.norm3(x)

        return x
