"""Informer model for long sequence time series forecasting.

Paper: "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting"
       (AAAI 2021 Best Paper)

Key features:
- ProbSparse self-attention: O(L log L) complexity
- Self-attention distilling: Reduces encoder depth
- Generative decoder: One-forward prediction
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from ..base import BaseModel
from ..registry import register_model
from .components.embedding import DataEmbedding
from .components.attention import (
    FullAttention,
    ProbAttention,
    AttentionLayer,
)
from .components.decomposition import EncoderLayer, DecoderLayer


class ConvLayer(nn.Module):
    """Distilling layer using convolution for sequence compression.

    Args:
        c_in: Number of input channels.
    """

    def __init__(self, c_in: int):
        super().__init__()
        self.down_conv = nn.Conv1d(
            in_channels=c_in,
            out_channels=c_in,
            kernel_size=3,
            padding=2,
            padding_mode="circular",
        )
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply distilling convolution.

        Args:
            x: Input tensor [batch, seq_len, d_model].

        Returns:
            Downsampled tensor [batch, seq_len // 2, d_model].
        """
        x = self.down_conv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.max_pool(x)
        x = x.transpose(1, 2)
        return x


class InformerEncoder(nn.Module):
    """Informer encoder with optional distilling.

    Args:
        attn_layers: List of encoder attention layers.
        conv_layers: List of distilling convolution layers.
        norm_layer: Optional normalization layer.
    """

    def __init__(
        self,
        attn_layers: nn.ModuleList,
        conv_layers: Optional[nn.ModuleList] = None,
        norm_layer: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.attn_layers = attn_layers
        self.conv_layers = conv_layers
        self.norm = norm_layer

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, list]:
        """Encode input sequence.

        Args:
            x: Input tensor [batch, seq_len, d_model].
            attn_mask: Optional attention mask.

        Returns:
            Tuple of (encoded output, list of attention weights).
        """
        attns = []

        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            # Last attention layer without conv
            x, attn = self.attn_layers[-1](x, attn_mask=attn_mask)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class InformerDecoder(nn.Module):
    """Informer decoder.

    Args:
        layers: List of decoder layers.
        norm_layer: Optional normalization layer.
        projection: Optional output projection layer.
    """

    def __init__(
        self,
        layers: nn.ModuleList,
        norm_layer: Optional[nn.Module] = None,
        projection: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.layers = layers
        self.norm = norm_layer
        self.projection = projection

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decode with encoder memory.

        Args:
            x: Decoder input [batch, seq_len, d_model].
            memory: Encoder output [batch, seq_len, d_model].
            self_attn_mask: Mask for decoder self-attention.
            cross_attn_mask: Mask for cross-attention.

        Returns:
            Decoder output.
        """
        for layer in self.layers:
            x = layer(x, memory, self_attn_mask, cross_attn_mask)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)

        return x


@register_model("informer")
class Informer(BaseModel):
    """Informer model for time series forecasting.

    Uses ProbSparse self-attention for O(L log L) complexity and
    self-attention distilling for efficient encoding of long sequences.

    Args:
        config: Configuration dictionary containing:
            - enc_in: Encoder input dimension
            - dec_in: Decoder input dimension
            - c_out: Output dimension
            - seq_len: Input sequence length
            - label_len: Start token length for decoder
            - pred_len: Prediction horizon
            - d_model: Model dimension (default: 512)
            - n_heads: Number of attention heads (default: 8)
            - e_layers: Number of encoder layers (default: 2)
            - d_layers: Number of decoder layers (default: 1)
            - d_ff: Feed-forward dimension (default: 2048)
            - factor: ProbSparse sampling factor (default: 5)
            - attn: Attention type 'prob' or 'full' (default: 'prob')
            - dropout: Dropout probability (default: 0.05)
            - embed: Embedding type (default: 'timeF')
            - freq: Time frequency (default: 'h')
            - activation: Activation function (default: 'gelu')
            - distil: Whether to use distilling (default: True)
            - output_attention: Whether to output attention weights
    """

    def __init__(self, config: Dict):
        super().__init__(config)

        # Model parameters from config
        model_config = config.get("model", config)

        self.enc_in = config.get("enc_in", 7)
        self.dec_in = config.get("dec_in", 7)
        self.c_out = config.get("c_out", 7)

        self.d_model = model_config.get("d_model", 512)
        self.n_heads = model_config.get("n_heads", 8)
        self.e_layers = model_config.get("e_layers", 2)
        self.d_layers = model_config.get("d_layers", 1)
        self.d_ff = model_config.get("d_ff", 2048)
        self.factor = model_config.get("factor", 5)
        self.attn_type = model_config.get("attn", "prob")
        self.dropout = model_config.get("dropout", 0.05)
        self.embed = model_config.get("embed", "timeF")
        self.freq = config.get("data", {}).get("freq", "h")
        self.activation = model_config.get("activation", "gelu")
        self.distil = model_config.get("distil", True)
        self.output_attention = model_config.get("output_attention", False)

        # Embeddings
        self.enc_embedding = DataEmbedding(
            self.enc_in, self.d_model, self.embed, self.freq, self.dropout
        )
        self.dec_embedding = DataEmbedding(
            self.dec_in, self.d_model, self.embed, self.freq, self.dropout
        )

        # Select attention mechanism
        Attn = ProbAttention if self.attn_type == "prob" else FullAttention

        # Encoder
        encoder_layers = nn.ModuleList([
            EncoderLayer(
                AttentionLayer(
                    Attn(
                        mask_flag=False,
                        factor=self.factor,
                        attention_dropout=self.dropout,
                        output_attention=self.output_attention,
                    ),
                    self.d_model,
                    self.n_heads,
                ),
                self.d_model,
                self.d_ff,
                dropout=self.dropout,
                activation=self.activation,
            )
            for _ in range(self.e_layers)
        ])

        # Distilling layers
        if self.distil:
            conv_layers = nn.ModuleList([
                ConvLayer(self.d_model)
                for _ in range(self.e_layers - 1)
            ])
        else:
            conv_layers = None

        self.encoder = InformerEncoder(
            encoder_layers,
            conv_layers,
            norm_layer=nn.LayerNorm(self.d_model),
        )

        # Decoder
        decoder_layers = nn.ModuleList([
            DecoderLayer(
                AttentionLayer(
                    Attn(
                        mask_flag=True,
                        factor=self.factor,
                        attention_dropout=self.dropout,
                        output_attention=False,
                    ),
                    self.d_model,
                    self.n_heads,
                ),
                AttentionLayer(
                    FullAttention(
                        mask_flag=False,
                        factor=self.factor,
                        attention_dropout=self.dropout,
                        output_attention=False,
                    ),
                    self.d_model,
                    self.n_heads,
                ),
                self.d_model,
                self.d_ff,
                dropout=self.dropout,
                activation=self.activation,
            )
            for _ in range(self.d_layers)
        ])

        self.decoder = InformerDecoder(
            decoder_layers,
            norm_layer=nn.LayerNorm(self.d_model),
            projection=nn.Linear(self.d_model, self.c_out, bias=True),
        )

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
            x_mark_enc: Encoder time features [batch, seq_len, time_features].
            x_dec: Decoder input [batch, label_len + pred_len, dec_in].
            x_mark_dec: Decoder time features [batch, label_len + pred_len, time_features].

        Returns:
            Predictions [batch, pred_len, c_out].
        """
        # Embed encoder input
        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        # Encode
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Embed decoder input
        dec_out = self.dec_embedding(x_dec, x_mark_dec)

        # Decode
        dec_out = self.decoder(dec_out, enc_out)

        # Return only the prediction part (last pred_len steps)
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        return dec_out[:, -self.pred_len:, :]
