"""Attention mechanisms for time series transformers."""

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FullAttention(nn.Module):
    """Standard multi-head self-attention.

    Args:
        mask_flag: Whether to apply causal mask.
        factor: Unused (for API compatibility with ProbAttention).
        attention_dropout: Dropout probability for attention weights.
        output_attention: Whether to return attention weights.
    """

    def __init__(
        self,
        mask_flag: bool = True,
        factor: int = 5,
        attention_dropout: float = 0.1,
        output_attention: bool = False,
    ):
        super().__init__()
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute full attention.

        Args:
            queries: Query tensor [batch, heads, seq_len_q, d_k].
            keys: Key tensor [batch, heads, seq_len_k, d_k].
            values: Value tensor [batch, heads, seq_len_v, d_v].
            attn_mask: Optional attention mask.

        Returns:
            Tuple of (output, attention_weights or None).
        """
        B, H, L_Q, D = queries.shape
        _, _, L_K, _ = keys.shape
        scale = 1.0 / math.sqrt(D)

        # Compute attention scores
        scores = torch.matmul(queries, keys.transpose(-2, -1)) * scale

        # Apply mask if provided
        if self.mask_flag and attn_mask is not None:
            scores.masked_fill_(attn_mask, float("-inf"))

        # Softmax and dropout
        attn = self.dropout(F.softmax(scores, dim=-1))

        # Compute output
        output = torch.matmul(attn, values)

        if self.output_attention:
            return output, attn
        return output, None


class ProbAttention(nn.Module):
    """ProbSparse self-attention from Informer.

    Reduces complexity from O(L^2) to O(L log L) by selecting
    the top-k queries based on sparsity measurement.

    Args:
        mask_flag: Whether to apply causal mask.
        factor: Sampling factor for ProbSparse attention.
        attention_dropout: Dropout probability.
        output_attention: Whether to return attention weights.
    """

    def __init__(
        self,
        mask_flag: bool = True,
        factor: int = 5,
        attention_dropout: float = 0.1,
        output_attention: bool = False,
    ):
        super().__init__()
        self.mask_flag = mask_flag
        self.factor = factor
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        sample_k: int,
        n_top: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate the sparse Q-K attention.

        Args:
            Q: Query tensor [batch, heads, seq_len_q, d_k].
            K: Key tensor [batch, heads, seq_len_k, d_k].
            sample_k: Number of keys to sample.
            n_top: Number of top queries to select.

        Returns:
            Tuple of (top_scores, top_indices).
        """
        B, H, L_K, D = K.shape
        _, _, L_Q, _ = Q.shape

        # Sample keys
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, D)
        index_sample = torch.randint(L_K, (L_Q, sample_k), device=K.device)
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]

        # Compute Q-K scores for sampled keys
        Q_K_sample = torch.matmul(
            Q.unsqueeze(-2), K_sample.transpose(-2, -1)
        ).squeeze(-2)

        # Find sparsity measurement (max - mean)
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)

        # Select top-n_top queries
        M_top = M.topk(n_top, sorted=False)[1]

        # Compute full attention for selected queries
        Q_reduce = Q[
            torch.arange(B)[:, None, None],
            torch.arange(H)[None, :, None],
            M_top,
            :,
        ]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))

        return Q_K, M_top

    def _get_initial_context(
        self,
        V: torch.Tensor,
        L_Q: int,
    ) -> torch.Tensor:
        """Get initial context using cumulative mean of values.

        Args:
            V: Value tensor [batch, heads, seq_len_v, d_v].
            L_Q: Query sequence length.

        Returns:
            Initial context tensor.
        """
        B, H, L_V, D = V.shape

        if not self.mask_flag:
            # Use mean of all values
            V_mean = V.mean(dim=-2)
            context = V_mean.unsqueeze(-2).expand(B, H, L_Q, D).clone()
        else:
            # Use cumulative mean for causal attention
            context = V.cumsum(dim=-2)
            context = context / torch.arange(1, L_V + 1, device=V.device).view(1, 1, -1, 1)

        return context

    def _update_context(
        self,
        context: torch.Tensor,
        V: torch.Tensor,
        scores: torch.Tensor,
        index: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update context with attended values.

        Args:
            context: Current context tensor.
            V: Value tensor.
            scores: Attention scores for top queries.
            index: Indices of top queries.
            attn_mask: Optional attention mask.

        Returns:
            Tuple of (updated_context, attention_weights).
        """
        B, H, L_V, D = V.shape

        if self.mask_flag and attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1)  # [B, 1, L_Q, L_K]
            scores.masked_fill_(attn_mask, float("-inf"))

        attn = self.dropout(F.softmax(scores, dim=-1))
        context[
            torch.arange(B)[:, None, None],
            torch.arange(H)[None, :, None],
            index,
            :,
        ] = torch.matmul(attn, V)

        if self.output_attention:
            # Create full attention matrix
            attentions = torch.zeros(B, H, context.shape[2], L_V, device=context.device)
            attentions[
                torch.arange(B)[:, None, None],
                torch.arange(H)[None, :, None],
                index,
                :,
            ] = attn
            return context, attentions

        return context, None

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute ProbSparse attention.

        Args:
            queries: Query tensor [batch, heads, seq_len_q, d_k].
            keys: Key tensor [batch, heads, seq_len_k, d_k].
            values: Value tensor [batch, heads, seq_len_v, d_v].
            attn_mask: Optional attention mask.

        Returns:
            Tuple of (output, attention_weights or None).
        """
        B, H, L_Q, D = queries.shape
        _, _, L_K, _ = keys.shape

        # Calculate sample sizes
        U_part = self.factor * int(np.ceil(np.log(L_K)))
        u = self.factor * int(np.ceil(np.log(L_Q)))

        U_part = min(U_part, L_K)
        u = min(u, L_Q)

        # Get sparse attention scores
        scores_top, index = self._prob_QK(queries, keys, U_part, u)
        scale = 1.0 / math.sqrt(D)
        scores_top = scores_top * scale

        # Get initial context
        context = self._get_initial_context(values, L_Q)

        # Update context with attended values
        context, attn = self._update_context(
            context, values, scores_top, index, attn_mask
        )

        return context, attn


class AttentionLayer(nn.Module):
    """Multi-head attention layer wrapper.

    Args:
        attention: Attention mechanism (FullAttention or ProbAttention).
        d_model: Model dimension.
        n_heads: Number of attention heads.
        d_keys: Key dimension (defaults to d_model // n_heads).
        d_values: Value dimension (defaults to d_model // n_heads).
    """

    def __init__(
        self,
        attention: nn.Module,
        d_model: int,
        n_heads: int,
        d_keys: Optional[int] = None,
        d_values: Optional[int] = None,
    ):
        super().__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.n_heads = n_heads

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply multi-head attention.

        Args:
            queries: Query tensor [batch, seq_len_q, d_model].
            keys: Key tensor [batch, seq_len_k, d_model].
            values: Value tensor [batch, seq_len_v, d_model].
            attn_mask: Optional attention mask.

        Returns:
            Tuple of (output, attention_weights or None).
        """
        B, L_Q, _ = queries.shape
        _, L_K, _ = keys.shape
        H = self.n_heads

        # Project Q, K, V
        queries = self.query_projection(queries).view(B, L_Q, H, -1).transpose(1, 2)
        keys = self.key_projection(keys).view(B, L_K, H, -1).transpose(1, 2)
        values = self.value_projection(values).view(B, L_K, H, -1).transpose(1, 2)

        # Apply attention
        out, attn = self.inner_attention(queries, keys, values, attn_mask)

        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(B, L_Q, -1)
        out = self.out_projection(out)

        return out, attn
