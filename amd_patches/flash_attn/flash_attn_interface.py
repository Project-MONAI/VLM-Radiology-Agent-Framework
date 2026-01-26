# Flash Attention interface stub for AMD ROCm
# Provides API compatibility when flash_attn is not available

import torch
import torch.nn.functional as F
from typing import Optional, Tuple

def flash_attn_func(
    q, k, v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    return_attn_probs=False,
):
    """Flash attention function stub using standard PyTorch attention."""
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    
    # Standard attention computation
    scores = torch.matmul(q, k.transpose(-2, -1)) * softmax_scale
    
    if causal:
        seq_len = scores.shape[-1]
        mask = torch.triu(torch.ones(seq_len, seq_len, device=scores.device, dtype=torch.bool), diagonal=1)
        scores.masked_fill_(mask, float('-inf'))
    
    attn_weights = F.softmax(scores, dim=-1)
    
    if dropout_p > 0.0 and torch.is_grad_enabled():
        attn_weights = F.dropout(attn_weights, p=dropout_p)
    
    output = torch.matmul(attn_weights, v)
    
    if return_attn_probs:
        return output, attn_weights
    return output


def flash_attn_varlen_func(
    q, k, v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    return_attn_probs=False,
):
    """Variable length flash attention stub."""
    return flash_attn_func(q, k, v, dropout_p, softmax_scale, causal, return_attn_probs)


def flash_attn_qkvpacked_func(
    qkv,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    return_attn_probs=False,
):
    """QKV packed flash attention stub."""
    q, k, v = qkv.unbind(dim=2)
    return flash_attn_func(q, k, v, dropout_p, softmax_scale, causal, return_attn_probs)


def flash_attn_varlen_qkvpacked_func(
    qkv,
    cu_seqlens,
    max_seqlen,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    return_attn_probs=False,
):
    """Variable length QKV packed flash attention stub."""
    q, k, v = qkv.unbind(dim=2)
    return flash_attn_func(q, k, v, dropout_p, softmax_scale, causal, return_attn_probs)


# Aliases for compatibility
flash_attn_unpadded_qkvpacked_func = flash_attn_varlen_qkvpacked_func
flash_attn_unpadded_func = flash_attn_varlen_func
