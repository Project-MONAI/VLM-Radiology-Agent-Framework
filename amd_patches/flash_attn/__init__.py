# Flash Attention stub for AMD ROCm
# This provides compatibility when flash_attn is not available
# Uses standard PyTorch attention as fallback

import torch
import torch.nn.functional as F

__version__ = "2.0.0.stub"

def _attention_forward(q, k, v, causal=False, softmax_scale=None):
    """Standard attention fallback."""
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    
    scores = torch.matmul(q, k.transpose(-2, -1)) * softmax_scale
    
    if causal:
        mask = torch.triu(torch.ones(scores.shape[-2:], device=scores.device, dtype=torch.bool), diagonal=1)
        scores.masked_fill_(mask, float('-inf'))
    
    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, v)
