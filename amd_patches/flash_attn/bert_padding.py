# Flash Attention bert_padding stub for AMD ROCm
import torch
from typing import Tuple

def unpad_input(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Remove padding from input sequences.
    
    Args:
        hidden_states: (batch, seqlen, dim)
        attention_mask: (batch, seqlen), 1 for valid tokens, 0 for padding
    
    Returns:
        hidden_states_unpad: (total_tokens, dim)
        indices: indices to reconstruct padded tensor
        cu_seqlens: cumulative sequence lengths
        max_seqlen: maximum sequence length
    """
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen = seqlens_in_batch.max().item()
    cu_seqlens = torch.nn.functional.pad(
        torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0)
    )
    hidden_states_unpad = hidden_states.view(-1, hidden_states.shape[-1])[indices]
    return hidden_states_unpad, indices, cu_seqlens, max_seqlen


def pad_input(hidden_states_unpad: torch.Tensor, indices: torch.Tensor, batch: int, seqlen: int) -> torch.Tensor:
    """
    Pad input sequences back to original shape.
    
    Args:
        hidden_states_unpad: (total_tokens, dim)
        indices: indices from unpad_input
        batch: batch size
        seqlen: sequence length
    
    Returns:
        hidden_states: (batch, seqlen, dim)
    """
    dim = hidden_states_unpad.shape[-1]
    output = torch.zeros(batch * seqlen, dim, device=hidden_states_unpad.device, dtype=hidden_states_unpad.dtype)
    output[indices] = hidden_states_unpad
    return output.view(batch, seqlen, dim)


def index_first_axis(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Index the first axis of a tensor."""
    return x[indices]


def index_put_first_axis(values: torch.Tensor, indices: torch.Tensor, first_axis_dim: int) -> torch.Tensor:
    """Put values at indices in the first axis."""
    output = torch.zeros(first_axis_dim, *values.shape[1:], device=values.device, dtype=values.dtype)
    output[indices] = values
    return output
