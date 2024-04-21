# This script intends to exercise the forward and backward of
# scaled dot product attention without using automatic 
# differentiation

import math
import torch
import torch.nn.functional as F
import pytest

def softmax_fwd(p):
    p_max = torch.max(p, dim=-1), keepdim=True).values
    p_safe = p - p_max
    p_safe = torch.exp(p_safe)
    p_safe_sum = torch.sum(p_safe, dim=-1, keepdim=True)
    return p_safe / p_safe_sum


def softmax_bwd(ds):
    pass


def scaled_dot_product_attention_fwd(q, k, v):
    '''
    q: [b, h, s, d]
    k: [b, h, s, d]
    v: [b, h, s, d]
    '''
    scale_factor = 1.0 / math.sqrt(q.shape[-1])
    qk_dot = q @ k.transpose(-2, -1) * scale_factor
    #s = F.softmax(qk_dot, dim=-1)
    # Use hand written softmax
    qk_dot_max = torch.max(qk_dot, dim=-1, keepdim=True).values
    qk_dot = qk_dot - qk_dot_max
    qk_dot = torch.exp(qk_dot)
    qk_dot_sum = torch.sum(qk_dot, dim=-1, keepdim=True)
    s = qk_dot / qk_dot_sum

    out = s @ v
    return out


@pytest.mark.parametrize("N, H, S, D", [
    (16, 8, 64, 64)
])
def test_sdpa_fwd(N, H, S, D):
    q = torch.randn(N, H, S, D, device='cuda')
    k = torch.randn(N, H, S, D, device='cuda')
    v = torch.randn(N, H, S, D, device='cuda')
    out = scaled_dot_product_attention_fwd(q, k, v)
    ref_out = F.scaled_dot_product_attention(q, k, v) 
    torch.testing.assert_close(ref_out, out)


