import torch

import triton
import triton.language as tl
import sys
import argparse
import pytest

@triton.jit
def fwd_conv_kernel():
    pass

def fwd_conv(in_data, in_filter):
    assert in_data.is_contiguous(), "Input Data must but contiguous"
    assert in_filter.is_contiguous(), "Input Filter must but contiguous"
    pass



def fwd_conv_naive_cpu(in_data, in_filter, PadH=0, PadW=0, U=1, V=1, DilH=1, DilW=1):
    assert in_data.shape[3] == in_filter.shape[3], 'Input channel numbers do not match!'
    N, H, W, C = in_data.shape
    K, R, S, _ = in_filter.shape
    P = ( H + 2 * PadH - (R - 1) * DilH ) // U 
    Q = ( W + 2 * PadW - (S - 1) * DilW ) // V 
    out = torch.zeros((N, P, Q, K), dtype=in_data.dtype)
    for n in range(N):
        for p in range(P):
            for q in range(Q):
                for k in range(K):
                    acc = 0.
                    for r in range(R):
                        for s in range(S):
                            for c in range(C):
                                #f = p * U + R - r - 1 - PadH
                                #g = q * V + S - s - 1 - PadW
                                #if f >=0 and f < H and g >=0 and g < W:
                                    #acc += in_data[n, f, g, c] * in_filter[k, R-1-r, S-1-s, c]
                                f = p * U +  r * DilH - PadH
                                g = q * V +  s * DilW - PadW
                                if f >=0 and f < H and g >=0 and g < W:
                                    acc += in_data[n, f, g, c] * in_filter[k, r, s, c]
                    out[n, p, q, k] = acc
    return out

def fwd_conv_implicit_gemm_cpu(in_data, in_filter, PadH=0, PadW=0, U=1, V=1, DilH=1, DilW=1):
    assert in_data.shape[3] == in_filter.shape[3], 'Input channel numbers do not match!'
    N, H, W, C = in_data.shape
    K, R, S, _ = in_filter.shape
    P = ( H + 2 * PadH - (R - 1) * DilH ) // U 
    Q = ( W + 2 * PadW - (S - 1) * DilW ) // V 

    GEMM_M = N * P * Q
    GEMM_N = K
    GEMM_K = C * R * S
    print(f'GEMM_M={GEMM_M},GEMM_N={GEMM_N},GEMM_K={GEMM_K}')

    out = torch.zeros((GEMM_M, GEMM_N), dtype=in_data.dtype)
    for gemm_i in range(GEMM_M):
        for gemm_j in range(GEMM_N):
            n = gemm_i // (P*Q)
            npq_residual = gemm_i % (P*Q)
            
            p = npq_residual // Q
            q = npq_residual % Q
            
            acc = 0.
            for gemm_k in range(GEMM_K):
                k = gemm_j
                c = gemm_k // (R*S)
                crs_residual = gemm_k % (R*S)

                r = crs_residual // S
                s = crs_residual % S

                f = p * U +  r * DilH - PadH
                g = q * V +  s * DilW - PadW
                # If it is the convolution in signal processing,
                #f = p * U + (R - r -1) * DilH - PadH
                #g = q * V + (S - s -1) * DilW - PadW

                if f >=0 and f < H and g >=0 and g < W:
                    acc += in_data[n, f, g, c] * in_filter[k, r, s, c]
            #print(f'gemm_i={gemm_i}, gemm_j={gemm_j}')
            out[gemm_i, gemm_j] = acc

    out = out.reshape((N, P, Q, K))
    return out
                
                    

@pytest.mark.parametrize("N, H, W, C, K, R, S",
[ (4, 16, 16, 4, 8, 3, 3) 
]
)
def test_correctness(N, H, W, C, K, R, S):
    '''
    N: batch size
    H: input height
    W: input width
    C: input channels
    K: output channels
    R: filter height
    S: filter width
    U: vertical stride
    V: horizontal stride
    '''
    torch.manual_seed(0)
    in_data = torch.randn(N, H, W, C)
    weight = torch.randn(K, R, S, C)
    in_data_torch = in_data.permute(0, 3, 1, 2).to(device='cuda')
    weight_torch = weight.permute(0, 3, 1, 2).to(device='cuda')
    torch_output = torch.nn.functional.conv2d(in_data_torch, weight_torch, padding=1, dilation=2, stride=(2,1)).permute(0, 2, 3, 1)
    #cpu_output = fwd_conv_naive_cpu(in_data, weight, PadH=1, PadW=1, U=2, V=1, DilH=2, DilW=2)
    cpu_output = fwd_conv_implicit_gemm_cpu(in_data, weight, PadH=1, PadW=1, U=2, V=1, DilH=2, DilW=2)
    print(f'torch_output.shape={torch_output.shape}')
    print(f'cpu_output.shape={cpu_output.shape}')
    print(f'torch_output={torch_output}')
    print(f'cpu_output={cpu_output}')
    if torch.allclose(cpu_output.to('cuda'), torch_output, atol=1e-2, rtol=1e-2):
        print("✅ CPU and Torch match")
    else:
        print("❌ CPU and Torch differ")
    
    

