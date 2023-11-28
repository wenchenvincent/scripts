import torch
import numpy as np

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

    out = np.zeros((GEMM_M, GEMM_N), dtype=in_data.dtype)
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
                    if gemm_i==0 and gemm_j==0:
                        print(f'gemm_k={gemm_k}, n={n},npq_residual={npq_residual},p={p},q={q},crs_residual={crs_residual},f={f},g={g},c={c},r={r},s={s},k={k}')
                        print(f'gemm_k={gemm_k}, a={in_data[n, f, g, c]}, b={in_filter[k, r, s, c]}, acc={acc}')
                     
            #print(f'gemm_i={gemm_i}, gemm_j={gemm_j}')
            out[gemm_i, gemm_j] = acc

    out = out.reshape((N, P, Q, K))
    return out
                
def int_cdiv(M, N):
    return (M + N - 1) // N

def fwd_conv_implicit_gemm_cpu_blocked(in_data, in_filter, PadH=0, PadW=0, U=1, V=1, DilH=1, DilW=1):
    assert in_data.shape[3] == in_filter.shape[3], 'Input channel numbers do not match!'
    N, H, W, C = in_data.shape
    K, R, S, _ = in_filter.shape
    P = ( H + 2 * PadH - (R - 1) * DilH ) // U 
    Q = ( W + 2 * PadW - (S - 1) * DilW ) // V 

    GEMM_M = N * P * Q
    GEMM_N = K
    GEMM_K = C * R * S
    print(f'N={N},H={H},W={W},C={C},K={K},R={R},S={S},P={P},Q={Q}')
    print(f'GEMM_M={GEMM_M},GEMM_N={GEMM_N},GEMM_K={GEMM_K}')

    BLOCK_SIZE_GEMM_M = 16
    BLOCK_SIZE_GEMM_N = 16
    BLOCK_SIZE_GEMM_K = 16
    
    in_data_reshaped = in_data.reshape(N*H*W,C)
    in_filter_reshaped = in_filter.reshape(K,R*S*C)
    print(f'in_data_reshaped={in_data_reshaped}')
    print(f'in_filter_reshaped={in_filter_reshaped}')
    out = np.zeros((GEMM_M, GEMM_N), dtype=in_data.dtype)
    for gemm_i in range(int_cdiv(GEMM_M, BLOCK_SIZE_GEMM_M)):
        n = ( gemm_i * BLOCK_SIZE_GEMM_M + np.arange(BLOCK_SIZE_GEMM_M) ) // (P*Q) 
        npq_residual = ( gemm_i *  BLOCK_SIZE_GEMM_M + np.arange(BLOCK_SIZE_GEMM_M) ) % (P*Q)
        
        p = npq_residual // Q
        q = npq_residual % Q
        for gemm_j in range(int_cdiv(GEMM_N, BLOCK_SIZE_GEMM_N)):
            k = gemm_j * BLOCK_SIZE_GEMM_N + np.arange(BLOCK_SIZE_GEMM_N)

            print(f'gemm_i={gemm_i}, gemm_j={gemm_j}')
            gemm_m = gemm_i * BLOCK_SIZE_GEMM_M + np.arange(BLOCK_SIZE_GEMM_M)
            gemm_n = gemm_j * BLOCK_SIZE_GEMM_N + np.arange(BLOCK_SIZE_GEMM_N)
            print(f'gemm_m={gemm_m}, gemm_n={gemm_n}')
            acc = np.zeros((BLOCK_SIZE_GEMM_M, BLOCK_SIZE_GEMM_N), dtype=in_data.dtype)
            for gemm_k in range(int_cdiv(GEMM_K, BLOCK_SIZE_GEMM_K)):
                c = ( gemm_k * BLOCK_SIZE_GEMM_K + np.arange(BLOCK_SIZE_GEMM_K) ) // (R*S)  
                crs_residual = ( gemm_k * BLOCK_SIZE_GEMM_K + np.arange(BLOCK_SIZE_GEMM_K) ) % (R*S)

                r = crs_residual // S
                s = crs_residual % S

                f = p[:,None] * U +  r[None,:] * DilH - PadH
                g = q[:,None] * V +  s[None,:] * DilW - PadW
                #print(f'gemm_k={gemm_k}, n={n},npq_residual={npq_residual},p={p},q={q},crs_residual={crs_residual},f={f},g={g},c={c},r={r},s={s},k={k}')
                mask_a = (n[:,None] < N) & (f>=0) & (f<H) & (g>=0) & (g<W) & (c[None,:]<C)
                a = np.where(mask_a, in_data_reshaped[(n[:,None]*H*W+f*W+g)%(N*H*W), c[None,:]%C], 0.)
                mask_b = (k[:,None] < K) & (r[None,:]<R) & (s[None,:]<S) & (c[None,:]<C)
                b = np.where(mask_b, in_filter_reshaped[k[:,None]%K,(r*S*C+s*C+c)[None,:]%(C*R*S)], 0.)
                #a = a.reshape(BLOCK_SIZE_GEMM_M, -1)
                #b = b.reshape(BLOCK_SIZE_GEMM_N, -1)
                acc += a@(b.transpose())
                #if (gemm_i==0 and gemm_j==0):
                    #print(f'(c*R*S+r*S+s)%(C*R*S)={(c*R*S+r*S+s)%(C*R*S)}')
                    #print(f'(c*R*S+r*S+s)={(c*R*S+r*S+s)}')
                    #print(f'(n[:,None]*H*W+f*W+g)%(N*H*W)={(n[:,None]*H*W+f*W+g)%(N*H*W)}')
                    #print(f'(n[:,None]*H*W+f*W+g)={(n[:,None]*H*W+f*W+g)}')
                    #print(f'mask_a={mask_a}')
                    #print(f'a={a}')
                    #print(f'b={b}')
                    #print(f'a@b={a@b.transpose()}')
                    #print(f'gemm_k={gemm_k}')
                    #print(f'acc={acc}')
            #mask_out = (gemm_m[:, None] < GEMM_M) & (gemm_n[None,:] < GEMM_N)
            #out[gemm_m[:,None], gemm_n[None,:]] = np.where(mask_out, acc, 0.)
            ## Deal with the case where GEMM_M or GEMM_N are not multiples of BLOCK_SIZE_GEMM_M or BLOCK_SIZE_GEMM_N
            if gemm_m[-1] >= GEMM_M:
                gemm_m = gemm_i * BLOCK_SIZE_GEMM_M + np.arange(GEMM_M-gemm_i * BLOCK_SIZE_GEMM_M)
            if gemm_n[-1] >= GEMM_N:
                gemm_n = gemm_j * BLOCK_SIZE_GEMM_N + np.arange(GEMM_N-gemm_j * BLOCK_SIZE_GEMM_N)
            out[gemm_m[:,None], gemm_n[None,:]] = acc[gemm_m[:,None]-gemm_i*BLOCK_SIZE_GEMM_M, gemm_n[None,:]-gemm_j*BLOCK_SIZE_GEMM_N] 
            #if (gemm_i==0 and gemm_j==0):
                #print(f'acc={acc}')

    out = out.reshape((N, P, Q, K))
    return out


def fwd_conv_implicit_gemm(in_data, in_filter, PadH=0, PadW=0, U=1, V=1, DilH=1, DilW=1):
    assert in_data.shape[3] == in_filter.shape[3], 'Input channel numbers do not match!'
    N, H, W, C = in_data.shape
    K, R, S, _ = in_filter.shape
    P = ( H + 2 * PadH - (R - 1) * DilH ) // U 
    Q = ( W + 2 * PadW - (S - 1) * DilW ) // V 

    GEMM_M = N * P * Q
    GEMM_N = K
    GEMM_K = C * R * S
    print(f'N={N},H={H},W={W},C={C},K={K},R={R},S={S},P={P},Q={Q}')
    print(f'GEMM_M={GEMM_M},GEMM_N={GEMM_N},GEMM_K={GEMM_K}')

    out = torch.empty((GEMM_M, GEMM_N), dtype=in_data.dtype, device='cuda')

    grid = lambda META: (
        triton.cdiv(GEMM_M, META['BLOCK_SIZE_GEMM_M']) * triton.cdiv(GEMM_N, META['BLOCK_SIZE_GEMM_N']),
    )
    fwd_conv_implicit_gemm_kernel[grid](
        in_data, in_filter, out,
        N, H, W, C, K, R, S, P, Q,
        GEMM_M, GEMM_N, GEMM_K,
        in_data.stride(0), in_data.stride(1), in_data.stride(2), in_data.stride(3),
        in_filter.stride(0), in_filter.stride(1), in_filter.stride(2),
        U, V, PadH, PadW, DilH, DilW
    )
    return out.reshape((N,P,Q,K))

@triton.autotune(
    configs = [
        triton.Config({'BLOCK_SIZE_GEMM_M': 16, 'BLOCK_SIZE_GEMM_N': 16, 'BLOCK_SIZE_GEMM_K': 16})
    ],
    key = ['GEMM_M', 'GEMM_N', 'GEMM_K']
)
@triton.jit
def fwd_conv_implicit_gemm_kernel(
    # Pointers
    data_ptr, filter_ptr, out_ptr,
    N, H, W, C, K, R, S, P, Q,
    GEMM_M, GEMM_N, GEMM_K,
    stride_n, stride_h, stride_w, stride_c,
    stride_k, stride_r, stride_s,
    U, V, PadH, PadW, DilH, DilW,
    BLOCK_SIZE_GEMM_M: tl.constexpr, BLOCK_SIZE_GEMM_N: tl.constexpr, BLOCK_SIZE_GEMM_K: tl.constexpr, 
):
    pid = tl.program_id(axis=0)
    num_pid_gemm_m = tl.cdiv(GEMM_M, BLOCK_SIZE_GEMM_M)
    num_pid_gemm_n = tl.cdiv(GEMM_N, BLOCK_SIZE_GEMM_N)
    pid_gemm_m = pid // num_pid_gemm_n
    pid_gemm_n = pid % num_pid_gemm_n

    offs_gemm_m = pid_gemm_m * BLOCK_SIZE_GEMM_M + tl.arange(0, BLOCK_SIZE_GEMM_M)
    offs_gemm_n = pid_gemm_n * BLOCK_SIZE_GEMM_N + tl.arange(0, BLOCK_SIZE_GEMM_N) #Need %?

    offs_n =  offs_gemm_m // (P*Q)
    offs_npq_residual = offs_gemm_m % (P*Q)

    offs_p = offs_npq_residual // Q
    offs_q = offs_npq_residual % Q

    offs_k = offs_gemm_n

    accumulator = tl.zeros((BLOCK_SIZE_GEMM_M, BLOCK_SIZE_GEMM_N), dtype=tl.float32)
    for gemm_k in range(0, tl.cdiv(GEMM_K, BLOCK_SIZE_GEMM_K)):
        offs_gemm_k = gemm_k * BLOCK_SIZE_GEMM_K + tl.arange(0, BLOCK_SIZE_GEMM_K)
        offs_c = offs_gemm_k // (R*S)
        offs_crs_residual = offs_gemm_k % (R*S)

        offs_r = offs_crs_residual // S
        offs_s = offs_crs_residual % S

        offs_h = offs_p[:,None] * U + offs_r[None,:] * DilH - PadH
        offs_w = offs_q[:,None] * U + offs_s[None,:] * DilW - PadW

        #Triton can only handle two dimensional pointer arithemtics
        #a_ptrs = data_ptr + offs_n[:, None, None, None] * stride_n + offs_h[None, :, :, None] * stride_h \
                             #+ offs_w[None, :, :, None] * stride_w + offs_k[None, None, None, :] * stride_k 
        # filter is [K, R, S, C], filter.transpose is [C, R, S, K]
        #b_ptrs = filter_ptr + offs_c[:, None, None, None] * stride_c + offs_r[None, :, None, None] * stride_r \
                                       #+ offs_s[None, None, :, None] * stride_s + offs_k[None, None, None, :] * stride_k
        #mask_a = (offs_n[:, None, None, None] < N) & (offs_h[None, :, :, None] >= 0) & (offs_h[None, :, :, None] < H) \
        #        & (offs_w[None, :, :, None] >=0) & (offs_w[None, :, :, None] < W) & (offs_c[None, None, None, :] <C)
        #mask_b = (offs_c[:, None, None, None] < C) & (offs_r[None, :, None, None] < R) & (offs_s[None, None, :, None] < S) & (offs_k[None, None, None, :] < K)
        a_ptrs = data_ptr + offs_n[:, None] * stride_n + offs_h * stride_h \
                          + offs_w * stride_w + offs_c[None, :] * stride_c 
        b_ptrs = filter_ptr + offs_c[:, None] * stride_c + offs_r[:, None] * stride_r \
                            + offs_s[:, None] * stride_s + offs_k[None, :] * stride_k
        mask_a = (offs_n[:, None] < N) & (offs_h >= 0) & (offs_h < H) \
                & (offs_w >=0) & (offs_w < W) & (offs_c[None, :] <C)
        mask_b = (offs_c[:, None] < C) & (offs_r[:, None] < R) & (offs_s[:, None] < S) & (offs_k[None, :] < K)
        a = tl.load(a_ptrs, mask_a, other=0.0)
        b = tl.load(b_ptrs, mask_b, other=0.0)
        accumulator += tl.dot(a, b)
    c = accumulator

    out_ptrs = out_ptr + offs_gemm_m[:, None] * GEMM_N + offs_gemm_n[None, :]
    mask_out = (offs_gemm_m[:, None] < GEMM_M) & (offs_gemm_n[None, :] < GEMM_N)
    tl.store(out_ptrs, c, mask_out)
         


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
    in_data = torch.randn(N, H, W, C, device='cuda')
    weight = torch.randn(K, R, S, C, device='cuda')
    in_data_torch = in_data.permute(0, 3, 1, 2)
    weight_torch = weight.permute(0, 3, 1, 2)
    torch_output = torch.nn.functional.conv2d(in_data_torch, weight_torch, padding=1, dilation=1, stride=(1,1)).permute(0, 2, 3, 1)
    triton_output = fwd_conv_implicit_gemm(in_data, weight, PadH=1, PadW=1, U=1, V=1, DilH=1, DilW=1)
    #cpu_output = fwd_conv_naive_cpu(in_data, weight, PadH=1, PadW=1, U=2, V=1, DilH=2, DilW=2)
    #cpu_output0 = fwd_conv_implicit_gemm_cpu(in_data.numpy(), weight.numpy(), PadH=1, PadW=1, U=1, V=1, DilH=1, DilW=1)
    #cpu_output = fwd_conv_implicit_gemm_cpu_blocked(in_data.numpy(), weight.numpy(), PadH=1, PadW=1, U=1, V=1, DilH=1, DilW=1)
    print(f'torch_output.shape={torch_output.shape}')
    print(f'triton_output.shape={triton_output.shape}')
    print(f'torch_output={torch_output}')
    print(f'triton_output={triton_output}')
    if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=1e-2):
        print("✅ CPU and Torch match")
    else:
        print("❌ CPU and Torch differ")
    
    

