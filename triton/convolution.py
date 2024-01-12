import torch
import numpy as np

import triton
import triton.language as tl
import sys
import argparse
import pytest

def fwd_conv_implicit_gemm(in_data, in_filter, PadH=0, PadW=0, U=1, V=1, DilH=1, DilW=1):
    assert in_data.shape[3] == in_filter.shape[3], 'Input channel numbers do not match!'
    N, H, W, C = in_data.shape
    K, R, S, _ = in_filter.shape
    P = ( H + 2 * PadH - (R - 1) * DilH ) // U 
    Q = ( W + 2 * PadW - (S - 1) * DilW ) // V 

    GEMM_M = N * P * Q
    GEMM_N = K
    GEMM_K = C * R * S

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
        U, V, PadH, PadW, DilH, DilW,
    )
    return out.reshape((N,P,Q,K))

'''
@triton.autotune(
    configs = [
        triton.Config({'BLOCK_SIZE_GEMM_M': 32, 'BLOCK_SIZE_GEMM_N': 16, 'BLOCK_SIZE_GEMM_K': 16}, num_stages=0),
        triton.Config({'BLOCK_SIZE_GEMM_M': 32, 'BLOCK_SIZE_GEMM_N': 32, 'BLOCK_SIZE_GEMM_K': 32}, num_stages=0),
        triton.Config({'BLOCK_SIZE_GEMM_M': 128, 'BLOCK_SIZE_GEMM_N': 32, 'BLOCK_SIZE_GEMM_K': 32}, num_stages=0),
    ],
    key = ['GEMM_M', 'GEMM_N', 'GEMM_K']
)
'''
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_GEMM_M': 128, 'BLOCK_SIZE_GEMM_N': 256, 'BLOCK_SIZE_GEMM_K': 64, 'GROUP_SIZE_GEMM_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_GEMM_M': 64, 'BLOCK_SIZE_GEMM_N': 256, 'BLOCK_SIZE_GEMM_K': 32, 'GROUP_SIZE_GEMM_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_GEMM_M': 128, 'BLOCK_SIZE_GEMM_N': 128, 'BLOCK_SIZE_GEMM_K': 32, 'GROUP_SIZE_GEMM_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_GEMM_M': 128, 'BLOCK_SIZE_GEMM_N': 64, 'BLOCK_SIZE_GEMM_K': 32, 'GROUP_SIZE_GEMM_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_GEMM_M': 64, 'BLOCK_SIZE_GEMM_N': 128, 'BLOCK_SIZE_GEMM_K': 32, 'GROUP_SIZE_GEMM_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_GEMM_M': 128, 'BLOCK_SIZE_GEMM_N': 32, 'BLOCK_SIZE_GEMM_K': 32, 'GROUP_SIZE_GEMM_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_GEMM_M': 64, 'BLOCK_SIZE_GEMM_N': 32, 'BLOCK_SIZE_GEMM_K': 32, 'GROUP_SIZE_GEMM_M': 8}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_SIZE_GEMM_M': 32, 'BLOCK_SIZE_GEMM_N': 64, 'BLOCK_SIZE_GEMM_K': 32, 'GROUP_SIZE_GEMM_M': 8}, num_stages=5,
                      num_warps=2),
    ] if torch.version.hip is None else [
        triton.Config({'BLOCK_SIZE_GEMM_M': 128, 'BLOCK_SIZE_GEMM_N': 256, 'BLOCK_SIZE_GEMM_K': 16, 'GROUP_SIZE_GEMM_M': 1, 'waves_per_eu': 2},
                      num_warps=4, num_stages=0),
        triton.Config({'BLOCK_SIZE_GEMM_M': 256, 'BLOCK_SIZE_GEMM_N': 256, 'BLOCK_SIZE_GEMM_K': 16, 'GROUP_SIZE_GEMM_M': 4, 'waves_per_eu': 2},
                      num_warps=8, num_stages=0),
        triton.Config({'BLOCK_SIZE_GEMM_M': 128, 'BLOCK_SIZE_GEMM_N': 128, 'BLOCK_SIZE_GEMM_K': 32, 'GROUP_SIZE_GEMM_M': 1, 'waves_per_eu': 2},
                      num_warps=8, num_stages=0),
        triton.Config({'BLOCK_SIZE_GEMM_M': 64, 'BLOCK_SIZE_GEMM_N': 128, 'BLOCK_SIZE_GEMM_K': 32, 'GROUP_SIZE_GEMM_M': 8, 'waves_per_eu': 3},
                      num_warps=4, num_stages=0),
        triton.Config({'BLOCK_SIZE_GEMM_M': 64, 'BLOCK_SIZE_GEMM_N': 64, 'BLOCK_SIZE_GEMM_K': 32, 'GROUP_SIZE_GEMM_M': 1, 'waves_per_eu': 8},
                      num_warps=4, num_stages=0),
    ],
    key=['GEMM_M', 'GEMM_N', 'GEMM_K'],
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
    GROUP_SIZE_GEMM_M: tl.constexpr,  
):
    pid = tl.program_id(axis=0)
    num_pid_gemm_m = tl.cdiv(GEMM_M, BLOCK_SIZE_GEMM_M)
    num_pid_gemm_n = tl.cdiv(GEMM_N, BLOCK_SIZE_GEMM_N)
    num_pid_in_group = GROUP_SIZE_GEMM_M * num_pid_gemm_n
    group_id = pid // num_pid_in_group
    first_pid_gemm_m = group_id * GROUP_SIZE_GEMM_M
    group_size_gemm_m = min(num_pid_gemm_m - first_pid_gemm_m, GROUP_SIZE_GEMM_M)
    pid_gemm_m = first_pid_gemm_m + (pid % group_size_gemm_m)
    pid_gemm_n = (pid % num_pid_in_group) // group_size_gemm_m

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
        # filter is [K, R, S, C], filter.transpose is [C, R, S, K]
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
    c = accumulator.to(tl.float16)

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
    in_data = torch.randn((N, H, W, C), device='cuda')
    weight = torch.randn((K, R, S, C), device='cuda')
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
    
    
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N', 'H', 'W', 'C', 'K', 'R', 'S', 'U', 'V', 'PadH', 'PadW', 'DilH', 'DilW'],  # Argument names to use as an x-axis for the plot
        x_vals=[ 
        (64, 224, 224, 3) + (64, 7, 7) + (2, 2, 3, 3, 1, 1),
        (64, 56, 56, 64)  + (64, 1, 1) + (1, 1, 0, 0, 1, 1),
        (64, 56, 56, 64)  + (64, 3, 3) + (1, 1, 1, 1, 1, 1),
        (64, 56, 56, 64)  + (256,1, 1) + (1, 1, 0, 0, 1, 1),
        (64, 56, 56, 64)  + (64, 3, 3) + (1, 1, 1, 1, 1, 1),
        (64, 28, 28, 512) + (128,3, 3) + (2, 2, 1, 1, 1, 1),
        (64, 28, 28, 128) + (512,1, 1) + (1, 1, 0, 0, 1, 1),
        (64, 28, 28, 256) + (256,3, 3) + (2, 2, 1, 1, 1, 1),
        (64, 14, 14, 1024)+ (256,1, 1) + (1, 1, 0, 0, 1, 1)

        ],  # Different possible values for `x_name`
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot
        # Possible values for `line_arg`
        line_vals=['pytorch', 'triton'],
        # Label name for the lines
        line_names=["Pytorch", "Triton"],
        # Line styles
        styles=[('green', '-'), ('blue', '-')],
        ylabel="TFLOPS",  # Label name for the y-axis
        plot_name="conv-performance",  # Name for the plot, used also as a file name for saving the plot.
        args={},
    )
)
def benchmark(N, H, W, C, K, R, S, U, V, PadH, PadW, DilH, DilW, provider):
    in_data = torch.randn((N, H, W, C), device='cuda', dtype=torch.float16)
    weight = torch.randn((K, R, S, C), device='cuda', dtype=torch.float16)
    in_data_torch = in_data.permute(0, 3, 1, 2)
    weight_torch = weight.permute(0, 3, 1, 2)
    
    P = ( H + 2 * PadH - (R - 1) * DilH ) // U 
    Q = ( W + 2 * PadW - (S - 1) * DilW ) // V 

    quantiles = [0.5, 0.2, 0.8]
    if provider == 'pytorch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.nn.functional.conv2d(in_data_torch, weight_torch), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: fwd_conv_implicit_gemm(in_data, weight), quantiles=quantiles)
        global verbose
        if verbose:
            print(f'SIZE: {N},{H},{W},{C}, {K},{R},{S}   Best tuning config: ({fwd_conv_implicit_gemm_kernel.get_best_config()})')
    perf = lambda ms: 2 * (N*P*Q) * K * (C*R*S) * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Conv tutorial example",
        allow_abbrev=False,
    )

    parser.add_argument("-v", action='store_true', default=False, help="Print out the best tuning config")
    args = parser.parse_args()

    return args


def main():
    # assign to a global verbose var to indicate whether print
    # best tuning config
    global verbose
    args = parse_args()
    verbose=args.v
    benchmark.run(show_plots=True, print_data=True, save_path='.')

if __name__ == '__main__':
    sys.exit(main())
