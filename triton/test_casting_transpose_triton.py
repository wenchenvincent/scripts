import torch
import triton
import triton.language as tl

@triton.autotune(
        configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'GROUP_M': 1}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'GROUP_M': 8}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'GROUP_M': 8}, num_warps=8),
        ],
        key=['M', 'N']
)
@triton.jit
def _transpose_triton(A, C, T, stride_am, stride_an, stride_bn, stride_bm, M, N, scale_ptr, amax_ptr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    scale = tl.load(scale_ptr)

    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size
    
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    A = A + (rm[:, None] * stride_am + rn[None, :] * stride_an)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    a = tl.load(A, mask=mask)

    scaled_a = a * scale
    fp8_a = scaled_a.to(tl.float8e4b8)
    C = C + (rm[:, None] * stride_am + rn[None, :] * stride_an)
    tl.store(C, fp8_a, mask=mask)
    
    amax = tl.max(tl.abs(scaled_a))
    tl.atomic_max(amax_ptr, amax,sem='relaxed')
    # rematerialize to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    T = T + (rm[:, None] * stride_bm + rn[None, :] * stride_bn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    tl.store(T, fp8_a, mask=mask)

def transpose_triton(input, input_scale, cast_out=None, trans_out=None, amax_out=None):
    M, N = input.shape
    if cast_out is None:
        cast_out = input.new_zeros(M, N, dtype=torch.float8_e4m3fnuz)
    if trans_out is None:
        trans_out = input.new_zeros(N, M, dtype=torch.float8_e4m3fnuz)
    if amax_out is None:
        amax_out = torch.empty(1,dtype=torch.float32, device='cuda')

    
    assert trans_out.size(0) == N and trans_out.size(1) == M
    assert input.stride(0) == 1 or input.stride(1) == 1
    assert trans_out.stride(0) == 1 or trans_out.stride(1) == 1
    
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    _transpose_triton[grid](input, cast_out, trans_out, input.stride(0), input.stride(1), trans_out.stride(0), trans_out.stride(1), M, N, input_scale, amax_out)
    return cast_out, trans_out, amax_out

def torch_cast_transpose(input_tensor, scale):
    scaled_tensor = scale * input_tensor
    casted_tensor = scaled_tensor.to(torch.float8_e4m3fnuz)
    transposed_out = casted_tensor.transpose(0,1).contiguous()
    amax = torch.max(torch.abs(scaled_tensor))
    return casted_tensor, transposed_out, amax
    
# Correctness Test Function
def correctness_test():
    input_tensor = torch.tensor([[5407]], dtype=torch.uint16, device='cuda')
    input_tensor = input_tensor.view(dtype=torch.float16)
    print('input_tensor=', input_tensor)
    scale_tensor = torch.tensor(1.0, dtype=torch.float32, device='cuda')
    # Transpose using PyTorch for correctness check
    #torch_output = input_tensor.transpose(0, 1).contiguous()
    torch_c, torch_t, torch_amax = torch_cast_transpose(input_tensor, scale_tensor)
    print('torch_c=', torch_c)

    # Transpose using Triton
    triton_c = torch.empty(1, 1, dtype=torch.float8_e4m3fnuz, device='cuda')
    triton_t = torch.empty(1, 1, dtype=torch.float8_e4m3fnuz, device='cuda')
    triton_amax = torch.empty(1, dtype=torch.float32, device='cuda')
    triton_c, triton_t, triton_amax = transpose_triton(input_tensor, scale_tensor, triton_c, triton_t, triton_amax)
    print('triton_c=', triton_c)

    # Compare results
    if not torch.allclose(torch_c.to(torch.float32), triton_c.to(torch.float32), atol=1e-6):
        return False
    if not torch.allclose(torch_t.to(torch.float32), triton_t.to(torch.float32), atol=1e-6):
        return False
    if not torch.allclose(torch_amax, triton_amax, atol=1e-6):
        return False
    return True


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M', 'N'],  # Argument names to use as an x-axis for the plot.
        #x_vals=[2**i for i in range(12, 28, 1)],  # Different possible values for `x_name`.
        x_vals = [(1024, 1024), (24576, 1024), (3072, 1024), (4096, 1024), (1024, 4096), (24576, 4096)],
        #x_vals = [(24576, 1024), (24576, 4096)],
        x_log=True,  # x axis is logarithmic.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['triton', 'torch'],  # Possible values for `line_arg`.
        line_names=['Triton', 'Torch'],  # Label name for the lines.
        styles=[('blue', '-'), ('green', '-')],  # Line styles.
        ylabel='GB/s',  # Label name for the y-axis.
        plot_name='cast-transpose-performance',  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    ))
def benchmark(M, N, provider):
    x = torch.rand((M,N), device='cuda', dtype=torch.float16)
    scale = torch.rand(1, device='cuda', dtype=torch.float32)
    c = torch.empty((M,N), device='cuda', dtype=torch.float8_e4m3fnuz)
    t = torch.empty((N,M), device='cuda', dtype=torch.float8_e4m3fnuz)
    amax = torch.empty(1, device='cuda', dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch_cast_transpose(x, scale), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: transpose_triton(x, scale,c,t), quantiles=quantiles)
    #gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    #return gbps(ms), gbps(max_ms), gbps(min_ms)
    return ms, min_ms, max_ms

# Main Function to run correctness and benchmarking tests
def main():
    #Run correctness test
    if not correctness_test():
        print("Correctness tests failed!")
        return

    # Shapes to benchmark
    #shapes = [(1024, 1024), (24576, 1024), (3072, 1024), (4096, 1024), (1024, 4096), (24576, 4096)]

    # Run benchmarking for each shape
    benchmark.run(print_data=True, show_plots=True)

# Run the main function
if __name__ == "__main__":
    main()
