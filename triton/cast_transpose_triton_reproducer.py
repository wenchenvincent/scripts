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
    A = A + rm[:, None] * stride_am + rn[None, :] * stride_an
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    a = tl.load(A, mask=mask)
    a = a.to(tl.float32)

    scaled_a = a * scale
    scaled_a = tl.clamp(scaled_a, -240.0, 240.0)
    fp8_a = scaled_a.to(tl.float8e4b8)
    C = C + rm[:, None] * stride_am + rn[None, :] * stride_an
    tl.store(C, fp8_a, mask=mask)
    
    # rematerialize to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    T = T + rm[:, None] * stride_bm + rn[None, :] * stride_bn
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    tl.store(T, fp8_a, mask=mask)
    amax = tl.max(tl.abs(a))
    tl.atomic_max(amax_ptr, amax, sem='relaxed')

def transpose_triton(input, input_scale, cast_out=None, trans_out=None, amax_out=None):
    M, N = input.shape
    if cast_out is None:
        cast_out = input.new_zeros(M, N, dtype=torch.float8_e4m3fnuz)
    if trans_out is None:
        trans_out = input.new_zeros(N, M, dtype=torch.float8_e4m3fnuz)
    if amax_out is None:
        amax_out = torch.zeros(1,dtype=torch.float32, device='cuda')

    
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
    amax = torch.max(torch.abs(input_tensor.to(torch.float32)))
    return casted_tensor, transposed_out, amax
    
# Correctness Test Function
def correctness_test():
    shapes = [(40960, 128256)]
    for M, N in shapes:
        print(f"Running correctness test for shape: {M}x{N}...")
        input_tensor = torch.randn(M, N, dtype=torch.float16, device='cuda')
        scale_tensor = torch.randn(1, dtype=torch.float32, device='cuda')
        # Transpose using PyTorch for correctness check
        #torch_output = input_tensor.transpose(0, 1).contiguous()
        torch_c, torch_t, torch_amax = torch_cast_transpose(input_tensor, scale_tensor)

        # Transpose using Triton
        triton_c = torch.empty(M, N, dtype=torch.float8_e4m3fnuz, device='cuda')
        triton_t = torch.empty(N, M, dtype=torch.float8_e4m3fnuz, device='cuda')
        triton_amax = torch.zeros(1, dtype=torch.float32, device='cuda')
        triton_c, triton_t, triton_amax = transpose_triton(input_tensor, scale_tensor, triton_c, triton_t, triton_amax)

        # Compare results
        if not torch.allclose(torch_c.to(torch.float32), triton_c.to(torch.float32), atol=1e-6):
            print(f"Test failed for shape: {M}x{N}")
            return False
        else:
            print(f"Test passed for shape: {M}x{N}")
        if not torch.allclose(torch_t.to(torch.float32), triton_t.to(torch.float32), atol=1e-6):
            print(f"Test failed for shape: {M}x{N}")
            return False
        else:
            print(f"Test passed for shape: {M}x{N}")
        print('torch_amax=', torch_amax, 'triton_amax=', triton_amax)
        if not torch.allclose(torch_amax, triton_amax, atol=1e-6):
            print(f"Test failed for shape: {M}x{N}")
            return False
        else:
            print(f"Test passed for shape: {M}x{N}")
    return True

# Main Function to run correctness and benchmarking tests
def main():
    #Run correctness test
    if not correctness_test():
        print("Correctness tests failed!")
        return

# Run the main function
if __name__ == "__main__":
    main()
