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
def _copy_triton(A, C, stride_am, stride_an,  M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, GROUP_M: tl.constexpr):
    pid = tl.program_id(0)

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

    C = C + rm[:, None] * stride_am + rn[None, :] * stride_an
    tl.store(C, a, mask=mask)
    
def copy_triton(input, cast_out=None):
    M, N = input.shape
    if cast_out is None:
        #cast_out = input.new_zeros(M, N, dtype=torch.float8_e4m3fnuz)
        cast_out = input.new_zeros(M, N, dtype=input.dtype)

    assert input.stride(0) == 1 or input.stride(1) == 1
    
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    _copy_triton[grid](input, cast_out, input.stride(0), input.stride(1), M, N)
    return cast_out

def torch_copy(input_tensor):
    casted_tensor = torch.clone(input_tensor) 
    return casted_tensor
    
# Correctness Test Function
def correctness_test():
    shapes = [(40960, 128256)]
    #shapes = [(4096, 1280)]
    for M, N in shapes:
        print(f"Running correctness test for shape: {M}x{N}...")
        input_tensor = torch.randn(M, N, dtype=torch.float16, device='cuda')
        torch_c = torch_copy(input_tensor)

        # Transpose using Triton
        triton_c = torch.empty(M, N, dtype=torch.float16, device='cuda')
        triton_c = copy_triton(input_tensor, triton_c)

        # Compare results
        if not torch.allclose(torch_c.to(torch.float32), triton_c.to(torch.float32), atol=1e-6):
            print(f"Test failed for shape: {M}x{N}")
            return False
        else:
            print(f"Test passed for shape: {M}x{N}")
    return True

def main():
    if not correctness_test():
        print("Correctness tests failed!")
        return

if __name__ == "__main__":
    main()
