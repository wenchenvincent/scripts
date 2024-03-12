import torch
import triton
import triton.language as tl

@triton.jit
def mul_kernel(src, dst, BLOCK_SIZE : tl.constexpr):

    exponent_compensator : tl.constexpr = 2.0 ** (127 - 15)
    idxs = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(src + idxs)
    y = x * exponent_compensator
    tl.store(dst + idxs, y)

def launch_mul_kernel(src, BLOCK_SIZE=1):
    dst = torch.empty(src.shape, dtype=torch.float32, device='cuda')
    mul_kernel[(src.shape[0] // BLOCK_SIZE,)](src, dst, BLOCK_SIZE)
    return dst

torch.set_printoptions(precision=20)
src = torch.tensor([8323072], dtype=torch.int32, device='cuda')
src = src.view(torch.float32)
print('src=', src)
dst = launch_mul_kernel(src)
print('dst=', dst)
dst2 = (2.0 ** (127 - 15)) * src
print('dst2=', dst2)


