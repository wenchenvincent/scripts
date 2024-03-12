import torch
import triton
import triton.language as tl

@triton.jit
def fp32_to_fp16_kernel(src, dst, BLOCK_SIZE : tl.constexpr):
    idxs = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(src + idxs)
    dummy = tl.inline_asm_elementwise("""s_setreg_imm32_b32 0x1801, 0xc""", "=r", [], dtype=tl.int32, is_pure=False, pack=1)
    y = x.to(tl.float16)
    tl.store(dst + idxs, y)

def launch_cast_kernel(src, BLOCK_SIZE=1):
    dst = torch.empty(src.shape, dtype=torch.float32, device='cuda')
    fp32_to_fp16_kernel[(src.shape[0] // BLOCK_SIZE,)](src, dst, BLOCK_SIZE)
    return dst

torch.set_printoptions(precision=20)
src = torch.tensor([ 1065359360], dtype=torch.int32, device='cuda')
src = src.view(torch.float32)
print('src=', src)
dst = launch_cast_kernel(src)
print('dst=', dst)
dst2 = src.to(torch.float16)
print('dst2=', dst2)


