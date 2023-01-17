#include <hip/hip_runtime.h>
#include <cstdio>

__global__
void convert_fp8X4_to_fp16X4_kernel(uint32_t* in, uint32_t* out1, uint32_t* out2) {
	           uint32_t tmp;
		   uint32_t input = in[threadIdx.x];
	           asm volatile("s_mov_b32 s0, 0x0;                     \n");
	           asm volatile("s_mov_b32 s1, 0x05000400;              \n");
		   asm volatile("v_perm_b32 v0, %1, 0, s1;              \n" : "=v"(tmp): "v"(input) );
		   //asm volatile("v_perm_b32 %0, %1, 0, s1;              \n" : "=v"(out1[threadIdx.x]): "v"(input) );
	           asm volatile("s_mov_b32 s1, 0x07000600;              \n");
		   asm volatile("v_perm_b32 v1, %1, 0, s1;              \n" : "=v"(tmp): "v"(input) );
		   //asm volatile("v_perm_b32 %0, %1, 0, s1;              \n" : "=v"(out2[threadIdx.x]): "v"(input) );

		   //asm volatile("s_mov_b32 s1, 0x7fff7fff;              \n");
		   //asm volatile("v_and_b32_e32 v2, s1, v0;              \n" );
		   //asm volatile("v_and_b32_e32 v3, s1, v1;              \n" );
		   uint32_t mask = 0x7fff7fff;
		   asm volatile("v_and_b32_e32 %0, %1, v0;              \n" : "=v"(out1[threadIdx.x]) : "v"(mask) );
		   asm volatile("v_and_b32_e32 %0, %1, v1;              \n" : "=v"(out2[threadIdx.x]) : "v"(mask) );
		   /*
		   asm volatile("v_lshrrev_b32_e32 v2, 1, v2;           \n");
		   asm volatile("v_lshrrev_b32_e32 v3, 1, v3;           \n");
		   asm volatile("s_mov_b32 s1, 0x80008000;              \n");
		   asm volatile("v_and_or_b32 %0, v0, s1, v2;           \n" : "=v"(out1[threadIdx.x]) );
		   asm volatile("v_and_or_b32 %0, v1, s1, v3;           \n" : "=v"(out2[threadIdx.x]) );
		   */
}

int main() {
	uint32_t in_h = 0x3f3f3f3f;
	uint32_t out1_h, out2_h;
	uint32_t *in, *out1, *out2;
	hipMalloc(&in, sizeof(uint32_t));
	hipMalloc(&out1, sizeof(uint32_t));
	hipMalloc(&out2, sizeof(uint32_t));
	hipMemcpy(in, &in_h, sizeof(uint32_t), hipMemcpyHostToDevice);
	
	dim3 block_size(1);
	dim3 num_blocks(1);
	hipLaunchKernelGGL(convert_fp8X4_to_fp16X4_kernel, num_blocks, block_size, 0, 0, in, out1, out2);
	hipMemcpy(&out1_h, out1, sizeof(uint32_t), hipMemcpyDeviceToHost);
	hipMemcpy(&out2_h, out2, sizeof(uint32_t), hipMemcpyDeviceToHost);

	printf("out1 = %x, out2 = %x\n", out1_h, out2_h);
	return 0;
}
/*
    //Fp8X4 to Fp16X4
    GCNBuilder builder;
    auto *gcnAsm = "{                                      \n"
	           "s_mov_b32 s0, 0x05000400;              \n"
		   "v_perm_b32 v0, %2, 0, s0;              \n" 
	           "s_mov_b32 s0, 0x07000600;              \n"
		   "v_perm_b32 v1, %2, 0, s0;              \n" 
		   "s_mov_b32 s0, 0x7fff7fff;              \n"
		   "v_and_b32_e32 v2, s0, v0;              \n" 
		   "v_and_b32_e32 v3, s0, v1;              \n" 
		   "v_lshrrev_b32_e32 v2, 1, v2;           \n"
		   "v_lshrrev_b32_e32 v3, 1, v3;           \n"
		   "s_mov_b32 s0, 0x80008000;              \n"
		   "v_and_or_b32 %0, v0, s0, v2;           \n"
		   "v_and_or_b32 %1, v1, s0, v3;           \n"
                   "}";

    auto &call = *builder.create(gcnAsm);

    auto *o0 = builder.newOperand("=v");
    auto *o1 = builder.newOperand("=v");
    auto *i = builder.newOperand(fp8x4Vec, "v");
    call({o0, o1, i}, {});

    //Fp16X4 to Fp8X4
    GCNBuilder builder;
    auto *gcnAsm = "{                                      \n"
	           "v_lshlrev_b32_e32 v0, 1, %1;           \n"
	           "v_lshlrev_b32_e32 v1, 1, %2;           \n"
	           "s_mov_b32 s0, 0x7fff7fff;              \n"
		   "s_mov_b32 s1, 0x00800080;              \n"
		   "v_and_b32_e32 v0, s0, v0;              \n"
		   "v_and_b32_e32 v1, s0, v1;              \n"
	           "v_add_u32_e32 v0, s1, v0;              \n"
	           "v_add_u32_e32 v1, s1, v1;              \n"
		   "s_mov_b32 s1, 0x80008000;              \n"
		   "v_and_or_b32 v0, %1, s1, v0;           \n"
		   "v_and_or_b32 v1, %2, s1, v1;           \n"
		   "s_mov_b32 s2, 0x07050301;              \n"
		   "v_perm_b32 %0, v1, v0, s2;             \n"
                   "}";
    auto &call = *builder.create(gcnAsm);

    auto *o = builder.newOperand("=v");
    auto *i0 = builder.newOperand(fp16x2Vec0, "v");
    auto *i1 = builder.newOperand(fp16x2Vec1, "v");
    call({o, i0, i1}, {});

    */
