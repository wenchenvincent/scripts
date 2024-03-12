#include <cstdio>
#include <cmath>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

__global__ void cast_kernel(float* src, __half* dst, int N) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	// S_SETREG_IMM32_B32
	// SIMM16 = {size[4:0], offset[4:0], hwRegId[5:0]}; offset is 0..31, size is 1..32.
	// mode - hwRegId=1
	// FP_ROUND 3:0
	// [1:0] Single-precision round mode. (round to single precision)
	// [3:2] Double/Half-precision round mode. (round to half precision)
	// Round Modes: 0=nearest even; 1= +infinity; 2= -infinity, 3= toward zero.
	// SIMM32 = 0xF, size = 4, offset=0, hwRegId=1
	// SIMM16 = 00011, 00000, 000001 = 0x1801
	// S_SETREG_IMM32_B32
	__asm("s_setreg_imm32_b32 0x1801, 0xc"); // Set half rounding mode to rtz
	if (idx < N) {
		__half y = static_cast<__half>(src[idx]);
		dst[idx] = y;
	}
}

int main() {
	int32_t src = 1065359360;
	float src_f = reinterpret_cast<float&>(src);
	__half dst = (__half)src_f;
	printf("src=%.10e, dst=%.10e\n", src_f, (float)dst);


	float* src_d;
	__half *dst_d;
	__half dst_h;
	hipMalloc(&src_d, sizeof(float));
	hipMalloc(&dst_d, sizeof(__half));
	hipMemcpy(src_d, &src_f, sizeof(float), hipMemcpyHostToDevice);
	
	cast_kernel<<<1,1>>>(src_d, dst_d, 1);

	hipMemcpy(&dst_h, dst_d, sizeof(__half), hipMemcpyDeviceToHost);
	printf("dst_h=%.10e\n", (float)dst_h);

	hipFree(src_d);
	hipFree(dst_d);

	return 0;
}
