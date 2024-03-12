#include <cstdio>
#include <cmath>
#include <hip/hip_runtime.h>

__global__ void mul_kernel(float* src, float* dst, int N) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < N) {
		float exponent_compensator = pow(2.0, (127-15));
		dst[idx] = src[idx] * exponent_compensator;
	}
}

int main() {
	int32_t src = 8323072;
	float src_f = reinterpret_cast<float&>(src);
	float exponent_compensator = powf(2.0,(127-15));
	float dst = src_f * exponent_compensator;
	printf("src=%.10e, dst=%.10e\n", src_f, dst);


	float* src_d, *dst_d;
	float dst_h;
	hipMalloc(&src_d, sizeof(float));
	hipMalloc(&dst_d, sizeof(float));
	hipMemcpy(src_d, &src_f, sizeof(float), hipMemcpyHostToDevice);
	
	mul_kernel<<<1,1>>>(src_d, dst_d, 1);

	hipMemcpy(&dst_h, dst_d, sizeof(float), hipMemcpyDeviceToHost);
	printf("dst_h=%.10e\n", dst_h);

	hipFree(src_d);
	hipFree(dst_d);

	return 0;
}
