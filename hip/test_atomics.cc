// hipcc --offload-arch=gfx90a -O3 test_atomics.cc -save-temps -Wno-deprecated-declarations  
#include "hip/hip_runtime.h"

__global__ void atomicAddKernel(float *__restrict__ a) {
	  // Return value is clearly unused.
	     atomicAdd(a, 1.0);
	     }

template <typename T, typename F>
__device__ T GpuAtomicCasHelper(T* ptr, F accumulate) {
	  T old = *ptr;
          T assumed;
          do {
		  assumed = old;
	          old = atomicCAS(ptr, assumed, accumulate(assumed));
	     } while (assumed != old);
	return old;
}

__device__ inline float GpuAtomicAdd(float* ptr, float value) {
	  return GpuAtomicCasHelper(ptr,
				    [value](float a) { return a + value; });
}

__global__ void atomicCASKernel(float *__restrict__ a) {
	     GpuAtomicAdd(a, 1.0);
}

__global__ void atomicAddNoRetKernel(float *__restrict__ a) {

// 'atomicAddNoRet' is deprecated: use atomicAdd instead
 atomicAddNoRet(a, 1);
 }

 int main() { return 0; }
