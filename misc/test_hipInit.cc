#include <cstdio>
#include <hip/hip_runtime.h>

int main() {
  hipError_t res = hipErrorNoDevice;
  res = hipInit(0 /* = flags */);
  if (res == hipSuccess) 
    printf("HIP init succeeded!\n");
  else
    printf("HIP init failed!\n");

  size_t free, total;
  res = hipMemGetInfo(&free, &total);
  if (res == hipSuccess) {
    printf("Free memory is %lu. Total memory is %lu.\n", free, total);
  }
  else
    printf("hiMemGetInfo failed.\n");
  return 0;
}
