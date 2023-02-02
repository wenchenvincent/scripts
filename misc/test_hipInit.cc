#include <cstdio>
#include <hip/hip_runtime.h>

int main() {
  hipError_t res = hipErrorNoDevice;
  res = hipInit(0 /* = flags */);
  if (res == hipSuccess) 
    printf("HIP init succeeded!\n");
  else
    printf("HIP init failed!\n");
  return 0;
}
