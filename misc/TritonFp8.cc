#include <hip/hip_runtime.h>

__device__
uint32_t cast_fp16X4_to_fp8X4(uint32_t fp16X2_1, uint32_t fp16X2_2) {
  uint32_t a0 = fp16X2_1 << 1;
  uint32_t a1 = fp16X2_2 << 1;
  a0 = a0 & 0x7fff7fff;
  a1 = a1 & 0x7fff7fff;
  a0 = a0 + 0x00800080;
  a1 = a1 + 0x00800080;
  uint32_t b0 = (fp16X2_1 & 0x80008000) | a0;
  uint32_t b1 = (fp16X2_2 & 0x80008000) | a1;
  uint32_t fp8X4 = (b1 & 0xff000000) |  ( (b1 & 0x0000ff00) << 8 ) | ( (b0 & 0xff000000) >> 16 ) | ( (b0 & 0x0000ff00) >> 8 );
  return fp8X4;
}

__global__ void cast_fp16X4_to_fp8X4_kernel(uint32_t* in1, uint32_t* in2, uint32_t* out) {
   out[threadIdx.x] = cast_fp16X4_to_fp8X4(in1[threadIdx.x], in2[threadIdx.x]);
}

struct fp16X4vec {   
   uint32_t fp16X2_0;
   uint32_t fp16X2_1;
   __device__
   fp16X4vec& operator=(const fp16X4vec& rhs) {
     fp16X2_0 = rhs.fp16X2_0;
     fp16X2_1 = rhs.fp16X2_1;
     return *this;
   }
};

__device__
fp16X4vec cast_fp8X4_to_fp16X4(uint32_t fp8X4) {
  uint32_t a0 = ( (fp8X4 & 0x0000ff00) << 16 ) | ( (fp8X4 & 0x000000ff) << 8 );
  uint32_t a1 = (fp8X4 & 0xff000000) | ( (fp8X4 & 0x00ff0000) >> 8 );
  uint32_t b0 = a0 & 0x7fff7fff;
  uint32_t b1 = a1 & 0x7fff7fff;
  b0 = b0 >> 1;
  b1 = b1 >> 1;
  fp16X4vec fp16X4;
  fp16X4.fp16X2_0 = b0 | ( a0 & 0x80008000);
  fp16X4.fp16X2_1 = b1 | ( a1 & 0x80008000);
  return fp16X4;
}


__global__ void cast_fp8X4_to_fp16X4_kernel(uint32_t* in, fp16X4vec* out) {
   out[threadIdx.x] = cast_fp8X4_to_fp16X4(in[threadIdx.x]);
}

struct bf16X4vec {   
   uint32_t bf16X2_0;
   uint32_t bf16X2_1;
   __device__
   bf16X4vec& operator=(const bf16X4vec& rhs) {
     bf16X2_0 = rhs.bf16X2_0;
     bf16X2_1 = rhs.bf16X2_1;
     return *this;
   }
};
__device__
bf16X4vec cast_fp8X4_to_bf16X4(uint32_t fp8X4) {
  uint32_t a0 = ( (fp8X4 & 0x0000ff00) << 16 ) | ( (fp8X4 & 0x000000ff) << 8 );
  uint32_t a1 = (fp8X4 & 0xff000000) | ( (fp8X4 & 0x00ff0000) >> 8 );
  uint32_t sign0 = a0 & 0x80008000;
  uint32_t sign1 = a1 & 0x80008000;
  uint32_t nosign0 = a0 & 0x7fff7fff;
  uint32_t nosign1 = a1 & 0x7fff7fff;
  nosign0 = nosign0 >> 4;
  nosign1 = nosign1 >> 4;
  nosign0 = nosign0 + 0x38003800;
  nosign1 = nosign1 + 0x38003800;
  bf16X4vec bf16X4;
  bf16X4.bf16X2_0 = sign0 | nosign0;
  bf16X4.bf16X2_1 = sign1 | nosign1;

  return bf16X4;
   
}

__device__
uint32_t cast_bf16X4_to_fp8X4(uint32_t bf16X2_1, uint32_t bf16X2_2) {
  uint32_t fp8_min = 0x38003800;
  uint32_t fp8_max = 0x3ff03ff0;
  uint32_t rn_ = 0x80008;
  uint32_t zero = 0;
  uint32_t sign0 = bf16X2_1 & 0x80008000;
  uint32_t sign1 = bf16X2_2 & 0x80008000;
  uint32_t sign = (sign1 & 0xff000000) | ( (sign1 & 0x0000ff00) << 8 ) | ( (sign0 & 0xff000000) >> 16 ) | ( (sign0 & 0x0000ff00) >> 8 );

  uint32_t nosign0 = bf16X2_1 & 0x7fff7fff;
  uint32_t nosign1 = bf16X2_2 & 0x7fff7fff;

  uint32_t nosign_0_0 = nosign0 & 0xffff0000;
  nosign_0_0 = max(nosign_0_0, 0x38000000);
  nosign_0_0 = min(nosign_0_0, 0x3ff00000);
  uint32_t nosign_0_1 = nosign0 & 0x0000ffff;
  nosign_0_1 = max(nosign_0_1, 0x3800);
  nosign_0_1 = min(nosign_0_1, 0x3ff0);
  nosign0 = nosign_0_0 | nosign_0_1;

  uint32_t nosign_1_0 = nosign1 & 0xffff0000;
  nosign_1_0 = max(nosign_1_0, 0x38000000);
  nosign_1_0 = min(nosign_1_0, 0x3ff00000);
  uint32_t nosign_1_1 = nosign1 & 0x0000ffff;
  nosign_1_1 = max(nosign_1_1, 0x3800);
  nosign_1_1 = min(nosign_1_1, 0x3ff0);
  nosign1 = nosign_1_0 | nosign_1_1;

  nosign0 = nosign0 + rn_;
  nosign1 = nosign1 + rn_;
  nosign0 = nosign0 - 0x38003800;
  nosign1 = nosign1 - 0x38003800;
  nosign0 = nosign0 >> 4;
  nosign1 = nosign1 >> 4;

  unit32_t nosign = ( (nosign1 & 0x00ff0000) << 8 ) | ( (nosign1 & 0x000000ff) << 16 ) | ( (nosign0 & 0x00ff0000) >> 8 ) |  (nosign0 & 0x000000ff) ;

  return nosign | sign;


}

int main() {

	return 0;
}
