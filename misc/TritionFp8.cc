
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
  uint32_t fp8X4 = (a1 & 0xff000000) |  ( (a1 & 0x0000ff00) << 8 ) | ( (a0 & 0xff000000) >> 16 ) | ( (a0 & 0x0000ff00) >> 8 );
  return fp8X4;
}

struct fp16X4vec {   
   uint32_t fp16X2_0;
   uint32_t fp16X2_1;
};

__device__
fp16X4vec cast_fp8X4_to_fp16X4(uint8_t fp8X4) {
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

int main() {

	return 0;
}
