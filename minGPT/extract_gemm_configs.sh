#!/bin/bash
# m=1280 k=5120 n=8192 transa=T transb=N A_type=fp32 B_type=fp32 D_type=fp32 bias_type=fp32 grad=0 bias=1 gelu=0 use_fp8=0 A_scale_inverse=1.14794e-40 B_scale_inverse=1.4013e-45 accumulate=0 use_split_accumulator=0 math_sm_count=0
#grep transa $1 | cut -d " " -f  10,11,12,6,7,8,9,4,5 | sort | uniq
grep transa $1 | awk '{print $10,$11,$12,$6,$7,$8,$9,$4,$5}' | sort | uniq
