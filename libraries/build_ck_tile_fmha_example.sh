git clone https://github.com/ROCm/composable_kernel.git
cd composable_kernel
## Refer to instructions in https://github.com/ROCm/composable_kernel/tree/develop/example/ck_tile/01_fmha#build
# in the root of ck_tile
mkdir build && cd build
# you can replace <arch> with the appropriate architecture (for example gfx90a or gfx942) or leave it blank
#sh ../script/cmake-ck-dev.sh  ../ <arch>
sh ../script/cmake-ck-dev.sh  ../ gfx942 
# How do we know the makefile target. Look at the Makefile generated.
# Or refer to https://github.com/ROCm/composable_kernel/blob/de3c6bf58599c095f464c7c6a5cdf267c32ecf0d/example/ck_tile/01_fmha/CMakeLists.txt#L57-L61
make tile_example_fmha_fwd -j
# build command for bwd
#make tile_example_fmha_bwd -j
