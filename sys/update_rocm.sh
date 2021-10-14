# If we need to remove the previous versions of ROCm, uncomment the following two lines
# sudo apt autoremove --purge miopen-hip miopengemm rocm-clang-ocl rocm-cmake rocm-device-libs rocm-opencl rocm-opencl-dev rocm-profiler rocm-smi rocm-utils rocm_bandwidth_test rocminfo rocrand
# sudo apt autoremove --purge rock-dkms rocm-dev 

# DEB_ROCM_REPO=http://repo.radeon.com/rocm/misc/.ea/apt/.apt_3.7.0.20_priv/
DEB_ROCM_REPO=http://repo.radeon.com/rocm/apt/4.3.1/
wget -qO - $DEB_ROCM_REPO/rocm.gpg.key | sudo apt-key add -
sudo rm -rf /etc/apt/sources.list.d/rocm.list
sudo sh -c  "echo deb [arch=amd64] $DEB_ROCM_REPO xenial main > /etc/apt/sources.list.d/rocm.list"

sudo apt update && sudo apt install rock-dkms -y

# We only install the kernel driver. If we would like to install the entire ROCm package, uncomment the follow lines
# sudo apt update && \
#     sudo apt install -y \
# 	 hipblas \
# 	 miopen-hip \
# 	 miopengemm \
# 	 rocblas \
# 	 rocfft \
# 	 rock-dkms \
# 	 rocm-clang-ocl \
# 	 rocm-cmake \
# 	 rocm-dev \
# 	 rocm-device-libs \
# 	 rocm-dkms \
# 	 rocm-libs \
# 	 rocm-opencl \
# 	 rocm-opencl-dev \
# 	 rocm-profiler \
# 	 rocm-smi \
# 	 rocm-utils \
# 	 rocm_bandwidth_test \
# 	 rocminfo \
# 	 rocrand
    
