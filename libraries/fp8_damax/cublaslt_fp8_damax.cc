// nvcc -lcublasLt -lcublas cublaslt_fp8_damax.cc
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include "sample_cublasLt_LtFp8Matmul.h"

#ifndef CHECK_CUDA_ERROR
#define CHECK_CUDA_ERROR(error)                    \
    if(error != cudaSuccess)                       \
    {                                             \
        fprintf(stderr,                           \
                "CUDA error: '%s'(%d) at %s:%d\n", \
                cudaGetErrorString(error),         \
                error,                            \
                __FILE__,                         \
                __LINE__);                        \
        exit(EXIT_FAILURE);                       \
    }
#endif

#ifndef CHECK_CUBLASLT_ERROR
#define CHECK_CUBLASLT_ERROR(error)                                                      \
    if(error != CUBLAS_STATUS_SUCCESS)                                                   \
    {                                                                                     \
        fprintf(stderr, "cudaBLASLt error(Err=%s) at %s:%d\n", cublasGetStatusName(error), __FILE__, __LINE__); \
        fprintf(stderr, "\n");                                                            \
        exit(EXIT_FAILURE);                                                               \
    }
#endif



void simpleGemm(cublasLtHandle_t  handle,
                cublasOperation_t trans_a,
                cublasOperation_t trans_b,
                int64_t            m,
                int64_t            n,
                int64_t            k,
                float&             alpha,
                float&             beta,
                void*              d_a,
                void*              d_b,
                void*              d_c,
                void*              d_d,
								void*              d_scale_a,
								void*              d_scale_b,
								void*              d_scale_c,
								void*              d_scale_d,
								void*              d_damax,
                void*              d_workspace,
                cublasLtEpilogue_t epilogue,
                int64_t            max_workspace_size,
                cudaStream_t        stream)
{
    cublasLtMatrixLayout_t matA, matB, matC, matD;
		int64_t lda = trans_a == CUBLAS_OP_N ? m : k;
		int64_t ldb = trans_b == CUBLAS_OP_N ? k : n;
    CHECK_CUBLASLT_ERROR(cublasLtMatrixLayoutCreate(&matA, CUDA_R_8F_E4M3, trans_a == CUBLAS_OP_N ? m : k, trans_a == CUBLAS_OP_N ? k : m, lda));
    CHECK_CUBLASLT_ERROR(cublasLtMatrixLayoutCreate(&matB, CUDA_R_8F_E4M3, trans_b == CUBLAS_OP_N ? k : n, trans_b == CUBLAS_OP_N ? n : k, ldb));
    //CHECK_CUBLASLT_ERROR(cublasLtMatrixLayoutCreate(&matC, CUDA_R_8F_E4M3, m, n, m));
    CHECK_CUBLASLT_ERROR(cublasLtMatrixLayoutCreate(&matC, CUDA_R_16F, m, n, m));
    CHECK_CUBLASLT_ERROR(cublasLtMatrixLayoutCreate(&matD, CUDA_R_8F_E4M3, m, n, m));


    cublasLtMatmulDesc_t matmul;
    CHECK_CUBLASLT_ERROR(
        cublasLtMatmulDescCreate(&matmul, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    CHECK_CUBLASLT_ERROR(cublasLtMatmulDescSetAttribute(
        matmul, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(int32_t)));
    CHECK_CUBLASLT_ERROR(cublasLtMatmulDescSetAttribute(
        matmul, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(int32_t)));


    CHECK_CUBLASLT_ERROR(cublasLtMatmulDescSetAttribute(
        matmul, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));

		CHECK_CUBLASLT_ERROR(cublasLtMatmulDescSetAttribute(matmul,
												  CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
													&d_scale_a, sizeof(d_scale_a)));
		CHECK_CUBLASLT_ERROR(cublasLtMatmulDescSetAttribute(matmul,
												  CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
													&d_scale_b, sizeof(d_scale_b)));
		CHECK_CUBLASLT_ERROR(cublasLtMatmulDescSetAttribute(matmul,
												  CUBLASLT_MATMUL_DESC_C_SCALE_POINTER,
													&d_scale_c, sizeof(d_scale_c)));
		CHECK_CUBLASLT_ERROR(cublasLtMatmulDescSetAttribute(matmul,
												  CUBLASLT_MATMUL_DESC_D_SCALE_POINTER,
													&d_scale_d, sizeof(d_scale_d)));

		CHECK_CUBLASLT_ERROR(cublasLtMatmulDescSetAttribute(matmul,
												  CUBLASLT_MATMUL_DESC_AMAX_D_POINTER,
													&d_damax, sizeof(d_damax)));

    // Set User Preference attributes
    cublasLtMatmulPreference_t pref;
    CHECK_CUBLASLT_ERROR(cublasLtMatmulPreferenceCreate(&pref));
    CHECK_CUBLASLT_ERROR(
        cublasLtMatmulPreferenceSetAttribute(pref,
                                              CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                              &max_workspace_size,
                                              sizeof(max_workspace_size)));
		std::cout << "max_workspace_size=" << max_workspace_size << std::endl;

    const int                        request_solutions = 10;
    cublasLtMatmulHeuristicResult_t heuristicResult[request_solutions];
    int                              returnedAlgoCount = 0;
    CHECK_CUBLASLT_ERROR(cublasLtMatmulAlgoGetHeuristic(handle,
                                                          matmul,
                                                          matA,
                                                          matB,
                                                          matC,
                                                          matD,
                                                          pref,
                                                          request_solutions,
                                                          heuristicResult,
                                                          &returnedAlgoCount));
    std::cout << "Got heuristic results." << std::endl;
    std::cout << "returned Algo Count=" << returnedAlgoCount << std::endl;

    if(returnedAlgoCount == 0)
    {
        std::cerr << "No valid solution found!" << std::endl;
        return;
    }

    uint64_t workspace_size = 4194304;
    for(int i = 0; i < returnedAlgoCount; i++)
        workspace_size = std::max(workspace_size, heuristicResult[i].workspaceSize);
    // In this sample, the workspace is already allocated with max_workspace_size
    // If not, allocate d_workspace here
    // CHECK_HIP_ERRORcudaMalloc(&d_workspace, workspace_size));
		std::cout << "workspace_size=" << workspace_size << std::endl;


    CHECK_CUBLASLT_ERROR(cublasLtMatmul(handle,
                                          matmul,
                                          &alpha,
                                          d_a,
                                          matA,
                                          d_b,
                                          matB,
                                          &beta,
                                          d_c,
                                          matC,
                                          d_d,
                                          matD,
                                          &heuristicResult[0].algo,
                                          d_workspace,
                                          workspace_size,
                                          stream));
    return;
}

void load_binary_from_file(const char* file_name, void* buffer, int num_bytes) {
	std::ifstream fin(file_name, std::ios::binary);
	if (!fin) {
		std::cerr << "Failed to open file " << file_name << std::endl;
	}
	fin.read(reinterpret_cast<char*>(buffer), num_bytes);
}

void print_fp8_buffer(void* a, const char* array_name, int num_elem) {
	 __nv_fp8_storage_t* fp8_buffer = reinterpret_cast<__nv_fp8_storage_t*>(a); 
	 for (int i=0;i<num_elem;i++) {
		 float elem_a = __half2float(__nv_cvt_fp8_to_halfraw(fp8_buffer[i], __NV_E4M3));
		 std::cout << array_name << "[" << i << "]=" << elem_a  << std::endl;
	 }
}

void calculate_reference(void* a, void* b, int m, int n, int k, float scale_a, float scale_b, float scale_d) {
	 // a is transpose in column major == row major, b is column major
	 __nv_fp8_storage_t* a_buffer = reinterpret_cast<__nv_fp8_storage_t*>(a); 
	 __nv_fp8_storage_t* b_buffer = reinterpret_cast<__nv_fp8_storage_t*>(b); 
	 std::vector<float> d_buffer(m*n);
	 float damax =0.;
	 for (int i=0;i<m;i++) 
		 for (int j=0;j<n;j++) {
			 float acc = 0.;
			 for (int l=0;l<k;l++) {
				 __nv_fp8_storage_t elem_a_fp8 = a_buffer[i*k+l];
				 float elem_a = __half2float(__nv_cvt_fp8_to_halfraw(elem_a_fp8, __NV_E4M3));
				 __nv_fp8_storage_t elem_b_fp8 = b_buffer[l+j*k];
				 float elem_b = __half2float(__nv_cvt_fp8_to_halfraw(elem_b_fp8, __NV_E4M3));
				 acc += elem_a * elem_b;
			 }
			 acc *= (scale_a * scale_b);
			 if (damax<fabs(acc))
				 damax = fabs(acc);

			 d_buffer[i+j*m] = __half2float(__nv_cvt_fp8_to_halfraw ( __nv_cvt_float_to_fp8(acc * scale_d, __NV_SATFINITE, __NV_E4M3), __NV_E4M3 ) );

		 }
	for (int i=0;i<m*n;i++)
		std::cout << "d_ref[" << i << "]=" << d_buffer[i] << std::endl;
	std::cout << "damax_ref=" << damax << std::endl;
}

int main()
{
	int m = 16, n = 16, k = 32;
  float alpha = 1.0, beta = 0.0;
	void *a, *b, *d;
	void *d_a, *d_b, *d_d;
	float scale_a, scale_b, scale_d, damax;
	float *d_scale_a, *d_scale_b, *d_scale_d, *d_damax;
	void* workspace;
	size_t workspaceSize = 32*1024*1024;
  cublasLtHandle_t handle;
	{
    CHECK_CUBLASLT_ERROR(cublasLtCreate(&handle));
		CHECK_CUDA_ERROR(cudaMalloc((void**)&d_a, m * k * sizeof(char)));
		CHECK_CUDA_ERROR(cudaMalloc((void**)&d_b, n * k * sizeof(char)));
		CHECK_CUDA_ERROR(cudaMalloc((void**)&d_d, m * n * sizeof(char)));

		CHECK_CUDA_ERROR(cudaMalloc((void**)&d_scale_a, sizeof(float)));
		CHECK_CUDA_ERROR(cudaMalloc((void**)&d_scale_b, sizeof(float)));
		CHECK_CUDA_ERROR(cudaMalloc((void**)&d_scale_d, sizeof(float)));
		CHECK_CUDA_ERROR(cudaMalloc((void**)&d_damax, sizeof(float)));

		CHECK_CUDA_ERROR(cudaMalloc((void**)&workspace, workspaceSize));

		CHECK_CUDA_ERROR(cudaHostAlloc((void**)&a, m * k * sizeof(char), 0));
		CHECK_CUDA_ERROR(cudaHostAlloc((void**)&b, n * k * sizeof(char), 0));
		CHECK_CUDA_ERROR(cudaHostAlloc((void**)&d, m * n * sizeof(char), 0));
		
	}

  
  {
	  load_binary_from_file("a.bin", a, m*k);
	  load_binary_from_file("b.bin", b, n*k);
	  load_binary_from_file("a_scale.bin", &scale_a, sizeof(float));
	  load_binary_from_file("b_scale.bin", &scale_b, sizeof(float));
	  load_binary_from_file("d_scale.bin", &scale_d, sizeof(float));

  }
	{
	 std::cout << "scale_a=" << scale_a 
		         << " scale_b=" << scale_b 
						 << " scale_d=" << scale_d << std::endl;
	 print_fp8_buffer(a, "a", m*k);
	 print_fp8_buffer(b, "b", n*k);
   calculate_reference(a, b,  m, n,  k, scale_a,  scale_b,  scale_d);
	}

  {
		 CHECK_CUDA_ERROR(cudaMemcpy(
				d_a, a, m * k * sizeof(char), cudaMemcpyHostToDevice));
		 CHECK_CUDA_ERROR(cudaMemcpy(
				d_b, b, n * k * sizeof(char), cudaMemcpyHostToDevice));
		 CHECK_CUDA_ERROR(cudaMemcpy(
				d_scale_a, &scale_a, sizeof(float), cudaMemcpyHostToDevice));
		 CHECK_CUDA_ERROR(cudaMemcpy(
				d_scale_b, &scale_b, sizeof(float), cudaMemcpyHostToDevice));
		 CHECK_CUDA_ERROR(cudaMemcpy(
				d_scale_d, &scale_d, sizeof(float), cudaMemcpyHostToDevice));
	}

  simpleGemm(handle,
             CUBLAS_OP_T,
             CUBLAS_OP_N,
             m,
             n,
             k,
             alpha,
             beta,
             d_a,
             d_b,
             d_d,
             d_d,
						 d_scale_a,
						 d_scale_b,
						 d_scale_d,
						 d_scale_d,
						 d_damax,
             workspace,
             CUBLASLT_EPILOGUE_DEFAULT,
						 32*1024*1024,
             0);
/*
  LtFp8Matmul(handle,
             m,
             n,
             k,
             &alpha,
						 d_scale_a,
             (__nv_fp8_e4m3*)d_a,
						 k,//lda
						 d_scale_b,
             (__nv_fp8_e4m3*)d_b,
						 k,//ldb
						 d_scale_d,
             (__nv_fp8_e4m3*)d_d,
						 m,//ldc
						 d_scale_d,
						 d_damax,
             workspace,
						 32*1024*1024);
						 */

  {
	 CHECK_CUDA_ERROR(cudaMemcpy(
			d, d_d, m * n * sizeof(char), cudaMemcpyDeviceToHost));
	 CHECK_CUDA_ERROR(cudaMemcpy(
			&damax, d_damax, sizeof(float), cudaMemcpyDeviceToHost));
	 print_fp8_buffer(d, "d", m*n);
	 std::cout << "damax=" << damax << std::endl;
      
  }

  {
    CHECK_CUDA_ERROR(cudaFreeHost(a));
    CHECK_CUDA_ERROR(cudaFreeHost(b));
    CHECK_CUDA_ERROR(cudaFreeHost(d));
    CHECK_CUDA_ERROR(cudaFree(d_a));
    CHECK_CUDA_ERROR(cudaFree(d_b));
    CHECK_CUDA_ERROR(cudaFree(d_d));
    CHECK_CUDA_ERROR(cudaFree(d_scale_a));
    CHECK_CUDA_ERROR(cudaFree(d_scale_b));
    CHECK_CUDA_ERROR(cudaFree(d_scale_d));
    CHECK_CUDA_ERROR(cudaFree(d_damax));

    CHECK_CUDA_ERROR(cudaFree(workspace));
	}
    CHECK_CUBLASLT_ERROR(cublasLtDestroy(handle));
		return 0;
}
