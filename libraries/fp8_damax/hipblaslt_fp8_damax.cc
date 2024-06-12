// hipcc -lhipblaslt -g -fsanitize=address -fno-omit-frame-pointer    simple_hipblaslt_fp_damax.cc
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_fp8.h>
#include <hipblaslt/hipblaslt.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>

#ifndef CHECK_HIP_ERROR
#define CHECK_HIP_ERROR(error)                    \
    if(error != hipSuccess)                       \
    {                                             \
        fprintf(stderr,                           \
                "Hip error: '%s'(%d) at %s:%d\n", \
                hipGetErrorString(error),         \
                error,                            \
                __FILE__,                         \
                __LINE__);                        \
        exit(EXIT_FAILURE);                       \
    }
#endif

#ifndef CHECK_HIPBLASLT_ERROR
#define CHECK_HIPBLASLT_ERROR(error)                                                      \
    if(error != HIPBLAS_STATUS_SUCCESS)                                                   \
    {                                                                                     \
        fprintf(stderr, "hipBLASLt error(Err=%d) at %s:%d\n", error, __FILE__, __LINE__); \
        fprintf(stderr, "\n");                                                            \
        exit(EXIT_FAILURE);                                                               \
    }
#endif



void simpleGemm(hipblasLtHandle_t  handle,
                hipblasOperation_t trans_a,
                hipblasOperation_t trans_b,
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
                hipblasLtEpilogue_t epilogue,
                int64_t            max_workspace_size,
                hipStream_t        stream)
{
    hipblasLtMatrixLayout_t matA, matB, matC, matD;
		int64_t lda = trans_a == HIPBLAS_OP_N ? m : k;
		int64_t ldb = trans_b == HIPBLAS_OP_N ? k : n;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matA, HIP_R_8F_E4M3_FNUZ, trans_a == HIPBLAS_OP_N ? m : k, trans_a == HIPBLAS_OP_N ? k : m, lda));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matB, HIP_R_8F_E4M3_FNUZ, trans_b == HIPBLAS_OP_N ? k : n, trans_b == HIPBLAS_OP_N ? n : k, ldb));
    //CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matA, HIP_R_8F_E4M3_FNUZ, m, k, m));
    //CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matB, HIP_R_8F_E4M3_FNUZ, k, n, k));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matC, HIP_R_8F_E4M3_FNUZ, m, n, m));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matD, HIP_R_8F_E4M3_FNUZ, m, n, m));


    hipblasLtMatmulDesc_t matmul;
    CHECK_HIPBLASLT_ERROR(
        hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(int32_t)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(int32_t)));


    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));


		CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(matmul,
												  HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER,
													&d_scale_a, sizeof(d_scale_a)));
		CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(matmul,
												  HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER,
													&d_scale_b, sizeof(d_scale_b)));
		CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(matmul,
												  HIPBLASLT_MATMUL_DESC_C_SCALE_POINTER,
													&d_scale_c, sizeof(d_scale_c)));
		CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(matmul,
												  HIPBLASLT_MATMUL_DESC_D_SCALE_POINTER,
													&d_scale_d, sizeof(d_scale_d)));

		CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(matmul,
												  HIPBLASLT_MATMUL_DESC_AMAX_D_POINTER,
													&d_damax, sizeof(d_damax)));


    // Set User Preference attributes
    hipblasLtMatmulPreference_t pref;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceCreate(&pref));
    CHECK_HIPBLASLT_ERROR(
        hipblasLtMatmulPreferenceSetAttribute(pref,
                                              HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                              &max_workspace_size,
                                              sizeof(max_workspace_size)));
		std::cout << "max_workspace_size=" << max_workspace_size << std::endl;

    const int                        request_solutions = 10;
    hipblasLtMatmulHeuristicResult_t heuristicResult[request_solutions];
    int                              returnedAlgoCount = 0;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulAlgoGetHeuristic(handle,
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
        workspace_size = max(workspace_size, heuristicResult[i].workspaceSize);
    // In this sample, the workspace is already allocated with max_workspace_size
    // If not, allocate d_workspace here
    // CHECK_HIP_ERRORhipMalloc(&d_workspace, workspace_size));
		std::cout << "workspace_size=" << workspace_size << std::endl;


    CHECK_HIPBLASLT_ERROR(hipblasLtMatmul(handle,
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
	 __hip_fp8_storage_t* fp8_buffer = reinterpret_cast<__hip_fp8_storage_t*>(a); 
	 for (int i=0;i<num_elem;i++) {
		 __hip_fp8_e4m3_fnuz elem_a_fp8;
		 elem_a_fp8.__x = fp8_buffer[i];
		 float elem_a = static_cast<float>(elem_a_fp8);
		 std::cout << array_name << "[" << i << "]=" << elem_a  << std::endl;
	 }
}

void calculate_reference(void* a, void* b, int m, int n, int k, float scale_a, float scale_b, float scale_d) {
	 // a is transpose in column major == row major, b is column major
	 __hip_fp8_storage_t* a_buffer = reinterpret_cast<__hip_fp8_storage_t*>(a); 
	 __hip_fp8_storage_t* b_buffer = reinterpret_cast<__hip_fp8_storage_t*>(b); 
	 std::vector<float> d_buffer(m*n);
	 float damax =0.;
	 for (int i=0;i<m;i++) 
		 for (int j=0;j<n;j++) {
			 float acc = 0.;
			 for (int l=0;l<k;l++) {
				 __hip_fp8_e4m3_fnuz elem_a_fp8;
				 elem_a_fp8.__x = a_buffer[i*k+l];
				 float elem_a = static_cast<float>(elem_a_fp8);
				 __hip_fp8_e4m3_fnuz elem_b_fp8;
				 elem_b_fp8.__x = b_buffer[l+j*k];
				 float elem_b = static_cast<float>(elem_b_fp8);
				 acc += elem_a * elem_b;
			 }
			 acc *= (scale_a * scale_b);
			 if (damax<fabs(acc))
				 damax = fabs(acc);

			 d_buffer[i+j*m] = static_cast<float>( static_cast<__hip_fp8_e4m3_fnuz>(acc * scale_d) );

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
	size_t workspaceSize = 4194304;
  hipblasLtHandle_t handle;
	{
    CHECK_HIPBLASLT_ERROR(hipblasLtCreate(&handle));
		CHECK_HIP_ERROR(hipMalloc(&d_a, m * k * sizeof(char)));
		CHECK_HIP_ERROR(hipMalloc(&d_b, n * k * sizeof(char)));
		CHECK_HIP_ERROR(hipMalloc(&d_d, m * n * sizeof(char)));

		CHECK_HIP_ERROR(hipMalloc(&d_scale_a, sizeof(float)));
		CHECK_HIP_ERROR(hipMalloc(&d_scale_b, sizeof(float)));
		CHECK_HIP_ERROR(hipMalloc(&d_scale_d, sizeof(float)));
		CHECK_HIP_ERROR(hipMalloc(&d_damax, sizeof(float)));

		CHECK_HIP_ERROR(hipHostMalloc(&a, m * k * sizeof(char)));
		CHECK_HIP_ERROR(hipHostMalloc(&b, n * k * sizeof(char)));
		CHECK_HIP_ERROR(hipHostMalloc(&d, m * n * sizeof(char)));
		CHECK_HIP_ERROR(hipMalloc(&workspace, workspaceSize));
		
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
		 CHECK_HIP_ERROR(hipMemcpy(
				d_a, a, m * k * sizeof(char), hipMemcpyHostToDevice));
		 CHECK_HIP_ERROR(hipMemcpy(
				d_b, b, n * k * sizeof(char), hipMemcpyHostToDevice));
		 CHECK_HIP_ERROR(hipMemcpy(
				d_scale_a, &scale_a, sizeof(float), hipMemcpyHostToDevice));
		 CHECK_HIP_ERROR(hipMemcpy(
				d_scale_b, &scale_b, sizeof(float), hipMemcpyHostToDevice));
		 CHECK_HIP_ERROR(hipMemcpy(
				d_scale_d, &scale_d, sizeof(float), hipMemcpyHostToDevice));
	}

  simpleGemm(handle,
             HIPBLAS_OP_T,
             HIPBLAS_OP_N,
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
             HIPBLASLT_EPILOGUE_DEFAULT,
						 4194304,
             0);


  {
	 CHECK_HIP_ERROR(hipMemcpy(
			d, d_d, m * n * sizeof(char), hipMemcpyDeviceToHost));
	 CHECK_HIP_ERROR(hipMemcpy(
			&damax, d_damax, sizeof(float), hipMemcpyDeviceToHost));
	 print_fp8_buffer(d, "d", m*n);
	 std::cout << "damax=" << damax << std::endl;
      
  }

  {
    CHECK_HIP_ERROR(hipFree(a));
    CHECK_HIP_ERROR(hipFree(b));
    CHECK_HIP_ERROR(hipFree(d));
    CHECK_HIP_ERROR(hipFree(d_a));
    CHECK_HIP_ERROR(hipFree(d_b));
    CHECK_HIP_ERROR(hipFree(d_d));
    CHECK_HIP_ERROR(hipFree(d_scale_a));
    CHECK_HIP_ERROR(hipFree(d_scale_b));
    CHECK_HIP_ERROR(hipFree(d_scale_d));
    CHECK_HIP_ERROR(hipFree(d_damax));

    CHECK_HIP_ERROR(hipFree(workspace));
	}
    CHECK_HIPBLASLT_ERROR(hipblasLtDestroy(handle));
		return 0;
}
