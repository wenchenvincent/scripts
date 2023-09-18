// hipcc -lhipblaslt -g -fsanitize=address -fno-omit-frame-pointer    simple_hipblaslt_dot_add_bias.cc
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hipblaslt/hipblaslt.h>
#include <iostream>

//#define HIPBLASLT_R_16F HIPBLAS_R_16F
//#define HIPBLASLT_R_32F HIPBLAS_R_32F

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
                void*              d_bias,
                void*              d_workspace,
                hipblasLtEpilogue_t epilogue,
                int64_t            max_workspace_size,
                hipStream_t        stream)
{
    hipblasLtMatrixLayout_t matA, matB, matC, matD;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matA, HIPBLASLT_R_16F, m, k, m));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matB, HIPBLASLT_R_16F, k, n, k));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matC, HIPBLASLT_R_16F, m, n, m));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matD, HIPBLASLT_R_16F, m, n, m));


    hipblasLtMatmulDesc_t matmul;
    CHECK_HIPBLASLT_ERROR(
        hipblasLtMatmulDescCreate(&matmul, HIPBLASLT_COMPUTE_F32, HIPBLASLT_R_32F));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(int32_t)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(int32_t)));

    //hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_DEFAULT;
    //epilogue = HIPBLASLT_EPILOGUE_DEFAULT;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));

    if (epilogue == HIPBLASLT_EPILOGUE_BIAS) {
			CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
					matmul, HIPBLASLT_MATMUL_DESC_BIAS_POINTER, &d_bias, sizeof(d_bias)));
    }

    // Set User Preference attributes
    hipblasLtMatmulPreference_t pref;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceCreate(&pref));
    CHECK_HIPBLASLT_ERROR(
        hipblasLtMatmulPreferenceSetAttribute(pref,
                                              HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                              &max_workspace_size,
                                              sizeof(max_workspace_size)));
    std::cout << "Set Preference." << std::endl;

    const int                        request_solutions = 1;
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

    if(returnedAlgoCount == 0)
    {
        std::cerr << "No valid solution found!" << std::endl;
        return;
    }

    uint64_t workspace_size = 0;
    for(int i = 0; i < returnedAlgoCount; i++)
        workspace_size = max(workspace_size, heuristicResult[i].workspaceSize);
    // In this sample, the workspace is already allocated with max_workspace_size
    // If not, allocate d_workspace here
    // CHECK_HIP_ERRORhipMalloc(&d_workspace, workspace_size));

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

using InType = __half;
using OutType = __half;
int main()
{
	int m = 2, n = 2, k = 2;
  float alpha = 1.0, beta = 0.0;
	InType *a, *b, *d, *bias;
	InType *d_a, *d_b, *d_d, *d_bias;
	void* workspace;
	size_t workspaceSize = sizeof(float)*16*m*k*n;
  hipblasLtHandle_t handle;
	{
    CHECK_HIPBLASLT_ERROR(hipblasLtCreate(&handle));
		CHECK_HIP_ERROR(hipMalloc(&d_a, m * k * sizeof(InType)));
		CHECK_HIP_ERROR(hipMalloc(&d_b, n * k * sizeof(InType)));
		CHECK_HIP_ERROR(hipMalloc(&d_d, m * n * sizeof(OutType)));
		CHECK_HIP_ERROR(hipMalloc(&d_bias, m * sizeof(InType)));

		CHECK_HIP_ERROR(hipHostMalloc(&a, m * k * sizeof(InType)));
		CHECK_HIP_ERROR(hipHostMalloc(&b, n * k * sizeof(InType)));
		CHECK_HIP_ERROR(hipHostMalloc(&d, m * n * sizeof(OutType)));
		CHECK_HIP_ERROR(hipHostMalloc(&bias, m * sizeof(InType)));
		CHECK_HIP_ERROR(hipHostMalloc(&workspace, workspaceSize));
		
	}
  
  {
    a[0] = static_cast<InType>(10.0);
    a[1] = static_cast<InType>(12.0);
    a[2] = static_cast<InType>(11.0);
    a[3] = static_cast<InType>(13.0);
    
    b[0] = static_cast<InType>(1.0);
    b[1] = static_cast<InType>(3.0);
    b[2] = static_cast<InType>(2.0);
    b[3] = static_cast<InType>(4.0);

    bias[0] = static_cast<InType>(1.0);
    bias[1] = static_cast<InType>(2.0);


  }

  {
		 CHECK_HIP_ERROR(hipMemcpy(
				d_a, a, m * k * sizeof(InType), hipMemcpyHostToDevice));
		 CHECK_HIP_ERROR(hipMemcpy(
				d_b, b, n * k * sizeof(InType), hipMemcpyHostToDevice));
		 CHECK_HIP_ERROR(hipMemcpy(
				d_bias, bias, m * sizeof(InType), hipMemcpyHostToDevice));
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
             d_bias,
             workspace,
             HIPBLASLT_EPILOGUE_BIAS,
             32*1024*1024,
             0);


  {
	 CHECK_HIP_ERROR(hipMemcpy(
			d, d_d, m * n * sizeof(OutType), hipMemcpyDeviceToHost));
   hipDeviceSynchronize();
   for (int i=0;i<m;i++) {
     for (int j=0;j<n;j++) {
				std::cout << static_cast<float>(d[i*n+j]) << " ";
			}
      std::cout << std::endl;
		}
      
  }
  {
    CHECK_HIP_ERROR(hipFree(a));
    CHECK_HIP_ERROR(hipFree(b));
    CHECK_HIP_ERROR(hipFree(d));
    CHECK_HIP_ERROR(hipFree(bias));
    CHECK_HIP_ERROR(hipFree(d_a));
    CHECK_HIP_ERROR(hipFree(d_b));
    CHECK_HIP_ERROR(hipFree(d_d));
    CHECK_HIP_ERROR(hipFree(d_bias));
    CHECK_HIP_ERROR(hipFree(workspace));
	}
    CHECK_HIPBLASLT_ERROR(hipblasLtDestroy(handle));
		return 0;
}
