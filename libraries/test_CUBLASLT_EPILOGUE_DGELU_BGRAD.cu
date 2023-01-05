#include <string>
#include <stdexcept>
#include <cuda.h>
#include <cublasLt.h>
#include <cublas_v2.h>

inline void check_cublas_(cublasStatus_t status) {
    if ( status != CUBLAS_STATUS_SUCCESS ) {
        printf("CUBLAS Error: " + std::string(cublasGetStatusString(status)));
    }
}

#define NVTE_CHECK_CUBLAS(ans) { check_cublas_(ans); }

int main() {
   
    float one = 1.0;
    float zero = 0.0;
    float beta = zero;

    int m = 4, n = 3, k = 2;

    float *A_h, *B_h, *D_h, *pre_gelu_h, *bias_grad_h;
    A_h = (float*)malloc(sizeof(float)*m*k);
    B_h = (float*)malloc(sizeof(float)*n*k);
    D_h = (float*)malloc(sizeof(float)*m*n);
    pre_gelu_h = (float*)malloc(sizeof(float)*m*n);
    bias_grad_h = (float*)malloc(sizeof(float)*m);

    for (int i=0;i<m*k;i++)
	    A_h[i] = 0.1;

    for (int i=0;i<n*k;i++)
	    B_h[i] = 0.1;

    for (int i=0;i<m*n;i++)
	    pre_gelu_h[i] = 0.2;

    float *A, *B, *D, *pre_gelu, *bias_grad;
    cudaMalloc(&A, sizeof(float)*m*k);
    cudaMalloc(&B, sizeof(float)*n*k);
    cudaMalloc(&D, sizeof(float)*m*n);
    cudaMalloc(&pre_gelu, sizeof(float)*m*n);
    cudaMalloc(&bias_grad, sizeof(float)*m);

    int lda = m, ldb = k, ldd = m;
    cublasOperation_t transa = CUBLAS_OP_N; 
    cublasOperation_t transb = CUBLAS_OP_N;
    cudaDataType_t A_type = CUDA_R_32F;
    cudaDataType_t B_type = CUDA_R_32F;
    cudaDataType_t D_type = CUDA_R_32F;
    cudaDataType_t bias_type = CUDA_R_32F;

    cudaMemcpy(A, A_h, sizeof(float)*m*k, cudaMemcpyHostToDevice);
    cudaMemcpy(B, B_h, sizeof(float)*n*k, cudaMemcpyHostToDevice);
    cudaMemcpy(pre_gelu, pre_gelu_h, sizeof(float)*m*n, cudaMemcpyHostToDevice);

    void* workspace;
    size_t workspaceSize = sizeof(float)*2*m*k*n;
    cudaMalloc(&workspace, workspaceSize);


    cublasLtHandle_t handle;
    NVTE_CHECK_CUBLAS(cublasLtCreate(&handle));

    cublasLtMatmulDesc_t       operationDesc = nullptr;
    cublasLtMatrixLayout_t     Adesc = nullptr, Bdesc = nullptr, Ddesc = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;
    int                             returnedResults = 0;
    cublasLtMatmulHeuristicResult_t heuristicResult = {};
    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DGELU_BGRAD;

    int64_t ld_gelumat = (int64_t) ldd;

    cublasComputeType_t gemm_compute_type =  CUBLAS_COMPUTE_32F;


    // Create matrix descriptors. Not setting any extra attributes.
    NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Adesc, A_type,
                                                 transa == CUBLAS_OP_N ? m : k,
                                                 transa == CUBLAS_OP_N ? k : m,
                                                 lda));
    NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Bdesc, B_type,
                                                 transb == CUBLAS_OP_N ? k : n,
                                                 transb == CUBLAS_OP_N ? n : k,
                                                 ldb));
    NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Ddesc, D_type, m, n, ldd));

    NVTE_CHECK_CUBLAS(cublasLtMatmulDescCreate(&operationDesc, gemm_compute_type, CUDA_R_32F));
    NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA,
                                                     &transa, sizeof(transa)));
    NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB,
                                                     &transb, sizeof(transb)));



    NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc,
                                                         CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                         &bias_grad, sizeof(bias_grad)));
    NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
                                operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER,
                                &pre_gelu, sizeof(pre_gelu)));
    NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc,
                                                         CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD,
                                                         &ld_gelumat, sizeof(ld_gelumat)));


    NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc,
                                                     CUBLASLT_MATMUL_DESC_EPILOGUE,
                                                     &epilogue, sizeof(epilogue)));

    NVTE_CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&preference));
    NVTE_CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
                            preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                            &workspaceSize, sizeof(workspaceSize)));

    NVTE_CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(handle, operationDesc, Adesc, Bdesc, Ddesc,
                                                     Ddesc, preference, 1, &heuristicResult,
                                                     &returnedResults));

    if (returnedResults == 0) throw std::runtime_error("Unable to find any suitable algorithms");

    // D = alpha * (A * B) + beta * C
    NVTE_CHECK_CUBLAS(cublasLtMatmul(handle,
                                     operationDesc,
                                     static_cast<const void*>(&one),         /* alpha */
                                     A,                                      /* A */
                                     Adesc,
                                     B,                                      /* B */
                                     Bdesc,
                                     static_cast<const void*>(&beta),        /* beta */
                                     D,                                      /* C */
                                     Ddesc,
                                     D,                                      /* D */
                                     Ddesc,
                                     &heuristicResult.algo,                  /* algo */
                                     workspace,                              /* workspace */
                                     workspaceSize,
                                     0));                               /* stream */

    cudaMemcpy(D_h, D, sizeof(float)*m*n, cudaMemcpyDeviceToHost);
    cudaMemcpy(bias_grad_h, bias_grad, sizeof(float)*m, cudaMemcpyDeviceToHost);

    for (int i=0; i<m*n;i++)
	    printf("D[%d]=%f\n", i, D_h[i]);
    for (int i=0; i<m;i++)
	    printf("bias_grad[%d]=%f\n", i, bias_grad_h[i]);

    NVTE_CHECK_CUBLAS(cublasLtMatmulPreferenceDestroy(preference));
    NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(Ddesc));
    NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(Bdesc));
    NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(Adesc));
    NVTE_CHECK_CUBLAS(cublasLtMatmulDescDestroy(operationDesc));

    free(A_h);
    free(B_h);
    free(D_h);
    free(pre_gelu_h);
    free(bias_grad_h);

    cudaFree(A);
    cudaFree(B);
    cudaFree(D);
    cudaFree(pre_gelu);
    cudaFree(bias_grad);
    cudaFree(workspace);

}
