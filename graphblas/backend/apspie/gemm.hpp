#ifndef GRB_BACKEND_APSPIE_GEMM_HPP
#define GRB_BACKEND_APSPIE_GEMM_HPP

#include <iostream>

#include <cuda.h>
//#include <cublas.h>
#include <cublas_v2.h>

#include "graphblas/types.hpp"
#include "graphblas/util.hpp"

//#define TA     32
//#define TB     32
//#define NT     64

namespace graphblas
{
namespace backend
{
  template<typename c, typename a, typename b>
  Info gemm( DenseMatrix<c>&       C,
             const Semiring&       op,
             const DenseMatrix<a>& A,
             const DenseMatrix<b>& B )
  {
    Index A_nrows, A_ncols;
    Index B_nrows, B_ncols;
    Index C_nrows, C_ncols;

    A.nrows( A_nrows );
    A.ncols( A_ncols );
    B.nrows( B_nrows );
    B.ncols( B_ncols );
    C.nrows( C_nrows );
    C.ncols( C_ncols );

    // Dimension compatibility check
    if( (A_ncols != B_nrows) || (C_ncols != B_ncols) || (C_nrows != A_nrows ) )
    {
      std::cout << "Dim mismatch" << std::endl;
      std::cout << A_ncols << " " << B_nrows << std::endl;
      std::cout << C_ncols << " " << B_ncols << std::endl;
      std::cout << C_nrows << " " << A_nrows << std::endl;
      return GrB_DIMENSION_MISMATCH;
    }

    // Domain compatibility check
    // TODO: add domain compatibility check

    // Computation
    //cublasInit();

    cublasHandle_t handle;
    cublasCreate( &handle );
    cublasSetPointerMode( handle, CUBLAS_POINTER_MODE_HOST );
    cublasStatus_t status;

    float alpha = 1.0;
    float beta  = 0.0;
    status = cublasSgemm( handle, CUBLAS_OP_N, CUBLAS_OP_N,
                          A_nrows, B_ncols, A_ncols,
                          &alpha,
                          A.d_denseVal_, A_nrows,
                          B.d_denseVal_, B_nrows,
                          &beta,
                          C.d_denseVal_, A_nrows);

    switch( status ) {
        case CUBLAS_STATUS_SUCCESS:
            //std::cout << "SpMM successful!\n";
            break;
        case CUBLAS_STATUS_NOT_INITIALIZED:
            std::cout << "Error: Library not initialized.\n";
            break;
        case CUBLAS_STATUS_INVALID_VALUE:
            std::cout << "Error: Invalid parameters m, n, or nnz.\n";
            break;
        case CUBLAS_STATUS_EXECUTION_FAILED:
            std::cout << "Error: Failed to launch GPU.\n";
            break;
        case CUBLAS_STATUS_ARCH_MISMATCH:
            std::cout << "Error: Device architecture does not support.\n";
            break;
    }

    //cublasShutdown();
    C.need_update_ = true;  // Set flag that we need to copy data from GPU
    return GrB_SUCCESS;
  }

}  // backend
}  // graphblas

#endif  // GRB_BACKEND_APSPIE_GEMM_HPP
