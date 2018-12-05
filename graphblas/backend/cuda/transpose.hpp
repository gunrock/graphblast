#ifndef GRB_BACKEND_CUDA_TRANSPOSE_HPP
#define GRB_BACKEND_CUDA_TRANSPOSE_HPP

#include <cusparse.h>

#include <iostream>
#include <vector>

#include "graphblas/types.hpp"

namespace graphblas
{
namespace backend
{
  template <typename T>
  class Matrix;

  template <typename a, typename b>
  Info cusparseTranspose( SparseMatrix<b>&       B,
                          const SparseMatrix<a>& A );

  // For testing
  template <typename a, typename b>
  Info transpose( Matrix<b>&       B,
                  const Matrix<a>& A)
  {
    Storage B_storage, A_storage;
    B.getStorage( B_storage );
    A.getStorage( A_storage );

    Info err;
    if( A_storage == GrB_SPARSE )
    {
      err = cusparseTranspose( B.sparse_, A.sparse_ );
    }
    return err;
  }

  template <typename a, typename b>
  Info cusparseTranspose( SparseMatrix<b>&       B,
                          const SparseMatrix<a>& A )
  {
    Index A_nrows, A_ncols, A_nvals;
    Index B_nrows, B_ncols;

    A_nrows = A.nrows_;
    A_ncols = A.ncols_;
    A_nvals = A.nvals_;
    B_nrows = B.nrows_;
    B_ncols = B.ncols_;

    // Dimension compatibility check
    if( (A_ncols != B_nrows) || (A_nrows != B_ncols) )
    {
      std::cout << "Dim mismatch transpose" << std::endl;
      std::cout << A_ncols << " " << B_nrows << std::endl;
      std::cout << A_nrows << " " << B_ncols << std::endl;
      return GrB_DIMENSION_MISMATCH;
    }

    // Matrix transpose
    cusparseHandle_t handle;
    cusparseCreate(&handle);

    cusparseStatus_t status;

    // For CUDA 5.0+
    status = cusparseScsr2csc(
        handle, A_nrows, A_ncols, A_nvals, 
        A.d_csrVal_, A.d_csrRowPtr_, A.d_csrColInd_, 
        B.d_csrVal_, B.d_csrColInd_, B.d_csrRowPtr_, 
        CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO);

    switch( status )
    {
      case CUSPARSE_STATUS_SUCCESS:
        //std::cout << "csr2csc conversion successful!\n";
        break;
      case CUSPARSE_STATUS_NOT_INITIALIZED:
        std::cout << "Error: Library not initialized.\n";
        break;
      case CUSPARSE_STATUS_INVALID_VALUE:
        std::cout << "Error: Invalid parameters m, n, or nnz.\n";
        break;
      case CUSPARSE_STATUS_EXECUTION_FAILED:
        std::cout << "Error: Failed to launch GPU.\n";
        break;
      case CUSPARSE_STATUS_ALLOC_FAILED:
        std::cout << "Error: Resources could not be allocated.\n";
        break;
      case CUSPARSE_STATUS_ARCH_MISMATCH:
        std::cout << "Error: Device architecture does not support.\n";
        break;
      case CUSPARSE_STATUS_INTERNAL_ERROR:
        std::cout << "Error: An internal operation failed.\n";
        break;
    }
    // Important: destroy handle
    cusparseDestroy(handle);

    B.need_update_ = true;
    return GrB_SUCCESS;
  }

}  // backend
}  // graphblas

#endif  // GRB_BACKEND_CUDA_TRANSPOSE_HPP
