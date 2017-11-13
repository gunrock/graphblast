#ifndef GRB_BACKEND_APSPIE_SPGEMM_HPP
#define GRB_BACKEND_APSPIE_SPGEMM_HPP

#include <iostream>
#include <vector>

#include <cuda.h>
#include <cusparse.h>

#include "graphblas/backend/apspie/apspie.hpp"
#include "graphblas/backend/apspie/Descriptor.hpp"
#include "graphblas/backend/apspie/SparseMatrix.hpp"

namespace graphblas
{
namespace backend
{

  class Descriptor;

  template <int variant, typename c, typename a, typename b, typename m,
            typename BinaryOpT,      typename SemiringT>
  Info spgemm( SparseMatrix<c>*       C, 
               const SparseMatrix<m>* mask,
               const BinaryOpT*       accum, 
               const SemiringT*       op, 
               const SparseMatrix<a>* A,
               const SparseMatrix<b>* B,
               const Descriptor*      desc )
  {
    std::cout << "Error: Feature not implemented yet!\n";
    return GrB_SUCCESS;
  }

  template<typename c, typename a, typename b>
  Info cusparse_spgemm( SparseMatrix<c>&       C,
                        const SparseMatrix<a>& A,
                        const SparseMatrix<b>& B )
  {
    Index A_nrows, A_ncols, A_nvals;
    Index B_nrows, B_ncols, B_nvals;
    Index C_nrows, C_ncols, C_nvals;

    A_nrows = A.nrows_;
    A_ncols = A.ncols_;
    A_nvals = A.nvals_;
    B_nrows = B.nrows_;
    B_ncols = B.ncols_;
    B_nvals = B.nvals_;
    C_nrows = C.nrows_;
    C_ncols = C.ncols_;

    // Dimension compatibility check
    if( (A_ncols != B_nrows) || (C_ncols != B_ncols) || (C_nrows != A_nrows ) )
    {
      std::cout << "Dim mismatch mxm" << std::endl;
      std::cout << A_ncols << " " << B_nrows << std::endl;
      std::cout << C_ncols << " " << B_ncols << std::endl;
      std::cout << C_nrows << " " << A_nrows << std::endl;
      return GrB_DIMENSION_MISMATCH;
    }

    // SpGEMM Computation
    cusparseHandle_t handle;
    cusparseCreate( &handle );
    cusparseSetPointerMode( handle, CUSPARSE_POINTER_MODE_HOST );

    cusparseMatDescr_t descr;
    cusparseCreateMatDescr( &descr );

    cusparseSetMatType( descr, CUSPARSE_MATRIX_TYPE_GENERAL );
    cusparseSetMatIndexBase( descr, CUSPARSE_INDEX_BASE_ZERO );
    cusparseStatus_t status;

    int baseC;
    int *nnzTotalDevHostPtr = &(C_nvals);
    if( C.d_csrRowPtr_==NULL )
    {
      CUDA( cudaMalloc( &C.d_csrRowPtr_, (A_nrows+1)*sizeof(Index) ));
    }
    /*else
    {
      CUDA( cudaFree(&C.d_csrRowPtr_) );
      CUDA( cudaMalloc( &C.d_csrRowPtr_, (A_nrows+1)*sizeof(Index) ));
    }*/

    if( C.h_csrRowPtr_==NULL )
      C.h_csrRowPtr_ = (Index*)malloc((A_nrows+1)*sizeof(Index));
    /*else
    {
      free( C.h_csrRowPtr_ );
      C.h_csrRowPtr_ = (Index*)malloc((A_nrows+1)*sizeof(Index));
    }*/

    // Analyze
    status = cusparseXcsrgemmNnz( handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        A_nrows, B_ncols, A_ncols,
        descr, A_nvals, A.d_csrRowPtr_, A.d_csrColInd_,
        descr, B_nvals, B.d_csrRowPtr_, B.d_csrColInd_,
        descr, C.d_csrRowPtr_, nnzTotalDevHostPtr );

    switch( status ) {
        case CUSPARSE_STATUS_SUCCESS:
            //std::cout << "mxm analyze successful!\n";
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
        case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            std::cout << "Error: Matrix type not supported.\n";
    }

    if( nnzTotalDevHostPtr != NULL )
      C_nvals = *nnzTotalDevHostPtr;
    else {
      CUDA( cudaMemcpy( &(C_nvals), C.d_csrRowPtr_+A_nrows,
          sizeof(Index), cudaMemcpyDeviceToHost ));
      CUDA( cudaMemcpy( &(baseC), C.d_csrRowPtr_,
          sizeof(Index), cudaMemcpyDeviceToHost ));
      C_nvals -= baseC;
    }

    if( C_nvals > C.ncapacity_ ) {
      std::cout << "Increasing matrix C: " << C.ncapacity_ << " -> " << C_nvals << std::endl;
      C.ncapacity_ = C_nvals*C.kresize_ratio_;
      if( C.d_csrColInd_ != NULL )
      {
        CUDA( cudaFree( C.d_csrColInd_ ));
        CUDA( cudaFree( C.d_csrVal_    ));
      }
      CUDA( cudaMalloc( (void**) &C.d_csrColInd_,
          C.ncapacity_*sizeof(Index) ));
      CUDA( cudaMalloc( (void**) &C.d_csrVal_,
          C.ncapacity_*sizeof(c) ));

      if( C.h_csrColInd_ != NULL )
      {
        free( C.h_csrColInd_ );
        free( C.h_csrVal_ );
      }
      C.h_csrColInd_ = (Index*)malloc(C.ncapacity_*sizeof(Index));
      C.h_csrVal_    = (T*)    malloc(C.ncapacity_*sizeof(T));
    }

    // Compute
    status = cusparseScsrgemm( handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        A_nrows, B_ncols, A_ncols,
        descr, A_nvals, A.d_csrVal_, A.d_csrRowPtr_, A.d_csrColInd_,
        descr, B_nvals, B.d_csrVal_, B.d_csrRowPtr_, B.d_csrColInd_,
        descr,          C.d_csrVal_, C.d_csrRowPtr_, C.d_csrColInd_ );

    switch( status ) {
        case CUSPARSE_STATUS_SUCCESS:
            //std::cout << "mxm compute successful!\n";
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
        case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            std::cout << "Error: Matrix type not supported.\n";
    }

    C.need_update_ = true;  // Set flag that we need to copy data from GPU
    C.nvals_ = C_nvals;     // Update nnz count for C
    return GrB_SUCCESS;
  }

  template<typename c, typename a, typename b>
  Info cusparse_spgemm2( SparseMatrix<c>&       C,
                         const SparseMatrix<a>& A,
                         const SparseMatrix<b>& B )
  {
    Index A_nrows, A_ncols, A_nvals;
    Index B_nrows, B_ncols, B_nvals;
    Index C_nrows, C_ncols, C_nvals;

    A_nrows = A.nrows_;
    A_ncols = A.ncols_;
    A_nvals = A.nvals_;
    B_nrows = B.nrows_;
    B_ncols = B.ncols_;
    B_nvals = B.nvals_;
    C_nrows = C.nrows_;
    C_ncols = C.ncols_;

    // Dimension compatibility check
    if( (A_ncols != B_nrows) || (C_ncols != B_ncols) || (C_nrows != A_nrows ) )
    {
      std::cout << "Dim mismatch mxm2" << std::endl;
      std::cout << A_ncols << " " << B_nrows << std::endl;
      std::cout << C_ncols << " " << B_ncols << std::endl;
      std::cout << C_nrows << " " << A_nrows << std::endl;
      return GrB_DIMENSION_MISMATCH;
    }

    // SpGEMM Computation
    cusparseHandle_t handle;
    cusparseCreate( &handle );

    csrgemm2Info_t info = NULL;
    size_t bufferSize;
    //void *buffer = NULL;
    // nnzTotalDevHostPtr points to host memory
    c alpha = 1.0;
    c beta  = 0.0;
    cusparseSetPointerMode( handle, CUSPARSE_POINTER_MODE_HOST );

    cusparseMatDescr_t descr;
    cusparseCreateMatDescr( &descr );

    cusparseSetMatType( descr, CUSPARSE_MATRIX_TYPE_GENERAL );
    cusparseSetMatIndexBase( descr, CUSPARSE_INDEX_BASE_ZERO );
    cusparseStatus_t status;

    int baseC;
    int *nnzTotalDevHostPtr = &(C_nvals);
    if( C.d_csrRowPtr_==NULL )
    {
      CUDA( cudaMalloc( &C.d_csrRowPtr_, (A_nrows+1)*sizeof(Index) ));
    }
    /*else
    {
      CUDA( cudaFree(&C.d_csrRowPtr_) );
      CUDA( cudaMalloc( &C.d_csrRowPtr_, (A_nrows+1)*sizeof(Index) ));
    }*/

    if( C.h_csrRowPtr_==NULL )
      C.h_csrRowPtr_ = (Index*)malloc((A_nrows+1)*sizeof(Index));
    /*else
    {
      free( C.h_csrRowPtr_ );
      C.h_csrRowPtr_ = (Index*)malloc((A_nrows+1)*sizeof(Index));
    }*/

    // step 1: create an opaque structure
    cusparseCreateCsrgemm2Info(&info);

    // step 2: allocate buffer for csrgemm2Nnz and csrgemm2
    status = cusparseScsrgemm2_bufferSizeExt( handle,
        A_nrows, B_ncols, A_ncols, &alpha,
        descr, A_nvals, A.d_csrRowPtr_, A.d_csrColInd_,
        descr, B_nvals, B.d_csrRowPtr_, B.d_csrColInd_,
        &beta,
        descr, B_nvals, B.d_csrRowPtr_, B.d_csrColInd_,
        info, &bufferSize );
    switch( status ) {
        case CUSPARSE_STATUS_SUCCESS:
            //std::cout << "SpMM successful!\n";
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
        case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            std::cout << "Error: Matrix type not supported.\n";
    }

    if( bufferSize > C.buffer_size_ )
    {
      std::cout << "Increasing buffer " << C.buffer_size_ << " -> " << bufferSize << std::endl;
      C.buffer_size_ = bufferSize*C.kresize_ratio_;
      CUDA( cudaFree(C.buffer_) );
      CUDA( cudaMalloc(&C.buffer_, C.buffer_size_) );
    }

    // Analyze
    status = cusparseXcsrgemm2Nnz( handle,
        A_nrows, B_ncols, A_ncols,
        descr, A_nvals, A.d_csrRowPtr_, A.d_csrColInd_,
        descr, B_nvals, B.d_csrRowPtr_, B.d_csrColInd_,
        descr, B_nvals, B.d_csrRowPtr_, B.d_csrColInd_,
        descr, C.d_csrRowPtr_, nnzTotalDevHostPtr, info, C.buffer_ );

    switch( status ) {
        case CUSPARSE_STATUS_SUCCESS:
            //std::cout << "SpMM successful!\n";
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
        case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            std::cout << "Error: Matrix type not supported.\n";
    }

    if( nnzTotalDevHostPtr != NULL )
      C_nvals = *nnzTotalDevHostPtr;
    else {
      CUDA( cudaMemcpy( &(C_nvals), C.d_csrRowPtr_+A_nrows,
          sizeof(Index), cudaMemcpyDeviceToHost ));
      CUDA( cudaMemcpy( &(baseC), C.d_csrRowPtr_,
          sizeof(Index), cudaMemcpyDeviceToHost ));
      C_nvals -= baseC;
    }

    if( C_nvals > C.ncapacity_ ) {
      std::cout << "Increasing matrix C: " << C.ncapacity_ << " -> " << C_nvals << std::endl;
      if( C.d_csrColInd_ != NULL )
      {
        CUDA( cudaFree( C.d_csrColInd_ ));
        CUDA( cudaFree( C.d_csrVal_    ));
      }
      CUDA( cudaMalloc( (void**) &C.d_csrColInd_,
          C_nvals*sizeof(Index) ));
      CUDA( cudaMalloc( (void**) &C.d_csrVal_,
          C_nvals*sizeof(c) ));

      if( C.h_csrColInd_ != NULL )
      {
        free( C.h_csrColInd_ );
        free( C.h_csrVal_ );
      }
      C.h_csrColInd_ = (Index*)malloc(C_nvals*sizeof(Index));
      C.h_csrVal_    = (T*)    malloc(C_nvals*sizeof(T));

      C.ncapacity_ = C_nvals;
    }

    // Compute
    status = cusparseScsrgemm2( handle,
        A_nrows, B_ncols, A_ncols, &alpha,
        descr, A_nvals, A.d_csrVal_, A.d_csrRowPtr_, A.d_csrColInd_,
        descr, B_nvals, B.d_csrVal_, B.d_csrRowPtr_, B.d_csrColInd_,
        &beta,
        descr, B_nvals, B.d_csrVal_, B.d_csrRowPtr_, B.d_csrColInd_,
        descr,          C.d_csrVal_, C.d_csrRowPtr_, C.d_csrColInd_,
        info,  C.buffer_ );

    switch( status ) {
        case CUSPARSE_STATUS_SUCCESS:
            //std::cout << "SpMM successful!\n";
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
        case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            std::cout << "Error: Matrix type not supported.\n";
    }

    C.need_update_ = true;  // Set flag that we need to copy data from GPU
    C.nvals_ = C_nvals;     // Update nnz count for C
    return GrB_SUCCESS;
  }

}  // graphblas
}  // backend

#endif  // GRB_BACKEND_APSPIE_SPGEMM_HPP
