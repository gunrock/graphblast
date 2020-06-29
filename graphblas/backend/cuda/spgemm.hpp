#ifndef GRAPHBLAS_BACKEND_CUDA_SPGEMM_HPP_
#define GRAPHBLAS_BACKEND_CUDA_SPGEMM_HPP_

#include "graphblas/backend/cuda/sparse_matrix.hpp"

#include <cuda.h>
#include <cusparse.h>

#include <iostream>
#include <vector>

namespace graphblas {
namespace backend {
template <typename T>
class SparseMatrix;

template <typename T>
class DenseMatrix;

template <typename c, typename a, typename b, typename m,
          typename BinaryOpT,     typename SemiringT>
Info spgemmMasked(SparseMatrix<c>*       C,
                  const Matrix<m>*       mask,
                  BinaryOpT              accum,
                  SemiringT              op,
                  const SparseMatrix<a>* A,
                  const SparseMatrix<b>* B,
                  Descriptor*            desc) {
  Desc_value scmp_mode, inp0_mode, inp1_mode;
  CHECK(desc->get(GrB_MASK, &scmp_mode));
  CHECK(desc->get(GrB_INP0, &inp0_mode));
  CHECK(desc->get(GrB_INP1, &inp1_mode));

  bool use_mask   = (mask != NULL);
  bool use_scmp   = (scmp_mode == GrB_SCMP);
  bool use_tran_A = inp0_mode == GrB_TRAN;
  bool use_tran_B = inp1_mode == GrB_TRAN;

  // A Transpose (default is CSR):
  const Index* A_csrRowPtr = (use_tran_A) ? A->d_cscColPtr_ : A->d_csrRowPtr_;
  const Index* A_csrColInd = (use_tran_A) ? A->d_cscRowInd_ : A->d_csrColInd_;
  const a*     A_csrVal    = (use_tran_A) ? A->d_cscVal_    : A->d_csrVal_;
  const Index  A_nrows     = (use_tran_A) ? A->ncols_       : A->nrows_;

  if (desc->debug()) {
    std::cout << "cscColPtr: " << A->d_cscColPtr_ << std::endl;
    std::cout << "cscRowInd: " << A->d_cscRowInd_ << std::endl;
    std::cout << "cscVal:    " << A->d_cscVal_    << std::endl;

    std::cout << "csrRowPtr: " << A->d_csrRowPtr_ << std::endl;
    std::cout << "csrColInd: " << A->d_csrColInd_ << std::endl;
    std::cout << "csrVal:    " << A->d_csrVal_    << std::endl;
  }

  // B Transpose (default is CSC)
  const Index* B_cscColPtr = (use_tran_B) ? B->d_csrRowPtr_ : B->d_cscColPtr_;
  const Index* B_cscRowInd = (use_tran_B) ? B->d_csrColInd_ : B->d_cscRowInd_;
  const b*     B_cscVal    = (use_tran_B) ? B->d_csrVal_    : B->d_cscVal_;
  const Index  B_nrows     = (use_tran_B) ? B->ncols_       : B->nrows_;

  if (desc->debug()) {
    std::cout << "cscColPtr: " << B->d_cscColPtr_ << std::endl;
    std::cout << "cscRowInd: " << B->d_cscRowInd_ << std::endl;
    std::cout << "cscVal:    " << B->d_cscVal_    << std::endl;

    std::cout << "csrRowPtr: " << B->d_csrRowPtr_ << std::endl;
    std::cout << "csrColInd: " << B->d_csrColInd_ << std::endl;
    std::cout << "csrVal:    " << B->d_csrVal_    << std::endl;
  }

  // Get descriptor parameters for nthreads
  Desc_value nt_mode;
  CHECK(desc->get(GrB_NT, &nt_mode));
  const int nt = static_cast<int>(nt_mode);

  if (use_mask) {
    Storage mask_mat_type;
    CHECK(mask->getStorage(&mask_mat_type));
    if (mask_mat_type == GrB_DENSE) {
      std::cout << "SpGEMM with dense mask\n";
      std::cout << "Error: Feature not implemented yet!\n";
    } else {
      // C must share sparsity of pattern of mask
      if (C != A && C != B)
        CHECK(C->dup(&mask->sparse_));

      const SparseMatrix<m>* sparse_mask = &mask->sparse_;

      // Simple warp-per-row algorithm
      dim3 NT, NB;
      NT.x = nt;
      NT.y = 1;
      NT.z = 1;
      NB.x = (A_nrows + nt - 1) / nt * 32;
      NB.y = 1;
      NB.z = 1;

      spgemmMaskedKernel<<<NB, NT>>>(C->d_csrVal_, sparse_mask->d_csrRowPtr_,
          sparse_mask->d_csrColInd_, sparse_mask->d_csrVal_,
          extractMul(op), extractAdd(op), op.identity(),
          A_csrRowPtr, A_csrColInd, A_csrVal,
          B_cscColPtr, B_cscRowInd, B_cscVal,
          A_nrows);
    }
  }
  return GrB_SUCCESS;
}

template <typename c, typename a, typename b, typename m,
          typename BinaryOpT,     typename SemiringT>
Info cusparse_spgemm(SparseMatrix<c>*       C,
                     const Matrix<m>*       mask,
                     BinaryOpT              accum,
                     SemiringT              op,
                     const SparseMatrix<a>* A,
                     const SparseMatrix<b>* B,
                     Descriptor*            desc) {
  Index A_nrows, A_ncols, A_nvals;
  Index B_nrows, B_ncols, B_nvals;
  Index C_nrows, C_ncols, C_nvals;

  A_nrows = A->nrows_;
  A_ncols = A->ncols_;
  A_nvals = A->nvals_;
  B_nrows = B->nrows_;
  B_ncols = B->ncols_;
  B_nvals = B->nvals_;
  C_nrows = C->nrows_;
  C_ncols = C->ncols_;

  // Dimension compatibility check
  if ((A_ncols != B_nrows) || (C_ncols != B_ncols) || (C_nrows != A_nrows)) {
    std::cout << "Dim mismatch mxm" << std::endl;
    std::cout << A_ncols << " " << B_nrows << std::endl;
    std::cout << C_ncols << " " << B_ncols << std::endl;
    std::cout << C_nrows << " " << A_nrows << std::endl;
    return GrB_DIMENSION_MISMATCH;
  }

  // SpGEMM Computation
  cusparseHandle_t handle;
  cusparseCreate(&handle);
  cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);

  cusparseMatDescr_t descr;
  cusparseCreateMatDescr(&descr);

  cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
  cusparseStatus_t status;

  int baseC;
  int *nnzTotalDevHostPtr = &(C_nvals);
  if (C->d_csrRowPtr_ == NULL) {
    CUDA_CALL( cudaMalloc( &C->d_csrRowPtr_, (A_nrows+1)*sizeof(Index) ));
  }
  /*else
  {
    CUDA_CALL( cudaFree(&C->d_csrRowPtr_) );
    CUDA_CALL( cudaMalloc( &C->d_csrRowPtr_, (A_nrows+1)*sizeof(Index) ));
  }*/

  if (C->h_csrRowPtr_ == NULL)
    C->h_csrRowPtr_ = reinterpret_cast<Index*>(malloc((A_nrows+1)*
        sizeof(Index)));
  /*else
  {
    free( C->h_csrRowPtr_ );
    C->h_csrRowPtr_ = (Index*)malloc((A_nrows+1)*sizeof(Index));
  }*/

  // Analyze
  status = cusparseXcsrgemmNnz(handle,
      CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
      A_nrows, B_ncols, A_ncols,
      descr, A_nvals, A->d_csrRowPtr_, A->d_csrColInd_,
      descr, B_nvals, B->d_csrRowPtr_, B->d_csrColInd_,
      descr, C->d_csrRowPtr_, nnzTotalDevHostPtr);

  switch (status) {
    case CUSPARSE_STATUS_SUCCESS:
      // std::cout << "mxm analyze successful!\n";
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

  if (nnzTotalDevHostPtr != NULL) {
    C_nvals = *nnzTotalDevHostPtr;
  } else {
    CUDA_CALL(cudaMemcpy( &C_nvals, C->d_csrRowPtr_+A_nrows, sizeof(Index),
        cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy( &baseC, C->d_csrRowPtr_, sizeof(Index),
        cudaMemcpyDeviceToHost));
    C_nvals -= baseC;
  }

  if (C_nvals > C->ncapacity_) {
    if (desc->debug())
      std::cout << "Increasing matrix C: " << C->ncapacity_ << " -> " << C_nvals << std::endl;
    C->ncapacity_ = C_nvals*C->kresize_ratio_;
    if (C->d_csrColInd_ != NULL) {
      CUDA_CALL(cudaFree(C->d_csrColInd_));
      CUDA_CALL(cudaFree(C->d_csrVal_));
    }
    CUDA_CALL(cudaMalloc(&C->d_csrColInd_, C->ncapacity_*sizeof(Index)));
    CUDA_CALL(cudaMalloc(&C->d_csrVal_, C->ncapacity_*sizeof(c)));

    if (C->h_csrColInd_ != NULL) {
      free(C->h_csrColInd_);
      free(C->h_csrVal_);
    }
    C->h_csrColInd_ = reinterpret_cast<Index*>(malloc(C->ncapacity_*sizeof(
        Index)));
    C->h_csrVal_    = reinterpret_cast<T*>(malloc(C->ncapacity_*sizeof(T)));
  }

  // Compute
  status = cusparseScsrgemm(handle,
      CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
      A_nrows, B_ncols, A_ncols,
      descr, A_nvals, A->d_csrVal_, A->d_csrRowPtr_, A->d_csrColInd_,
      descr, B_nvals, B->d_csrVal_, B->d_csrRowPtr_, B->d_csrColInd_,
      descr,          C->d_csrVal_, C->d_csrRowPtr_, C->d_csrColInd_);

  switch (status) {
    case CUSPARSE_STATUS_SUCCESS:
      // std::cout << "mxm compute successful!\n";
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

  C->need_update_ = true;  // Set flag that we need to copy data from GPU
  C->nvals_ = C_nvals;     // Update nnz count for C
  return GrB_SUCCESS;
}

template <typename c, typename a, typename b, typename m,
          typename BinaryOpT,     typename SemiringT>
Info cusparse_spgemm2(SparseMatrix<c>*       C,
                      const Matrix<m>*       mask,
                      BinaryOpT              accum,
                      SemiringT              op,
                      const SparseMatrix<a>* A,
                      const SparseMatrix<b>* B,
                      Descriptor*            desc) {
  return GrB_NOT_IMPLEMENTED;
}

template <typename m, typename BinaryOpT, typename SemiringT>
Info cusparse_spgemm2(SparseMatrix<float>*       C,
                      const Matrix<m>*           mask,
                      BinaryOpT                  accum,
                      SemiringT                  op,
                      const SparseMatrix<float>* A,
                      const SparseMatrix<float>* B,
                      Descriptor*                desc) {
  Index A_nrows, A_ncols, A_nvals;
  Index B_nrows, B_ncols, B_nvals;
  Index C_nrows, C_ncols, C_nvals;

  A_nrows = A->nrows_;
  A_ncols = A->ncols_;
  A_nvals = A->nvals_;
  B_nrows = B->nrows_;
  B_ncols = B->ncols_;
  B_nvals = B->nvals_;
  C_nrows = C->nrows_;
  C_ncols = C->ncols_;

  // Dimension compatibility check
  if ((A_ncols != B_nrows) || (C_ncols != B_ncols) || (C_nrows != A_nrows)) {
    std::cout << "Dim mismatch mxm2" << std::endl;
    std::cout << A_ncols << " " << B_nrows << std::endl;
    std::cout << C_ncols << " " << B_ncols << std::endl;
    std::cout << C_nrows << " " << A_nrows << std::endl;
    return GrB_DIMENSION_MISMATCH;
  }

  // SpGEMM Computation
  cusparseHandle_t handle;
  cusparseCreate(&handle);

  csrgemm2Info_t info = NULL;
  size_t bufferSize;

  // nnzTotalDevHostPtr points to host memory
  float alpha = 1.0;
  float* beta = NULL;
  cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);

  cusparseMatDescr_t descr;
  cusparseCreateMatDescr(&descr);

  cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
  cusparseStatus_t status;

  int baseC;
  int *nnzTotalDevHostPtr = &(C_nvals);
  if (C->d_csrRowPtr_ == NULL) {
    CUDA_CALL(cudaMalloc(&C->d_csrRowPtr_, (A_nrows+1)*sizeof(Index)));
  }
  /*else
  {
    CUDA_CALL( cudaFree(&C.d_csrRowPtr_) );
    CUDA_CALL( cudaMalloc( &C.d_csrRowPtr_, (A_nrows+1)*sizeof(Index) ));
  }*/

  if (C->h_csrRowPtr_ == NULL)
    C->h_csrRowPtr_ = reinterpret_cast<Index*>(malloc((A_nrows+1)*sizeof(
        Index)));
  /*else
  {
    free( C.h_csrRowPtr_ );
    C.h_csrRowPtr_ = (Index*)malloc((A_nrows+1)*sizeof(Index));
  }*/

  // Step 1: create an opaque structure
  cusparseCreateCsrgemm2Info(&info);

  // Step 2: allocate buffer for csrgemm2Nnz and csrgemm2
  status = cusparseScsrgemm2_bufferSizeExt(handle,
      A_nrows, B_ncols, A_ncols, &alpha,
      descr, A_nvals, A->d_csrRowPtr_, A->d_csrColInd_,
      descr, B_nvals, B->d_csrRowPtr_, B->d_csrColInd_,
      beta,
      descr, B_nvals, B->d_csrRowPtr_, B->d_csrColInd_,
      info, &bufferSize);
  switch (status) {
    case CUSPARSE_STATUS_SUCCESS:
      // std::cout << "SpMM successful!\n";
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

  if (bufferSize > desc->d_buffer_size_)
    desc->resize(bufferSize, "buffer");

  // Analyze
  status = cusparseXcsrgemm2Nnz(handle,
      A_nrows, B_ncols, A_ncols,
      descr, A_nvals, A->d_csrRowPtr_, A->d_csrColInd_,
      descr, B_nvals, B->d_csrRowPtr_, B->d_csrColInd_,
      descr, B_nvals, B->d_csrRowPtr_, B->d_csrColInd_,
      descr, C->d_csrRowPtr_, nnzTotalDevHostPtr, info, desc->d_buffer_);

  switch (status) {
    case CUSPARSE_STATUS_SUCCESS:
      // std::cout << "SpMM successful!\n";
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

  if (nnzTotalDevHostPtr != NULL) {
    C_nvals = *nnzTotalDevHostPtr;
  } else {
    CUDA_CALL(cudaMemcpy(&(C_nvals), C->d_csrRowPtr_+A_nrows, sizeof(Index),
        cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(&(baseC), C->d_csrRowPtr_, sizeof(Index),
        cudaMemcpyDeviceToHost));
    C_nvals -= baseC;
  }

  if (C_nvals > C->ncapacity_) {
    if (desc->debug())
      std::cout << "Increasing matrix C: " << C->ncapacity_ << " -> " << C_nvals << std::endl;
    if (C->d_csrColInd_ != NULL) {
      CUDA_CALL(cudaFree(C->d_csrColInd_));
      CUDA_CALL(cudaFree(C->d_csrVal_));
    }
    CUDA_CALL(cudaMalloc(&C->d_csrColInd_, C_nvals*sizeof(Index)));
    CUDA_CALL(cudaMalloc(&C->d_csrVal_, C_nvals*sizeof(float)));

    if (C->h_csrColInd_ != NULL) {
      free(C->h_csrColInd_);
      free(C->h_csrVal_);
    }
    C->h_csrColInd_ = reinterpret_cast<Index*>(malloc(C_nvals*sizeof(Index)));
    C->h_csrVal_    = reinterpret_cast<T*>(malloc(C_nvals*sizeof(T)));

    C->ncapacity_ = C_nvals;
  }

  // Compute
  status = cusparseScsrgemm2(handle,
      A_nrows, B_ncols, A_ncols, &alpha,
      descr, A_nvals, A->d_csrVal_, A->d_csrRowPtr_, A->d_csrColInd_,
      descr, B_nvals, B->d_csrVal_, B->d_csrRowPtr_, B->d_csrColInd_,
      beta,
      descr, B_nvals, B->d_csrVal_, B->d_csrRowPtr_, B->d_csrColInd_,
      descr,          C->d_csrVal_, C->d_csrRowPtr_, C->d_csrColInd_,
      info,  desc->d_buffer_);

  switch (status) {
    case CUSPARSE_STATUS_SUCCESS:
      // std::cout << "SpMM successful!\n";
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

  C->need_update_ = true;  // Set flag that we need to copy data from GPU
  C->nvals_ = C_nvals;     // Update nnz count for C
  if (desc->debug())
    std::cout << C_nvals << " nonzeroes!\n";
  return GrB_SUCCESS;
}
}  // namespace backend
}  // namespace graphblas

#endif  // GRAPHBLAS_BACKEND_CUDA_SPGEMM_HPP_
