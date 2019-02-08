#ifndef GRAPHBLAS_BACKEND_CUDA_COLOR_HPP_
#define GRAPHBLAS_BACKEND_CUDA_COLOR_HPP_

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

template <typename W, typename a>
Info cusparse_color(DenseVector<W>*        w,
                    const SparseMatrix<a>* A,
                    Descriptor*            desc) {
  Index A_nrows, A_ncols, A_nvals;
  Index w_nvals;

  A_nrows = A->nrows_;
  A_ncols = A->ncols_;
  A_nvals = A->nvals_;
  w_nvals = w->nvals_;

  // Dimension compatibility check
  if (A_ncols != w_nvals) {
    std::cout << "Dim mismatch mxm" << std::endl;
    std::cout << A_ncols << " " << w_nvals << std::endl;
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

  float cusparseFraction = 1.f;
  int cusparseNcolors = 0;

  cusparseColorInfo_t info;
  cusparseCreateColorInfo(&info);

  // Analyze
  status = cusparseScsrcolor(handle, A_nrows, A_nvals, descr,
      A->d_csrVal_, A->d_csrRowPtr_, A->d_csrColInd_,
      &cusparseFraction, &cusparseNcolors, w->d_val_, NULL, info); 

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

  w->need_update_ = true;  // Set flag that we need to copy data from GPU
  return GrB_SUCCESS;
}

}  // namespace backend
}  // namespace graphblas

#endif  // GRAPHBLAS_BACKEND_CUDA_COLOR_HPP_
