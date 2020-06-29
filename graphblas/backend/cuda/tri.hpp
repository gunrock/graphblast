#ifndef GRAPHBLAS_BACKEND_CUDA_TRI_HPP_
#define GRAPHBLAS_BACKEND_CUDA_TRI_HPP_

#include <iostream>

namespace graphblas {
namespace backend {

template <typename a, typename c>
Info trilSparse(SparseMatrix<c>* C,
                SparseMatrix<a>* A,
                Descriptor*      desc) {
  // Get descriptor parameters for BACKEND
  Desc_value backend;
  CHECK(desc->get(GrB_BACKEND, &backend));

  if (desc->debug()) {
    std::cout << "Executing trilSparse\n";
  }

  if (backend == GrB_SEQUENTIAL) {
    CHECK(A->gpuToCpu());
    Index remove = 0;
    for (Index row = 0; row < A->nrows_; ++row) {
      Index edge_start = A->h_csrRowPtr_[row];
      Index edge_end = A->h_csrRowPtr_[row+1];

      // csrRowPtr_ update must be done after row loads edge_start
      A->h_csrRowPtr_[row] -= remove;

      for (Index edge = edge_start; edge < edge_end; ++edge) {
        Index col = A->h_csrColInd_[edge];
        if (row < col) {
          remove++;
        } else {
          A->h_csrColInd_[edge-remove] = col;
          A->h_csrVal_[edge-remove] = A->h_csrVal_[edge];
        }
      }
    }
    A->h_csrRowPtr_[A->nrows_] -= remove;
    A->nvals_ -= remove;

    CHECK(C->syncCpu());
    CHECK(C->cpuToGpu());
  } else {
    std::cout << "trilSparse GPU\n";
    std::cout << "Error: Feature not implemented yet!\n";
  }
  return GrB_SUCCESS;
}
}  // namespace backend
}  // namespace graphblas

#endif  // GRAPHBLAS_BACKEND_CUDA_TRI_HPP_
