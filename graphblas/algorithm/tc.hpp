#ifndef GRAPHBLAS_ALGORITHM_TC_HPP_
#define GRAPHBLAS_ALGORITHM_TC_HPP_

#include <limits>
#include <vector>
#include <string>

#include "graphblas/algorithm/test_tc.hpp"
#include "graphblas/backend/cuda/util.hpp"

namespace graphblas {
namespace algorithm {

// Use int for ntris and A
float tc(int*               ntris,
         const Matrix<int>* A,     // lower triangular matrix
         Matrix<int>*       B,     // buffer matrix
         Descriptor*        desc) {
  // Get number of vertices
  Index A_nrows;
  CHECK(A->nrows(&A_nrows));

  // Set second input to be transposed
  CHECK(desc->toggle(graphblas::GrB_INP1));

  int iter = 1;
  float error_last = 0.f;
  float error = 1.f;

  backend::GpuTimer gpu_tight;
  float gpu_tight_time = 0.f;
  gpu_tight.Start();

  if (desc->descriptor_.debug())
    std::cout << "=====TC Iteration " << iter - 1 << "=====\n";

  // B = A * A^T
  mxm<int, int, int, int>(B, A, GrB_NULL, PlusMultipliesSemiring<int>(),
      A, A, desc);

  // ntris = reduce(B)
  reduce<int, int>(ntris, GrB_NULL, PlusMonoid<int>(), B, desc);

  if (desc->descriptor_.debug())
    std::cout << "ntris: " << *ntris << std::endl;
  gpu_tight.Stop();
  std::string vxm_mode = (desc->descriptor_.lastmxv_ == GrB_PUSHONLY) ?
      "push" : "pull";
  if (desc->descriptor_.timing_ > 0)
    std::cout << iter - 1 << ", " << error << "/" << A_nrows << ", "
        << vxm_mode << ", " << gpu_tight.ElapsedMillis() << "\n";
  gpu_tight_time += gpu_tight.ElapsedMillis();
  return gpu_tight_time;
}

template <typename T, typename a>
int tcCpu(T*         ntris,
          Matrix<a>* A,
          bool       transpose = false) {
  if (transpose)
    return SimpleReferenceTc<T>(A->matrix_.nrows_,
        A->matrix_.sparse_.h_cscColPtr_, A->matrix_.sparse_.h_cscRowInd_,
        ntris);
  return SimpleReferenceTc<T>(A->matrix_.nrows_,
      A->matrix_.sparse_.h_csrRowPtr_, A->matrix_.sparse_.h_csrColInd_,
      ntris);
}
}  // namespace algorithm
}  // namespace graphblas

#endif  // GRAPHBLAS_ALGORITHM_TC_HPP_
