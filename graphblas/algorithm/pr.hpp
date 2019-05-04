#ifndef GRAPHBLAS_ALGORITHM_PR_HPP_
#define GRAPHBLAS_ALGORITHM_PR_HPP_

#include <limits>
#include <vector>
#include <string>

#include "graphblas/algorithm/test_pr.hpp"
#include "graphblas/backend/cuda/util.hpp"

namespace graphblas {
namespace algorithm {

// Use float for now for both v and A
float pr(Vector<float>*       p,
         const Matrix<float>* A,
         float                alpha, // teleportation constant
         float                eps,   // threshold
         Descriptor*          desc) {
  // Get number of vertices
  Index A_nrows;
  CHECK(A->nrows(&A_nrows));

  // Compute outdegrees
  Vector<float> outdegrees(A_nrows);
  reduce<float, float, float>(&outdegrees, GrB_NULL, GrB_NULL,
      PlusMonoid<float>(), A, desc);

  // A = alpha*A/outdegrees (broadcast variant)
  // TODO(@ctcyang): add inplace variant if have trouble storing graph in memory
  Matrix<float> A_temp(A_nrows, A_nrows);
  eWiseMult<float, float, float, float>(&A_temp, GrB_NULL, GrB_NULL, 
      PlusMultipliesSemiring<float>(), A, alpha, desc);
  eWiseMult<float, float, float, float>(A, GrB_NULL, GrB_NULL,
      PlusDividesSemiring<float>(), &A_temp, outdegrees, desc);

  /*// Diagonalize outdegrees
  Matrix<float> diag_outdegrees(A_nrows, A_nrows);
  diag<float, float>(&diag_outdegrees, &outdegrees, desc);

  // A = alpha*A*diag(outdegrees)
  Matrix<float> A_temp(A_nrows, A_nrows);
  scale<float, float, float>(&A_temp, MultipliesMonoid<float>(), A, alpha,
      desc);*/

  // Pagerank vector (p)
  CHECK(p->clear());
  CHECK(p->fill(1.f/A_nrows));

  // Previous pagerank vector (p_prev)
  Vector<float> p_prev(A_nrows);

  // Temporary pagerank (p_temp)
  Vector<float> p_swap(A_nrows);

  // Residual vector (r)
  Vector<float> r(A_nrows);
  r.fill(1.f);

  // Temporary residual (r_temp)
  Vector<float> r_temp(A_nrows);

  /*// (1-alpha)*1 (one_minus_alpha)
  Vector<float> one_minus_alpha(A_nrows);
  one_minus_alpha.fill(1.f-alpha);*/

  int iter;
  float error_last = 0.f;
  float error = 1.f;
  Index unvisited = A_nrows;

  backend::GpuTimer gpu_tight;
  float gpu_tight_time = 0.f;
  if (desc->descriptor_.timing_ > 0)
    gpu_tight.Start();
  for (iter = 1; error < eps && iter <= desc->descriptor_.max_niter_;
      ++iter) {
    if (desc->descriptor_.debug())
      std::cout << "=====PR Iteration " << iter - 1 << "=====\n";
    if (desc->descriptor_.timing_ == 2) {
      gpu_tight.Stop();
      if (iter > 1) {
        std::string vxm_mode = (desc->descriptor_.lastmxv_ == GrB_PUSHONLY) ?
            "push" : "pull";
        std::cout << iter - 1 << ", " << error << "/" << A_nrows << ", "
            << unvisited << ", " << vxm_mode << ", "
            << gpu_tight.ElapsedMillis() << "\n";
        gpu_tight_time += gpu_tight.ElapsedMillis();
      }
      unvisited -= static_cast<int>(error);
      gpu_tight.Start();
    }
    error_last = error;
    p_prev = p;

    // p = A*p + (1-alpha)*1
    vxm<float, float, float, float>(&p_swap, GrB_NULL, GrB_NULL,
        PlusMultipliesSemiring<float>(), &p_prev, A, desc);
    eWiseAdd<float, float, float, float>(p, GrB_NULL, GrB_NULL,
        PlusMultipliesSemiring<float>(), &p_swap, 1.f-alpha, desc);

    // error = l2loss(p, p_prev)
    eWiseMult<float, float, float, float>(&r, GrB_NULL, GrB_NULL,
        PlusMinusSemiring<float>(), p, &p_prev, desc);
    eWiseAdd<float, float, float, float>(&r_temp, GrB_NULL, GrB_NULL,
        MultipliesMultipliesSemiring<float>(), &r, &r, desc);
    reduce<float, float>(&error, GrB_NULL, PlusMonoid<float>(), &r_temp, desc);
    error = sqrt(error);

    if (desc->descriptor_.debug())
      std::cout << "error: " << error_last << std::endl;
  }
  if (desc->descriptor_.timing_ > 0) {
    gpu_tight.Stop();
    std::string vxm_mode = (desc->descriptor_.lastmxv_ == GrB_PUSHONLY) ?
        "push" : "pull";
    std::cout << iter - 1 << ", " << error << "/" << A_nrows << ", "
        << unvisited << ", " << vxm_mode << ", "
        << gpu_tight.ElapsedMillis() << "\n";
    gpu_tight_time += gpu_tight.ElapsedMillis();
    return gpu_tight_time;
  }
  return 0.f;
}

template <typename T, typename a>
int prCpu(T*         h_pr_cpu,
          Matrix<a>* A,
          float      alpha,               // teleportation constant
          float      eps,                 // threshold
          int        max_niter,
          bool       transpose = false) {
  int max_depth;

  if (transpose)
    max_depth = SimpleReferencePr<T>(A->matrix_.nrows_,
        A->matrix_.sparse_.h_cscColPtr_, A->matrix_.sparse_.h_cscRowInd_,
        A->matrix_.sparse_.h_cscVal_, h_pr_cpu, alpha, eps, max_niter);
  else
    max_depth = SimpleReferencePr<T>(A->matrix_.nrows_,
        A->matrix_.sparse_.h_csrRowPtr_, A->matrix_.sparse_.h_csrColInd_,
        A->matrix_.sparse_.h_csrVal_, h_pr_cpu, alpha, eps, max_niter);

  return max_depth;
}
}  // namespace algorithm
}  // namespace graphblas

#endif  // GRAPHBLAS_ALGORITHM_PR_HPP_
