#ifndef GRAPHBLAS_ALGORITHM_CC_HPP_
#define GRAPHBLAS_ALGORITHM_CC_HPP_

#include <limits>
#include <vector>
#include <string>

#include "graphblas/algorithm/test_cc.hpp"
#include "graphblas/backend/cuda/util.hpp"

namespace graphblas {
namespace algorithm {

float cc(Vector<int>*       v,
         const Matrix<int>* A,
         int                seed,
         Descriptor*        desc) {
  Index A_nrows;
  CHECK(A->nrows(&A_nrows));

  // Colors vector (v)
  // 0: no color, 1 ... n: color
  CHECK(v->fill(0));

  // Frontier vector (f)
  Vector<int> f(A_nrows);

  // Weight vectors (w)
  Vector<int> w(A_nrows);
  CHECK(w.fill(0));

  // Neighbor max (m)
  Vector<int> m(A_nrows);

  // Set seed
  setEnv("GRB_SEED", seed);

  int iter = 1;
  int succ = 0;
  int min_color = 0;
  Index unvisited = A_nrows;
  backend::GpuTimer gpu_tight;
  float gpu_tight_time = 0.f;

  if (desc->descriptor_.timing_ > 0)
    gpu_tight.Start();
  do {
    if (desc->descriptor_.debug()) {
      std::cout << "=====GC Iteration " << iter - 1 << "=====\n";
      CHECK(v->print());
      CHECK(w.print());
      CHECK(f.print());
      CHECK(m.print());
    }
    if (desc->descriptor_.timing_ == 2) {
      gpu_tight.Stop();
      if (iter > 1) {
        std::string vxm_mode = (desc->descriptor_.lastmxv_ == GrB_PUSHONLY) ?
            "push" : "pull";
        std::cout << iter - 1 << ", " << succ << "/" << A_nrows << ", "
            << unvisited << ", " << vxm_mode << ", "
            << gpu_tight.ElapsedMillis() << "\n";
        gpu_tight_time += gpu_tight.ElapsedMillis();
      }
      unvisited -= succ;
      gpu_tight.Start();
    }

    // find max of neighbors
    vxm<int, int, int, int>(&m, GrB_NULL, GrB_NULL,
        MaximumMultipliesSemiring<int>(), &w, A, desc);
    //vxm<int, int, int, int>(&m, &w, GrB_NULL,
    //    MaximumMultipliesSemiring<int>(), &w, A, desc);

    // find all largest nodes that are uncolored
    // eWiseMult<float, float, float, float>(&f, GrB_NULL, GrB_NULL,
    //     PlusGreaterSemiring<float>(), &w, &m, desc);
    eWiseAdd<int, int, int, int>(&f, GrB_NULL, GrB_NULL,
        GreaterPlusSemiring<int>(), &w, &m, desc);

    // stop when frontier is empty
    reduce<int, int>(&succ, GrB_NULL, PlusMonoid<int>(), &f, desc);

    if (succ == 0) {
      break;
    }

    // assign new color
    assign<int, int>(v, &f, GrB_NULL, iter, GrB_ALL, A_nrows, desc);

    // get rid of colored nodes in candidate list
    assign<int, int>(&w, &f, GrB_NULL, static_cast<int>(0), GrB_ALL, A_nrows,
        desc);

    iter++;
    if (desc->descriptor_.debug())
      std::cout << "succ: " << succ << " " << static_cast<int>(succ) <<
          std::endl;
    if (iter > desc->descriptor_.max_niter_)
      break;
  } while (succ > 0);
  if (desc->descriptor_.timing_ > 0) {
    gpu_tight.Stop();
    std::string vxm_mode = (desc->descriptor_.lastmxv_ == GrB_PUSHONLY) ?
        "push" : "pull";
    if (desc->descriptor_.timing_ == 2)
      std::cout << iter << ", " << succ << "/" << A_nrows << ", "
          << unvisited << ", " << vxm_mode << ", "
          << gpu_tight.ElapsedMillis() << "\n";
    gpu_tight_time += gpu_tight.ElapsedMillis();
    return gpu_tight_time;
  }
  return 0.f;
}

template <typename a>
int ccCpu(Index             seed,
          Matrix<a>*        A,
          std::vector<int>* h_cc_cpu) {
  SimpleReferenceCc(A->matrix_.nrows_, A->matrix_.sparse_.h_csrRowPtr_,
      A->matrix_.sparse_.h_csrColInd_, h_cc_cpu, seed);
}

template <typename a>
int verifyCc(const Matrix<a>*        A,
             const std::vector<int>& h_cc_cpu,
             bool                    suppress_zero = false) {
  SimpleVerifyCc(A->matrix_.nrows_, A->matrix_.sparse_.h_csrRowPtr_,
      A->matrix_.sparse_.h_csrColInd_, h_cc_cpu, suppress_zero);
}
}  // namespace algorithm
}  // namespace graphblas

#endif  // GRAPHBLAS_ALGORITHM_CC_HPP_
