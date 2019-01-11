#ifndef GRAPHBLAS_ALGORITHM_SSSP_HPP_
#define GRAPHBLAS_ALGORITHM_SSSP_HPP_

#include <limits>
#include <vector>
#include <string>

#include "graphblas/algorithm/test_sssp.hpp"
#include "graphblas/backend/cuda/util.hpp"

namespace graphblas {
namespace algorithm {

// Use float for now for both v and A
float sssp(Vector<float>*       v,
           const Matrix<float>* A,
           Index                s,
           Descriptor*          desc) {
  Index A_nrows;
  CHECK(A->nrows(&A_nrows));
  CHECK(v->clear());

  graphblas::Vector<float> w(A_nrows);
  graphblas::Vector<float> zero(A_nrows);
  zero.fill(std::numeric_limits<float>::max());

  Desc_value desc_value;
  CHECK(desc->get(GrB_MXVMODE, &desc_value));

  // Visited vector (use float for now)
  if (desc_value == GrB_PULLONLY) {
    CHECK(v->fill(std::numeric_limits<float>::max()));
    CHECK(v->setElement(0.f, s));
  } else {
    std::vector<Index> indices(1, s);
    std::vector<float>  values(1, 0.f);
    CHECK(v->build(&indices, &values, 1, GrB_NULL));
  }

  Index iter = 1;
  float succ_last = 0.f;
  float succ = 1.f;
  Index unvisited = A_nrows;

  backend::GpuTimer gpu_tight;
  float gpu_tight_time = 0.f;
  if (desc->descriptor_.timing_ > 0)
    gpu_tight.Start();
  do {
    if (desc->descriptor_.debug())
      std::cout << "=====SSSP Iteration " << iter - 1 << "=====\n";
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
      unvisited -= static_cast<int>(succ);
      gpu_tight.Start();
    }
    succ_last = succ;

    // TODO(@ctcyang): add inplace + accumulate version
    vxm<float, float, float, float>(&w, GrB_NULL, GrB_NULL,
        MinimumPlusSemiring<float>(), v, A, desc);
    eWiseAdd<float, float, float, float>(v, GrB_NULL, GrB_NULL,
        MinimumPlusSemiring<float>(), v, &w, desc);

    eWiseAdd<float, float, float, float>(&w, GrB_NULL, GrB_NULL,
        LessPlusSemiring<float>(), v, &zero, desc);
    reduce<float, float>(&succ, GrB_NULL, PlusMonoid<float>(), &w, desc);
    iter++;

    if (desc->descriptor_.debug())
      std::cout << "succ: " << succ_last << std::endl;
    if (iter > desc->descriptor_.max_niter_)
      break;
  } while (succ_last != succ);
  if (desc->descriptor_.timing_ > 0) {
    gpu_tight.Stop();
    std::string vxm_mode = (desc->descriptor_.lastmxv_ == GrB_PUSHONLY) ?
        "push" : "pull";
    std::cout << iter - 1 << ", " << succ << "/" << A_nrows << ", "
        << unvisited << ", " << vxm_mode << ", "
        << gpu_tight.ElapsedMillis() << "\n";
    gpu_tight_time += gpu_tight.ElapsedMillis();
    return gpu_tight_time;
  }
  return 0.f;
}

template <typename T, typename a>
int ssspCpu(Index        source,
            Matrix<a>*   A,
            T*           h_sssp_cpu,
            Index        depth,
            bool         transpose = false) {
  int max_depth;

  if (transpose)
    max_depth = SimpleReferenceSssp<T>(A->matrix_.nrows_,
        A->matrix_.sparse_.h_cscColPtr_, A->matrix_.sparse_.h_cscRowInd_,
        A->matrix_.sparse_.h_cscVal_, h_sssp_cpu, source, depth);
  else
    max_depth = SimpleReferenceSssp<T>(A->matrix_.nrows_,
        A->matrix_.sparse_.h_csrRowPtr_, A->matrix_.sparse_.h_csrColInd_,
        A->matrix_.sparse_.h_csrVal_, h_sssp_cpu, source, depth);

  return max_depth;
}
}  // namespace algorithm
}  // namespace graphblas

#endif  // GRAPHBLAS_ALGORITHM_SSSP_HPP_
