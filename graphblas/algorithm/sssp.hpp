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

  // Visited vector (use float for now)
  CHECK(v->fill(std::numeric_limits<float>::max()));
  CHECK(v->setElement(0.f, s));

  // Frontier vectors (use float for now)
  Vector<float> f1(A_nrows);
  Vector<float> f2(A_nrows);

  Desc_value desc_value;
  CHECK(desc->get(GrB_MXVMODE, &desc_value));

  // Visited vector (use float for now)
  if (desc_value == GrB_PULLONLY) {
    CHECK(f1.fill(std::numeric_limits<float>::max()));
    CHECK(f1.setElement(0.f, s));
  } else {
    std::vector<Index> indices(1, s);
    std::vector<float>  values(1, 0.f);
    CHECK(f1.build(&indices, &values, 1, GrB_NULL));
  }

  // Mask vector
  Vector<float> m(A_nrows);

  Index iter;
  Index f1_nvals = 1;
  float succ = 1.f;
  backend::GpuTimer gpu_tight;
  float gpu_tight_time = 0.f;
  gpu_tight.Start();

  for (iter = 1; iter <= desc->descriptor_.max_niter_; ++iter) {
    if (desc->descriptor_.debug())
      std::cout << "=====SSSP Iteration " << iter - 1 << "=====\n";
    gpu_tight.Stop();
    if (iter > 1) {
      std::string vxm_mode = (desc->descriptor_.lastmxv_ == GrB_PUSHONLY) ?
        "push" : "pull";
      if (desc->descriptor_.timing_ == 1)
        std::cout << iter - 1 << ", " << f1_nvals << "/" << A_nrows << ", "
            << vxm_mode << ", " << gpu_tight.ElapsedMillis() << "\n";
      gpu_tight_time += gpu_tight.ElapsedMillis();
    }
    gpu_tight.Start();

    vxm<float, float, float, float>(&f2, GrB_NULL, GrB_NULL,
        MinimumPlusSemiring<float>(), &f1, A, desc);

    //eWiseMult<float, float, float, float>(&m, GrB_NULL, GrB_NULL,
    //    PlusLessSemiring<float>(), &f2, v, desc);
    eWiseAdd<float, float, float, float>(&m, GrB_NULL, GrB_NULL,
        CustomLessPlusSemiring<float>(), &f2, v, desc);

    eWiseAdd<float, float, float, float>(v, GrB_NULL, GrB_NULL,
        MinimumPlusSemiring<float>(), v, &f2, desc);

    // Similar to BFS, except we need to filter out the unproductive vertices
    // here rather than as part of masked vxm
    CHECK(desc->toggle(GrB_MASK));
    assign<float, float>(&f2, &m, GrB_NULL, std::numeric_limits<float>::max(),
        GrB_ALL, A_nrows, desc);
    CHECK(desc->toggle(GrB_MASK));

    CHECK(f2.swap(&f1));

    CHECK(f1.nvals(&f1_nvals));
    reduce<float, float>(&succ, GrB_NULL, PlusMonoid<float>(), &m, desc);

    if (desc->descriptor_.debug())
      std::cout << succ << std::endl;
    if (f1_nvals == 0 || succ == 0)
      break;
  }
  gpu_tight.Stop();
  std::string vxm_mode = (desc->descriptor_.lastmxv_ == GrB_PUSHONLY) ?
      "push" : "pull";
  if (desc->descriptor_.timing_ == 1)
    std::cout << iter << ", " << f1_nvals << "/" << A_nrows << ", "
        << vxm_mode << ", " << gpu_tight.ElapsedMillis() << "\n";
  gpu_tight_time += gpu_tight.ElapsedMillis();
  return gpu_tight_time;
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
