#ifndef GRAPHBLAS_ALGORITHM_BFS_HPP_
#define GRAPHBLAS_ALGORITHM_BFS_HPP_

#include <string>
#include <vector>

#include "graphblas/algorithm/test_bfs.hpp"
#include "graphblas/backend/cuda/util.hpp"

namespace graphblas {
namespace algorithm {

// Use float for now for both v and A
float bfs(Vector<float>*       v,
          const Matrix<float>* A,
          Index                s,
          Descriptor*          desc) {
  Index A_nrows;
  CHECK(A->nrows(&A_nrows));

  // Visited vector (use float for now)
  CHECK(v->fill(0.f));

  // Frontier vectors (use float for now)
  Vector<float> f1(A_nrows);
  Vector<float> f2(A_nrows);

  Desc_value desc_value;
  CHECK(desc->get(GrB_MXVMODE, &desc_value));
  if (desc_value == GrB_PULLONLY) {
    CHECK(f1.fill(0.f));
    CHECK(f1.setElement(1.f, s));
  } else {
    std::vector<Index> indices(1, s);
    std::vector<float>  values(1, 1.f);
    CHECK(f1.build(&indices, &values, 1, GrB_NULL));
  }

  float iter;
  float succ = 0.f;
  Index unvisited = A_nrows;
  backend::GpuTimer gpu_tight;
  float gpu_tight_time = 0.f;
  gpu_tight.Start();

  for (iter = 1; iter <= desc->descriptor_.max_niter_; ++iter) {
    if (desc->descriptor_.debug()) {
      std::cout << "=====BFS Iteration " << iter - 1 << "=====\n";
      v->print();
      f1.print();
    }
    gpu_tight.Stop();
    if (iter > 1) {
      std::string vxm_mode = (desc->descriptor_.lastmxv_ == GrB_PUSHONLY) ?
          "push" : "pull";
      if (desc->descriptor_.timing_ == 1)
        std::cout << iter - 1 << ", " << succ << "/" << A_nrows << ", "
            << unvisited << ", " << vxm_mode << ", "
            << gpu_tight.ElapsedMillis() << "\n";
      gpu_tight_time += gpu_tight.ElapsedMillis();
    }
    unvisited -= static_cast<int>(succ);
    gpu_tight.Start();

    assign<float, float, float, Index>(v, &f1, GrB_NULL, iter, GrB_ALL, A_nrows,
        desc);
    CHECK(desc->toggle(GrB_MASK));
    vxm<float, float, float, float>(&f2, v, GrB_NULL,
        LogicalOrAndSemiring<float>(), &f1, A, desc);
    CHECK(desc->toggle(GrB_MASK));

    CHECK(f2.swap(&f1));
    reduce<float, float>(&succ, GrB_NULL, PlusMonoid<float>(), &f1, desc);

    if (desc->descriptor_.debug())
      std::cout << "succ: " << succ << std::endl;
    if (succ == 0)
      break;
  }
  gpu_tight.Stop();
  std::string vxm_mode = (desc->descriptor_.lastmxv_ == GrB_PUSHONLY) ?
      "push" : "pull";
  if (desc->descriptor_.timing_ == 1)
    std::cout << iter << ", " << succ << "/" << A_nrows << ", "
        << unvisited << ", " << vxm_mode << ", "
        << gpu_tight.ElapsedMillis() << "\n";
  gpu_tight_time += gpu_tight.ElapsedMillis();
  return gpu_tight_time;
}

template <typename T, typename a>
int bfsCpu(Index        source,
           Matrix<a>*   A,
           T*           h_bfs_cpu,
           Index        depth,
           bool         transpose = false) {
  Index* reference_check_preds = NULL;
  int max_depth;

  if (transpose)
    max_depth = SimpleReferenceBfs<T>(A->matrix_.nrows_,
        A->matrix_.sparse_.h_cscColPtr_, A->matrix_.sparse_.h_cscRowInd_,
        h_bfs_cpu, reference_check_preds, source, depth);
  else
    max_depth = SimpleReferenceBfs<T>(A->matrix_.nrows_,
        A->matrix_.sparse_.h_csrRowPtr_, A->matrix_.sparse_.h_csrColInd_,
        h_bfs_cpu, reference_check_preds, source, depth);

  return max_depth;
}
}  // namespace algorithm
}  // namespace graphblas

#endif  // GRAPHBLAS_ALGORITHM_BFS_HPP_
