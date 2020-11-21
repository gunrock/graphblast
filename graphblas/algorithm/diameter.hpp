#ifndef GRAPHBLAS_ALGORITHM_DIAMETER_HPP_
#define GRAPHBLAS_ALGORITHM_DIAMETER_HPP_

#include <vector>
#include <utility>

#include "graphblas/algorithm/test_bfs.hpp"
#include "graphblas/backend/cuda/util.hpp"

namespace graphblas {
namespace algorithm {

// Use float for now for both v and A
std::pair<int, int> diameter(Vector<float>*       v,
                             const Matrix<float>* A,
                             Index                s_start,
                             Index                s_end,
                             Descriptor*          desc) {
  Index A_nrows;
  A->nrows(&A_nrows);

  // Frontier vectors (use float for now)
  Vector<float> q1(A_nrows);
  Vector<float> q2(A_nrows);

  int diameter_max = 0;
  int diameter_ind = -1;

  for (int s = s_start; s < s_end; ++s) {
    v->fill(0.f);

    std::vector<Index> indices(1, s);
    std::vector<float>  values(1, 1.f);
    q1.build(&indices, &values, 1, GrB_NULL);

    int iter = 1;
    float succ = 0.f;
    do {
      assign<float, float, float, Index>(v, &q1, GrB_NULL,
          static_cast<float>(iter), GrB_ALL, A_nrows, desc);
      desc->toggle(GrB_MASK);
      vxm<float, float, float, float>(&q2, v, GrB_NULL,
          LogicalOrAndSemiring<float>(), &q1, A, desc);
      desc->toggle(GrB_MASK);
      q2.swap(&q1);
      reduce<float, float>(&succ, GrB_NULL, PlusMonoid<float>(), &q1, desc);
      iter++;
    } while (succ > 0);
    diameter_max = std::max(diameter_max, iter - 2);
    diameter_ind = (iter - 2 == diameter_max) ? s : diameter_ind;
  }
  return std::make_pair(diameter_max, diameter_ind);
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

#endif  // GRAPHBLAS_ALGORITHM_DIAMETER_HPP_
