#ifndef GRAPHBLAS_ALGORITHM_BFS_SIMPLE_HPP_
#define GRAPHBLAS_ALGORITHM_BFS_SIMPLE_HPP_

#include <vector>

#include "graphblas/algorithm/test_bfs.hpp"
#include "graphblas/backend/cuda/util.hpp"

namespace graphblas {
namespace algorithm {

// Use float for now for both v and A
float bfsSimple(Vector<float>*       v,
                const Matrix<float>* A,
                Index                s,
                Descriptor*          desc) {
  Index A_nrows;
  CHECK(A->nrows(&A_nrows));

  // Visited vector (use float for now)
  CHECK(v->fill(0.f));

  // Frontier vectors (use float for now)
  Vector<float> q1(A_nrows);
  Vector<float> q2(A_nrows);

  std::vector<Index> indices(1, s);
  std::vector<float>  values(1, 1.f);
  CHECK(q1.build(&indices, &values, 1, GrB_NULL));

  float iter = 1;
  float succ = 0.f;
  do {
    assign<float, float>(v, &q1, GrB_NULL, iter, GrB_ALL, A_nrows, desc);
    CHECK(desc->toggle(GrB_MASK));
    vxm<float, float, float, float>(&q2, v, GrB_NULL,
        LogicalOrAndSemiring<float>(), &q1, A, desc);
    CHECK(desc->toggle(GrB_MASK));
    CHECK(q2.swap(&q1));
    reduce<float, float>(&succ, GrB_NULL, PlusMonoid<float>(), &q1, desc);

    iter++;
  } while (succ > 0);
  return 0.f;
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

#endif  // GRAPHBLAS_ALGORITHM_BFS_SIMPLE_HPP_
