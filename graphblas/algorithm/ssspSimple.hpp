#ifndef GRB_ALGORITHM_SSSPSIMPLE_HPP
#define GRB_ALGORITHM_SSSPSIMPLE_HPP

#include <limits>
#include "graphblas/algorithm/testSssp.hpp"

namespace graphblas
{
namespace algorithm
{
  // Use float for now for both v and A
  graphblas::Info ssspSimple( Vector<float>*       v,
                              const Matrix<float>* A,
                              Index                s,
                              Descriptor*          desc )
  {
    // Get number of vertices
    graphblas::Index A_nrows;
    A->nrows(&A_nrows);

    // Distance vector (v)
    std::vector<graphblas::Index> indices(1, s);
    std::vector<float>  values(1, 0.f);
    v->build(&indices, &values, 1, GrB_NULL);

    // Buffer vector (w)
    graphblas::Vector<float> w(A_nrows);

    // Semiring zero vector (zero)
    graphblas::Vector<float> zero(A_nrows);
    zero.fill(std::numeric_limits<float>::max());

    // Initialize loop variables
    graphblas::Index iter = 1;
    float succ_last = 0.f;
    float succ = 1.f;

    do
    {
      succ_last = succ;
      
      // v = v + v * A^T (do relaxation on distance vector v)
      graphblas::vxm<float, float, float, float>(&w, GrB_NULL, GrB_NULL,
          graphblas::MinimumPlusSemiring<float>(), v, A, desc);
      graphblas::eWiseAdd<float, float, float, float>(v, GrB_NULL, GrB_NULL,
          graphblas::MinimumPlusSemiring<float>(), v, &w, desc);

      // w = v < FLT_MAX (get all reachable vertices)
      graphblas::eWiseMult<float, float, float, float>(&w, GrB_NULL, GrB_NULL,
          graphblas::PlusLessSemiring<float>(), v, &zero, desc);

      // succ = reduce(w) (do reduction on all reachable distances)
      graphblas::reduce<float, float>(&succ, GrB_NULL, 
          graphblas::PlusMonoid<float>(), &w, desc);
      iter++;

      // Loop until total reachable distance has converged
    } while (succ_last != succ);

    return GrB_SUCCESS;
  }

  template <typename T, typename a>
  int ssspCpu( Index        source,
               Matrix<a>*   A,
               T*           h_sssp_cpu,
               Index        depth,
               bool         transpose=false )
  {
    int max_depth;

    if( transpose )
      max_depth = SimpleReferenceSssp<T>( A->matrix_.nrows_, 
          A->matrix_.sparse_.h_cscColPtr_, A->matrix_.sparse_.h_cscRowInd_, 
          A->matrix_.sparse_.h_cscVal_, h_sssp_cpu, source, depth);
    else
      max_depth = SimpleReferenceSssp<T>( A->matrix_.nrows_, 
          A->matrix_.sparse_.h_csrRowPtr_, A->matrix_.sparse_.h_csrColInd_, 
          A->matrix_.sparse_.h_csrVal_, h_sssp_cpu, source, depth);

    return max_depth; 
  }

}  // algorithm
}  // graphblas

#endif  // GRB_ALGORITHM_SSSP_HPP
