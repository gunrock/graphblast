#ifndef GRB_ALGORITHM_LGCSIMPLE_HPP
#define GRB_ALGORITHM_LGCSIMPLE_HPP

#include "graphblas/algorithm/testLgc.hpp"
#include "graphblas/backend/apspie/util.hpp" // GpuTimer

namespace graphblas
{
namespace algorithm
{
  // Use float for now for both p and A
  float lgcSimple( Vector<float>*       p,      // PageRank result
                   const Matrix<float>* A,      // graph
                   Index                s,      // source vertex
                   double               alpha,  // teleport constant in (0,1]
                   double               eps,    // tolerance
                   Descriptor*          desc )
  {
    Index A_nrows;
    CHECK( A->nrows( &A_nrows ) );

    // degrees: compute the degree of each node
    Vector<float> degrees(A_nrows);
    reduce<float, float, float>(&degrees, GrB_NULL, GrB_NULL, 
        PlusMonoid<float>(), A, desc);

    // pagerank (p): initialized to 0
    CHECK( p->fill(0.f) );

    // residual (r): initialized to 0 except source to 1
    Vector<float> r(A_nrows);
    std::vector<Index> indices(1, s);
    std::vector<float> values(1, 1.f);

    // residual2 (r2)
    Vector<float> r2(A_nrows);
    CHECK( r2.fill(0.f) );

    CHECK( r.build(&indices, &values, 1, GrB_NULL) );

    // degrees_eps (d x eps): precompute degree of each node times eps
    Vector<float> eps_vector(A_nrows);
    Vector<float> degrees_eps(A_nrows);
    CHECK( eps_vector.fill(eps) );
    eWiseMult<float, float, float, float>( &degrees_eps, GrB_NULL, GrB_NULL,
        PlusMultipliesSemiring<float>(), &degrees, &eps_vector, desc );

    // frontier (f): portion of r(v) >= degrees(v) x eps
    // (use float for now)
    Vector<float> f(A_nrows);
    CHECK( f.build(&indices, &values, 1, GrB_NULL) );

    // alpha: TODO(@ctcyang): introduce vector-constant eWiseMult
    Vector<float> alpha_vector(A_nrows);
    CHECK( alpha_vector.fill(alpha) );
    Vector<float> alpha_vector2(A_nrows);
    CHECK( alpha_vector2.fill((1.-alpha)/2.) );

    Index nvals;
    CHECK( f.nvals(&nvals) );

    Index iter = 1;
    float succ;
    do
    {
      // p = p + alpha * r .* f
      CHECK( desc->toggle(GrB_MASK) );
      eWiseMult<float, float, float, float>(&r2, &f, GrB_NULL, 
          PlusMultipliesSemiring<float>(), &r, &alpha_vector, desc);
      CHECK( desc->toggle(GrB_MASK) );
      eWiseAdd<float, float, float, float>(p, GrB_NULL, GrB_NULL, 
          PlusMultipliesSemiring<float>(), p, &r2, desc);

      // r = (1 - alpha)/2 * r
      eWiseMult<float, float, float, float>(&r, &f, GrB_NULL,
          PlusMultipliesSemiring<float>(), &r, &alpha_vector2, desc);

      // r2 = r/d .* f
      CHECK( desc->toggle(GrB_MASK) );
      eWiseMult<float, float, float, float>(&r2, &f, GrB_NULL, 
          PlusDividesSemiring<float>(), &r, &degrees, desc);
      CHECK( desc->toggle(GrB_MASK) );

      // r = r + A^T * r2
      mxv<float, float, float, float>(&r, GrB_NULL, PlusMonoid<float>(), 
          PlusMultipliesSemiring<float>(), A, &r2, desc);

      // f = {v | r(v) >= d*eps}
      eWiseMult<float, float, float, float>(&f, GrB_NULL, GrB_NULL, 
          PlusGreaterSemiring<float>(), &r, &degrees_eps, desc);

      // Update frontier size
      reduce<float, float>(&succ, GrB_NULL, PlusMonoid<float>(), &f, desc);

      iter++;
    } while (succ > 0 && iter <= desc->descriptor_.max_niter_);
    return 0.f;
  }

  template <typename T, typename a>
  void lgcCpu( T*               h_lgc_cpu,
               const Matrix<a>* A,
               Index            s,
               double           alpha,
               double           eps,
               int              max_niter,
               bool             transpose=false )
  {
    if( transpose )
      SimpleReferenceLgc<T>( A->matrix_.nrows_, A->matrix_.sparse_.h_cscColPtr_,
          A->matrix_.sparse_.h_cscRowInd_, A->matrix_.sparse_.h_cscVal_, 
          h_lgc_cpu, s, alpha, eps, max_niter );
    else
      SimpleReferenceLgc<T>( A->matrix_.nrows_, A->matrix_.sparse_.h_csrRowPtr_,
          A->matrix_.sparse_.h_csrColInd_, A->matrix_.sparse_.h_csrVal_,
          h_lgc_cpu, s, alpha, eps, max_niter );
  }

}  // algorithm
}  // graphblas

#endif  // GRB_ALGORITHM_LGC_HPP
