#ifndef GRB_BACKEND_APSPIE_GEMV_HPP
#define GRB_BACKEND_APSPIE_GEMV_HPP

#include <iostream>

#include <cuda.h>
#include <cusparse.h>

#include <moderngpu.cuh>

#include "graphblas/backend/apspie/SparseMatrix.hpp"
#include "graphblas/backend/apspie/DenseMatrix.hpp"

namespace graphblas
{
namespace backend
{

  template <typename W, typename a, typename U, typename M,
            typename BinaryOpT,      typename SemiringT>
  Info gemv( DenseVector<W>*        w,
             const SparseVector<M>* mask,
             const BinaryOpT*       accum,
             const SemiringT*       op,
             const DenseMatrix<a>*  A,
             const DenseVector<U>*  u,
             Descriptor*            desc )
  {
    // Set transpose flag to true for A
    std::cout << "Error: Feature not implemented yet!\n";
    return GrB_SUCCESS;
  }

  template <typename W, typename a, typename U, typename M,
            typename BinaryOpT,      typename SemiringT>
  Info gemv( DenseVector<W>*        w,
             const SparseVector<M>* mask,
             const BinaryOpT*       accum,
             const SemiringT*       op,
             const DenseMatrix<a>*  A,
             const SparseVector<U>* u,
             Descriptor*            desc )
  {
    // Set transpose flag to true for A
    std::cout << "Error: Feature not implemented yet!\n";
    return GrB_SUCCESS;
  }
}  // backend
}  // graphblas

#endif  // GrB_BACKEND_APSPIE_GEMV_HPP
