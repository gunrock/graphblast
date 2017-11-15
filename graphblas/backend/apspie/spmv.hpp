#ifndef GRB_BACKEND_APSPIE_SPMV_HPP
#define GRB_BACKEND_APSPIE_SPMV_HPP

#include <iostream>

#include <cuda.h>
#include <cusparse.h>

#include <moderngpu.cuh>

#include "graphblas/backend/apspie/Descriptor.hpp"
#include "graphblas/backend/apspie/SparseMatrix.hpp"
#include "graphblas/backend/apspie/DenseMatrix.hpp"

namespace graphblas
{
namespace backend
{

  template <int variant, typename W, typename a, typename U, typename M,
            typename BinaryOpT,      typename SemiringT>
  Info spmv( DenseVector<W>*        w,
             const SparseVector<M>* mask,
             const BinaryOpT*       accum,
             const SemiringT*       op,
             const SparseMatrix<a>* A,
             const SparseVector<U>* u,
             const Descriptor*      desc )
  {
    std::cout << "Error: Feature not implemented yet!\n";
    return GrB_SUCCESS;
  }

  template <int variant, typename W, typename a, typename U, typename M,
            typename BinaryOpT,      typename SemiringT>
  Info spmv( DenseVector<W>*        w,
             const SparseVector<M>* mask,
             const BinaryOpT*       accum,
             const SemiringT*       op,
             const SparseMatrix<a>* A,
             const DenseVector<U>*  u,
             const Descriptor*      desc )
  {
    std::cout << "Error: Feature not implemented yet!\n";
    return GrB_SUCCESS;
  }

}  // backend
}  // graphblas

#endif  // GRB_BACKEND_APSPIE_SPMM_HPP
