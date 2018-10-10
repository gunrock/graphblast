#ifndef GRB_BACKEND_APSPIE_EWISEMULT_HPP
#define GRB_BACKEND_APSPIE_EWISEMULT_HPP

#include <iostream>

#include <cuda.h>
#include <cusparse.h>

#include <moderngpu.cuh>

namespace graphblas
{
namespace backend
{

  /*
   * \brief 4 vector variants
   */

  // Sparse x sparse vector
  template <typename W, typename U, typename V, typename M,
            typename BinaryOpT,      typename SemiringT>
  Info eWiseMult( SparseVector<W>*       w,
                  const Vector<M>*       mask,
                  BinaryOpT              accum,
                  SemiringT              op,
                  const SparseVector<U>* u,
                  const SparseVector<V>* v,
                  Descriptor*            desc )
  {
    std::cout << "Error: Feature not implemented yet!\n";
    return GrB_SUCCESS;
  }

  // Dense x dense vector
  template <typename W, typename U, typename V, typename M,
            typename BinaryOpT,      typename SemiringT>
  Info eWiseMult( DenseVector<W>*       w,
                  const Vector<M>*      mask,
                  BinaryOpT             accum,
                  SemiringT             op,
                  const DenseVector<U>* u,
                  const DenseVector<V>* v,
                  Descriptor*           desc )
  {
    std::cout << "Error: Feature not implemented yet!\n";
    return GrB_SUCCESS;
  }

  // Sparse x dense vector
  template <typename W, typename U, typename V, typename M,
            typename BinaryOpT,      typename SemiringT>
  Info eWiseMult( SparseVector<W>*       w,
                  const Vector<M>*       mask,
                  BinaryOpT              accum,
                  SemiringT              op,
                  const SparseVector<U>* u,
                  const DenseVector<V>*  v,
                  Descriptor*            desc )
  {
    std::cout << "Error: Feature not implemented yet!\n";
    return GrB_SUCCESS;
  }

}  // backend
}  // graphblas

#endif  // GrB_BACKEND_APSPIE_EWISEMULT_HPP
