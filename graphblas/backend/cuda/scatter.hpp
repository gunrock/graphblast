#ifndef GRAPHBLAS_BACKEND_CUDA_SCATTER_HPP_
#define GRAPHBLAS_BACKEND_CUDA_SCATTER_HPP_

#include <iostream>

namespace graphblas {
namespace backend {

// Dense vector constant variant
template <typename W, typename M, typename U, typename T>
Info scatterDense(DenseVector<W>*       w,
                  const Vector<M>*      mask,
                  const DenseVector<U>* u,
                  T                     val,
                  Descriptor*           desc)  {
  bool use_mask = (mask != NULL);

  // Get descriptor parameters for nthreads
  Desc_value nt_mode;
  CHECK(desc->get(GrB_NT, &nt_mode));
  const int nt = static_cast<int>(nt_mode);

  // Get number of elements
  Index u_nvals;
  u->nvals(&u_nvals);

  if (use_mask) {
    std::cout << "Error: Masked variant scatter not implemented yet!\n";
  } else {
    dim3 NT, NB;
    NT.x = nt;
    NT.y = 1;
    NT.z = 1;
    NB.x = (u_nvals + nt - 1) / nt;
    NB.y = 1;
    NB.z = 1;

    scatterKernel<<<NB, NT>>>(w->d_val_, u_nvals, u->d_val_, u_nvals, val);
  }
  w->need_update_ = true;

  return GrB_SUCCESS;
}

// Sparse vector constant variant
template <typename W, typename M, typename U, typename T>
Info scatterSparse(DenseVector<W>*        w,
                   const Vector<M>*       mask,
                   const SparseVector<U>* u,
                   T                      val,
                   Descriptor*            desc) {
  bool use_mask = (mask != NULL);

  // Get descriptor parameters for nthreads
  Desc_value nt_mode;
  CHECK(desc->get(GrB_NT, &nt_mode));
  const int nt = static_cast<int>(nt_mode);

  // Get number of elements
  Index u_nvals, w_nvals;
  u->nvals(&u_nvals);
  w->nvals(&w_nvals);

  if (use_mask) {
    std::cout << "Error: Masked variant scatter not implemented yet!\n";
  } else {
    dim3 NT, NB;
    NT.x = nt;
    NT.y = 1;
    NT.z = 1;
    NB.x = (u_nvals + nt - 1) / nt;
    NB.y = 1;
    NB.z = 1;

    scatterKernel<<<NB, NT>>>(w->d_val_, w_nvals, u->d_val_, u_nvals, val);
  }
  w->need_update_ = true;

  return GrB_SUCCESS;
}

// Dense vector indexed variant
template <typename W, typename M, typename U,
          typename BinaryOpT>
Info scatterIndexed(DenseVector<W>*       w,
                    const Vector<M>*      mask,
                    BinaryOpT             accum,
                    const DenseVector<U>* u,
                    int*                  indices,
                    Index                 nindices,
                    Descriptor*           desc) {
  bool use_mask = (mask != NULL);

  // Get descriptor parameters for nthreads
  Desc_value nt_mode;
  CHECK(desc->get(GrB_NT, &nt_mode));
  const int nt = static_cast<int>(nt_mode);

  // Get number of elements
  int w_nvals = 0;
  CHECK(w->nvals(&w_nvals));

  if (use_mask) {
    std::cout << "Error: Masked variant indexed scatter not implemented yet!\n";
  } else {
    dim3 NT, NB;
    NT.x = nt;
    NT.y = 1;
    NT.z = 1;
    NB.x = (nindices + nt - 1)/nt;
    NB.y = 1;
    NB.z = 1;
    if (indices == NULL) {
      scatterIndexedKernel<<<NB, NT>>>(w->d_val_, w_nvals, u->d_val_);
    } else {
      // TODO(ctcyang): implement non-GrB_ALL variant of this kernel.
      //scatterIndexedKernel<<<NB, NT>>>(w->d_val_, w_nvals,
      //    (indices->dense_).d_val_, nindices, u->d_val_);
      return GrB_NOT_IMPLEMENTED;
    }
  }
  w->need_update_ = true;

  return GrB_SUCCESS;
}
}  // namespace backend
}  // namespace graphblas

#endif  // GRAPHBLAS_BACKEND_CUDA_SCATTER_HPP_
