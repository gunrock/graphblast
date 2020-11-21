#ifndef GRAPHBLAS_BACKEND_CUDA_GATHER_HPP_
#define GRAPHBLAS_BACKEND_CUDA_GATHER_HPP_

#include <iostream>

namespace graphblas {
namespace backend {

template <typename W, typename M, typename U, typename I,
          typename BinaryOpT>
Info gatherIndexed(DenseVector<W>*  w,
                   const Vector<M>* mask,
                   BinaryOpT        accum,
                   U*               d_u_val,
                   I*               d_indices,
                   Index            nindices,
                   Descriptor*      desc) {
  bool use_mask = (mask != NULL);

  // Get descriptor parameters for nthreads
  Desc_value nt_mode;
  CHECK(desc->get(GrB_NT, &nt_mode));
  const int nt = static_cast<int>(nt_mode);

  // Get number of elements
  Index w_nvals = 0;
  CHECK(w->nvals(&w_nvals));

  if (use_mask) {
    std::cout << "Error: Masked variant indexed gather not implemented yet!\n";
  } else {
    dim3 NT, NB;
    NT.x = nt;
    NT.y = 1;
    NT.z = 1;
    NB.x = (nindices + nt - 1)/nt;
    NB.y = 1;
    NB.z = 1;
    if (d_indices == NULL) {
      gatherIndexedKernel<<<NB, NT>>>(w->d_val_, w_nvals, d_u_val);
    } else {
      gatherIndexedKernel<<<NB, NT>>>(w->d_val_, w_nvals, d_indices, nindices,
          d_u_val);
    }
  }
  w->need_update_ = true;

  return GrB_SUCCESS;
}
}  // namespace backend
}  // namespace graphblas

#endif  // GRAPHBLAS_BACKEND_CUDA_GATHER_HPP_
