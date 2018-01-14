#ifndef GRB_BACKEND_APSPIE_KERNELS_APPLY_HPP
#define GRB_BACKEND_APSPIE_KERNELS_APPLY_HPP

#include <cuda.h>

namespace graphblas
{
namespace backend
{

  // Iterating along u_ind and u_val is better
  template <bool UseScmp,
            typename U, typename M,
            typename BinaryOpT, typename UnaryOp>
  __global__ void applyKernel( Index*           u_ind,
                               U*               u_val,
                               const M*         mask_val,
                               const BinaryOpT* accum_op,
                               M                identity,
                               UnaryOp          op,
                               Index            u_nvals)
  {
    unsigned row = blockIdx.x*blockDim.x + threadIdx.x;

    for( ; row<u_nvals; row+=gridDim.x*blockDim.x )
    {
      Index ind = __ldg( u_ind   +row );
      M     val = __ldg( mask_val+ind );
      if( UseScmp^(val==identity) )
      {
        printf("Success: %d\n", row);
        u_ind[row] = -1;
      }
    }
  }

}  // backend
}  // graphblas

#endif  // GRB_BACKEND_APSPIE_KERNELS_APPLY_HPP
