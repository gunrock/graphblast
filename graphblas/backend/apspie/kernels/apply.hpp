#ifndef GRB_BACKEND_APSPIE_KERNELS_APPLY_HPP
#define GRB_BACKEND_APSPIE_KERNELS_APPLY_HPP

#include <cuda.h>

namespace graphblas
{
namespace backend
{

  // Iterating along u_ind and u_val is better
  template <bool UseScmp, bool UseAccum, bool UseRepl,
            typename W, typename U, typename M,
            typename BinaryOpT, typename UnaryOp>
  __global__ void applyKernel( Index*           w_ind,
                               W*               w_val,
                               const M*         mask_val,
                               const BinaryOpT* accum_op,
                               U                identity,
                               UnaryOp          op,
                               const Index*     u_ind,
                               const U*         u_val,
                               Index            u_nvals)
  {
    unsigned row = blockIdx.x*blockDim.x + threadIdx.x;

    for( ; row<u_nvals; row+=gridDim.x*blockDim.x )
    {
      Index ind = __ldg( u_ind   +row );
      M     val = __ldg( mask_val+ind );
      if( UseScmp^(val!=identity) )
      {
        w_ind[row] = -1;
        w_val[row] = identity;
      }
      else
      {
        w_ind[row] = ind;
        w_val[row] = op(__ldg(u_val+row));
      }
    }
  }

}  // backend
}  // graphblas

#endif  // GRB_BACKEND_APSPIE_KERNELS_APPLY_HPP
