#ifndef GRB_BACKEND_APSPIE_KERNELS_EWISEMULT_HPP
#define GRB_BACKEND_APSPIE_KERNELS_EWISEMULT_HPP

#include <iostream>

#include <cuda.h>
#include <cusparse.h>

#include <moderngpu.cuh>
#include <cub.cuh>

namespace graphblas
{
namespace backend
{

  template <bool UseScmp, bool UseAccum, bool UseRepl,
            typename W, typename U, typename V, typename M,
            typename BinaryOpT, typename MulOp>
  __global__ void eWiseMultKernel( Index*           w_ind,
                                   W*               w_val,
                                   const M*         mask_val,
                                   const BinaryOpT* accum_op,
                                   U                identity,
                                   MulOp            mul_op,
                                   const Index*     u_ind,
                                   const U*         u_val,
                                   Index            u_nvals,
                                   const V*         v_val )
  {
    Index row = blockIdx.x*blockDim.x + threadIdx.x;

    for( ; row<u_nvals; row+=gridDim.x*blockDim.x )
    {
      Index ind = __ldg( u_ind+row );
      V     val = __ldg( v_val+ind );
      if( UseScmp^(val!=identity) )
      {
        w_ind[row] = -1;
        w_val[row] = identity;
      }
      else
      {
        w_ind[row] = ind;
        w_val[row] = __ldg(u_val+row);
      }
    }
  }

}  // backend
}  // graphblas

#endif  // GRB_BACKEND_APSPIE_KERNELS_EWISEMULT_HPP
