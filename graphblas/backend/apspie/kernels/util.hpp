#ifndef GRB_BACKEND_APSPIE_KERNELS_UTIL_HPP
#define GRB_BACKEND_APSPIE_KERNELS_UTIL_HPP

#include <cuda.h>

namespace graphblas
{
namespace backend
{

  __global__ void updateFlagKernel( Index*       d_flag,
                                    Index        identity,
                                    const Index* u_ind,
                                    Index        u_nvals )
  {
    unsigned row = blockIdx.x*blockDim.x + threadIdx.x;

    for( ; row<u_nvals; row+=gridDim.x*blockDim.x )
    {
      Index ind = __ldg( u_ind+row );
      if( ind==identity )
        d_flag[row] = 0;
      else
        d_flag[row] = 1;
    }
  }

  template <typename W, typename U>
  __global__ void streamCompactKernel( Index* w_ind,
                                       W*     w_val,
                                       Index* d_scan,
                                       Index  identity,
                                       Index* u_ind,
                                       U*     u_val,
                                       Index  u_nvals )
  {
    unsigned row = blockIdx.x*blockDim.x + threadIdx.x;

    for( ; row<u_nvals; row+=gridDim.x*blockDim.x )
    {
      Index ind     = __ldg( u_ind +row );
      U     val     = __ldg( u_val +row );
      Index scatter = __ldg( d_scan+row );

      if(ind!=identity)
      {
        w_ind[scatter] = ind;
        w_val[scatter] = val;
      }
    }
  }

}  // backend
}  // graphblas

#endif  // GRB_BACKEND_APSPIE_KERNELS_APPLY_HPP
