#ifndef GRB_BACKEND_APSPIE_KERNELS_ASSIGNSPARSE_HPP
#define GRB_BACKEND_APSPIE_KERNELS_ASSIGNSPARSE_HPP

#include <cuda.h>

namespace graphblas
{
namespace backend
{

  // Iterating along u_ind and u_val is better
  template <bool UseScmp, bool UseMask, bool UseAll,
            typename U, typename M>
  __global__ void assignSparseKernel( Index*                 u_ind,
                                      U*                     u_val,
                                      Index                  u_nvals,
                                      const M*               mask_val,
                                      M                      mask_identity,
                                      const BinaryOp<U,U,U>* accum_op,
                                      U                      val,
                                      Index*                 indices,
                                      Index                  nindices )
  {
    unsigned row = blockIdx.x*blockDim.x + threadIdx.x;

    if( UseAll )
    {
      for( ; row<u_nvals; row+=gridDim.x*blockDim.x )
      {
        Index ind   = __ldg( u_ind   +row );
        if( UseMask )
        {
          M     m_val = __ldg( mask_val+ind );
          if( UseScmp^(m_val==mask_identity) )
          {
            //printf("Success: %d\n", row);
            u_val[row] = val;
          }
        }
        else
        {
          u_val[row] = val;
        }
      }
    }
    else
    {
      if( row==0 )
      {
        printf( "Selective Indices SpVec Assign Constant Kernel\n" );
        printf( "Error: Feature not implemented yet!\n" );
      }
    }
  }

}  // backend
}  // graphblas

#endif  // GRB_BACKEND_APSPIE_KERNELS_ASSIGNSPARSE_HPP
