#ifndef GRB_BACKEND_APSPIE_KERNELS_ASSIGNDENSE_HPP
#define GRB_BACKEND_APSPIE_KERNELS_ASSIGNDENSE_HPP

#include <cuda.h>

namespace graphblas
{
namespace backend
{

  // Iterating along u_ind and u_val is better
  template <bool UseScmp, bool UseMask, bool UseAll,
            typename U>
  __global__ void assignDenseKernel( U*               u_val,
                                     Index            u_nvals,
                                     const Index*     mask_ind,
                                     Index            mask_nvals,
                                     const BinaryOp<U,U,U>* accum_op,
                                     U                val,
                                     const Index*     indices,
                                     Index            nindices )
  {
    unsigned row = blockIdx.x*blockDim.x + threadIdx.x;

    if( UseAll )
    {
      if( UseMask )
      {
        if( UseScmp )
        {
          if( row==0 )
          {
            // TODO: requires binary searching the mask array
            printf( "All Indices DeVec Assign Constant Scmp Kernel\n" );
            printf( "Error: Feature not implemented yet!\n" );
          }
        }
        else
        {
          for( ; row<mask_nvals; row+=gridDim.x*blockDim.x )
          {
            Index ind  = __ldg( mask_ind   +row );
            u_val[ind] = val;
          }
        }
      }
      else
      {
        for( ; row<u_nvals; row+=gridDim.x*blockDim.x )
          u_val[row] = val;
      }
    }
    else
    {
      if( row==0 )
      {
        printf( "Selective Indices DeVec Assign Constant Kernel\n" );
        printf( "Error: Feature not implemented yet!\n" );
      }
    }
  }

}  // backend
}  // graphblas

#endif  // GRB_BACKEND_APSPIE_KERNELS_ASSIGNDENSE_HPP
