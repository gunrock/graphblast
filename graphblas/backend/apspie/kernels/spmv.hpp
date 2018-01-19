#ifndef GRB_BACKEND_APSPIE_KERNELS_SPMV_HPP
#define GRB_BACKEND_APSPIE_KERNELS_SPMV_HPP

#include <iostream>

#include <cuda.h>
#include <cusparse.h>

#include <moderngpu.cuh>
#include <cub.cuh>

namespace graphblas
{
namespace backend
{

  template <bool UseScmp, bool UseEarlyExit, bool UseOpReuse,
            typename W, typename a, typename U, typename M,
            typename AccumOp, typename MulOp, typename AddOp>
  __global__ void spmvDenseMaskedOrKernel( W*           w_val,
                                           const M*     mask_val,
                                           M            mask_identity,
                                           AccumOp      accum_op,
                                           a            identity,
                                           MulOp        mul_op,
                                           AddOp        add_op,
                                           Index        A_nrows,
                                           Index        A_nvals,
                                           const Index* A_csrRowPtr,
                                           const Index* A_csrColInd,
                                           const a*     A_csrVal,
                                           const U*     u_val )
  {
    unsigned row = blockIdx.x*blockDim.x + threadIdx.x;

    for( ; row<A_nrows; row+=gridDim.x*blockDim.x )
    {
      bool discoverable = false;

      M val = __ldg( mask_val+row );
      if( UseScmp^(val==mask_identity) )
      {
      }
      else
      {
        Index row_start   = __ldg( A_csrRowPtr+row   );
        Index row_end     = __ldg( A_csrRowPtr+row+1 );

        for( ; row_start<row_end; row_start++ )
        {
          Index col_ind   = __ldg( A_csrColInd+row_start );
          if( UseOpReuse )
          {
            val           = __ldg( mask_val+col_ind );
            if( val!=mask_identity )
            {
              discoverable = true;
              if( UseEarlyExit )
                break;
            }
          }
          else
          {
            val           = __ldg( u_val+col_ind );
            // Early exit if visited parent is discovered
            if( val!=identity )
            {
              discoverable = true;
              if( UseEarlyExit )
              break;
            }
          }
        }
      }

      if( discoverable )
        w_val[row] = 1.f;
      else
        w_val[row] = 0.f;
    }
  }

}  // backend
}  // graphblas

#endif  // GRB_BACKEND_APSPIE_KERNELS_SPMV_HPP
