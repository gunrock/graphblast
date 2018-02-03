#ifndef GRB_BACKEND_APSPIE_KERNELS_SPMSPV_HPP
#define GRB_BACKEND_APSPIE_KERNELS_SPMSPV_HPP

#include <iostream>

#include <cuda.h>
#include <cusparse.h>

#include <moderngpu.cuh>
#include <cub.cuh>

namespace graphblas
{
namespace backend
{

  template <bool UseScmp,
            typename W, typename a, typename U, typename M,
            typename AccumOp, typename MulOp, typename AddOp>
  __global__ void spmspvSimpleMaskedKernel( W*           w_val,
                                            const M*     mask_val,
                                            AccumOp      accum_op,
                                            a            identity,
                                            MulOp        mul_op,
                                            AddOp        add_op,
                                            Index        A_nrows,
                                            const Index* A_csrRowPtr,
                                            const Index* A_csrColInd,
                                            const a*     A_csrVal,
                                            const U*     u_val )
  {
    Index col = blockIdx.x*blockDim.x + threadIdx.x;

    for( ; col<A_nrows; col+=gridDim.x*blockDim.x )
    {
      U     val       = __ldg( u_val+col );
      if( val==identity )
        continue;
      Index row_start = __ldg( A_csrRowPtr+col   );
      Index row_end   = __ldg( A_csrRowPtr+col+1 );

      //printf("%d: %d\n", threadIdx.x, row_start);
      for( ; row_start<row_end; row_start++ )
      {
        Index col_ind = __ldg( A_csrColInd+row_start );
        M     m_val   = __ldg( mask_val+col_ind );

        //printf("%d: %d = %d\n", threadIdx.x, col_ind, UseScmp^((bool)m_val));
        if( UseScmp^((bool)m_val) )
        {
          w_val[col_ind] = (W)1;
        }
      }
    }
  }

  template <typename W, typename a, typename U,
            typename AccumOp, typename MulOp, typename AddOp>
  __global__ void spmspvSimpleKernel( W*           w_val,
                                      AccumOp      accum_op,
                                      a            identity,
                                      MulOp        mul_op,
                                      AddOp        add_op,
                                      Index        A_nrows,
                                      const Index* A_csrRowPtr,
                                      const Index* A_csrColInd,
                                      const a*     A_csrVal,
                                      const U*     u_val )
  {
    Index col = blockIdx.x*blockDim.x + threadIdx.x;

    for( ; col<A_nrows; col+=gridDim.x*blockDim.x )
    {
      U val = __ldg( u_val+col );
      Index row_start   = __ldg( A_csrRowPtr+col   );
      Index row_end     = __ldg( A_csrRowPtr+col+1 );

      for( ; row_start<row_end; row_start++ )
      {
      }
    }
  }

}  // backend
}  // graphblas

#endif  // GRB_BACKEND_APSPIE_KERNELS_SPMSPV_HPP
