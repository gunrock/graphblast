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

  template <typename MulOp>
  __global__ void spmvMaskedOrKernel( W*           w_val,
                                      const M*     mask_val,
                                      AccumOp      accum_op,
                                      MulOp        mul_op,
                                      AddOp        add_op,
                                      Index        A_nrows,
                                      Index        A_ncols,
                                      Index        A_nvals,
                                      const Index* A_csrRowPtr,
                                      const Index* A_csrColInd,
                                      const a*     A_csrVal,
                                      const U*     u_val )
  {
    unsigned row = blockIdx.x*blockDim.x + threadIdx.x;

    for( ; row<A_nrows; row+=gridDim.x*blockDim.x )
    {
      U val = __ldg( u_val+row );
      if( val!=-1.f )
        continue;

      
    }
  }

}  // backend
}  // graphblas

#endif  // GRB_BACKEND_APSPIE_KERNELS_SPMV_HPP
