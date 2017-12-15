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

  template <bool UseScmp, bool UseAccum, bool UseRepl,
            typename W, typename a, typename U,
            typename AccumOp, typename MulOp, typename AddOp>
  __global__ void spmspvKernel( W*           w_val,
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
  }

  //__global__ void filterKernel();

}  // backend
}  // graphblas

#endif  // GRB_BACKEND_APSPIE_KERNELS_SPMSPV_HPP
