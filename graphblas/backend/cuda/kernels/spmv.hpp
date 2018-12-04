#ifndef GRB_BACKEND_CUDA_KERNELS_SPMV_HPP
#define GRB_BACKEND_CUDA_KERNELS_SPMV_HPP

namespace graphblas
{
namespace backend
{

  template <bool UseScmp, bool UseEarlyExit, bool UseOpReuse,
            typename W, typename a, typename U, typename M,
            typename AccumOp, typename MulOp, typename AddOp>
  __global__ void spmvDenseMaskedOrKernel( W*           w_val,
                                           const M*     mask_val,
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
    Index row = blockIdx.x*blockDim.x + threadIdx.x;

    for( ; row<A_nrows; row+=gridDim.x*blockDim.x )
    {
      bool discoverable = false;

      M val = mask_val[row];
      if( UseScmp^(!(bool)val) )
      {
      }
      else
      {
        Index row_start   = A_csrRowPtr[row];
        Index row_end     = A_csrRowPtr[row + 1];

        for( ; row_start<row_end; row_start++ )
        {
          Index col_ind   = A_csrColInd[row_start];
          if( UseOpReuse )
          {
            val           = mask_val[col_ind];
            if( ((bool)val) )
            {
              discoverable = true;
              if( UseEarlyExit )
                break;
            }
          }
          else
          {
            val           = u_val[col_ind];
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
        w_val[row] = (W)1;
      else
        w_val[row] = (W)0;
    }
  }

  template <bool UseScmp, bool UseEarlyExit, bool UseOpReuse,
            typename W, typename a, typename U, typename M,
            typename AccumOp, typename MulOp, typename AddOp>
  __global__ void spmvDenseMaskedOrKernelBench( W*           w_val,
                                           int*         stats,
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
    Index row = blockIdx.x*blockDim.x + threadIdx.x;

    for( ; row<A_nrows; row+=gridDim.x*blockDim.x )
    {
      bool discoverable = false;

      M val = mask_val[row];
      if( UseScmp^(val==mask_identity) )
      {
        stats[row] = 0;
      }
      else
      {
        Index row_start   = A_csrRowPtr[row];
        Index row_end     = A_csrRowPtr[row + 1];
        Index row_start2  = row_start;

        for( ; row_start<row_end; row_start++ )
        {
          Index col_ind   = A_csrColInd[row_start];
          if( UseOpReuse )
          {
            val           = mask_val[col_ind];
            if( val!=mask_identity )
            {
              discoverable = true;
              stats[row] = row_start-row_start2;
              break;
            }
            else
            {
              if( row_start==row_end-1 )
                stats[row] = row_end-row_start2;
            }
          }
          else
          {
            val           = u_val[col_ind];
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
        w_val[row] = (W)1;
      else
      {
        w_val[row] = (W)0;
      }
    }
  }

}  // backend
}  // graphblas

#endif  // GRB_BACKEND_CUDA_KERNELS_SPMV_HPP
