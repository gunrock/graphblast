#ifndef GRB_BACKEND_CUDA_KERNELS_SPMSPV_HPP
#define GRB_BACKEND_CUDA_KERNELS_SPMSPV_HPP

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

  template <bool UseScmp,
            typename W, typename U, typename M>
  __global__ void setDenseMaskKernel( W*       w_val,
                                      const M* mask_val,
                                      const U  identity,
                                      Index    w_nvals )
  {
    Index row = blockIdx.x * blockDim.x + threadIdx.x;

    for (; row < w_nvals; row += gridDim.x * blockDim.x)
    {
      M val = __ldg(mask_val + row);
      //printf("ind: %d, use_scmp: %d, mask: %d, UseScmp ^ (!(val)): %d\n", row, UseScmp, val, UseScmp ^ (val == 0));
      if (UseScmp ^ (!(bool)val))
        w_val[row] = identity;
    }
  }

  template <typename AddOp>
  __device__ double atomic(double* address, double val, AddOp add_op)
  {
    unsigned long long int* address_as_ull =
        (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do
    {
      assumed = old;
      // Note: uses integer comparison to avoid hang in case of NaN
      // (since NaN != NaN)
      old = atomicCAS(address_as_ull, assumed, 
          __double_as_longlong(add_op(__longlong_as_double(assumed), val)));
    } while (assumed != old);
    return __longlong_as_double(old);
  }

  template <typename AddOp>
  __device__ float atomic(float* address, float val, AddOp add_op) {
    int* address_as_int = (int*)address;
    int old = *address_as_int, assumed;
    do
    {
      assumed = old;
      // Note: uses integer comparison to avoid hang in case of NaN
      // (since NaN != NaN)
      old = atomicCAS(address_as_int, assumed,
          __float_as_int(add_op(__int_as_float(assumed), val)));
    } while (assumed != old);
    return __int_as_float(old);
  }

  /*!
   * \brief Not load-balanced, naive Sparse Matrix x Sparse Vector kernel
   *        for functors with add_op != Plus
   */
  template <typename W, typename a, typename U,
            typename AccumOp, typename MulOp, typename AddOp>
  __global__ void spmspvSimpleKernel( W*           w_val,
                                      AccumOp      accum_op,
                                      a            identity,
                                      MulOp        mul_op,
                                      AddOp        add_op,
                                      const Index* A_csrRowPtr,
                                      const Index* A_csrColInd,
                                      const a*     A_csrVal,
                                      const Index* u_ind,
                                      const U*     u_val,
                                      Index        u_nvals )
  {
    Index row = blockIdx.x * blockDim.x + threadIdx.x;

    for (; row < u_nvals; row += gridDim.x * blockDim.x)
    {
      Index ind = __ldg( u_ind+row );
      U     val = __ldg( u_val+row );
      Index row_start   = __ldg( A_csrRowPtr+ind   );
      Index row_end     = __ldg( A_csrRowPtr+ind+1 );

      for (; row_start < row_end; row_start++)
      {
        Index dest_ind = __ldg( A_csrColInd+row_start );
        a     dest_val = __ldg( A_csrVal   +row_start );

        atomic( w_val+dest_ind, mul_op(val, dest_val), add_op );
      }
    }
  }

  /*!
   * \brief Not load-balanced, naive Sparse Matrix x Sparse Vector kernel
   *        specialized for functor being add_op == Plus
   */
  template <typename W, typename a, typename U,
            typename AccumOp, typename MulOp>
  __global__ void spmspvSimpleAddKernel( W*           w_val,
                                         AccumOp      accum_op,
                                         a            identity,
                                         MulOp        mul_op,
                                         const Index* A_csrRowPtr,
                                         const Index* A_csrColInd,
                                         const a*     A_csrVal,
                                         const Index* u_ind,
                                         const U*     u_val,
                                         Index        u_nvals )
  {
    Index row = blockIdx.x * blockDim.x + threadIdx.x;

    for (; row < u_nvals; row += gridDim.x * blockDim.x)
    {
      Index ind = __ldg( u_ind+row );
      U     val = __ldg( u_val+row );
      Index row_start   = __ldg( A_csrRowPtr+ind   );
      Index row_end     = __ldg( A_csrRowPtr+ind+1 );

      for (; row_start < row_end; row_start++)
      {
        Index dest_ind = __ldg( A_csrColInd+row_start );
        a     dest_val = __ldg( A_csrVal   +row_start );

        atomicAdd( w_val+dest_ind, mul_op(val, dest_val) );
      }
    }
  }

  /*!
   * \brief Not load-balanced, naive Sparse Matrix x Sparse Vector kernel
   *        specialized for functor being add_op == Plus
   */
  template <typename W, typename a, typename U,
            typename AccumOp, typename MulOp>
  __global__ void spmspvSimpleOrKernel( W*           w_val,
                                        AccumOp      accum_op,
                                        a            identity,
                                        MulOp        mul_op,
                                        const Index* A_csrRowPtr,
                                        const Index* A_csrColInd,
                                        const a*     A_csrVal,
                                        const Index* u_ind,
                                        const U*     u_val,
                                        Index        u_nvals )
  {
    Index row = blockIdx.x * blockDim.x + threadIdx.x;

    for (; row < u_nvals; row += gridDim.x * blockDim.x)
    {
      Index ind = __ldg( u_ind+row );
      U     val = __ldg( u_val+row );
      Index row_start   = __ldg( A_csrRowPtr+ind   );
      Index row_end     = __ldg( A_csrRowPtr+ind+1 );

      for (; row_start < row_end; row_start++)
      {
        Index dest_ind = __ldg( A_csrColInd+row_start );
        a     dest_val = __ldg( A_csrVal   +row_start );

        w_val[dest_ind] = 1;
      }
    }
  }
}  // backend
}  // graphblas

#endif  // GRB_BACKEND_CUDA_KERNELS_SPMSPV_HPP
