#ifndef GRB_BACKEND_CUDA_KERNELS_TRACE_HPP
#define GRB_BACKEND_CUDA_KERNELS_TRACE_HPP

namespace graphblas
{
namespace backend
{
  template <typename T, typename a, typename b,
            typename MulOp, typename AddOp>
  __global__ void traceKernel( T*           val,
                               T            identity,
                               MulOp        mul_op,
                               AddOp        add_op,
                               Index        A_nrows,
                               const Index* A_csrRowPtr,
                               const Index* A_csrColInd,
                               const a*     A_csrVal,
                               const Index* B_csrRowPtr,
                               const Index* B_csrColInd,
                               const b*     B_csrVal )
  {
    Index thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    Index warp_id   = thread_id>>5;
    int lane_id     = thread_id & (32 - 1);
    Index row       = warp_id;

    if (row < A_nrows)
    {
      Index A_row_start = __ldg(A_csrRowPtr + row);
      Index A_row_end   = __ldg(A_csrRowPtr + row + 1);
      Index B_row_start = __ldg(B_csrRowPtr + row);
      Index B_row_end   = __ldg(B_csrRowPtr + row + 1);

      Index A_col = -1;
      T A_val     = identity;
      T sum       = identity;
      Index jj    = A_row_start + lane_id;

      for (Index jj_start = A_row_start; jj_start < A_row_end; jj_start += 32)
      {
        if (jj < A_row_end)
        {
          A_col = __ldg(A_csrColInd + jj);
          A_val = __ldg(A_csrVal    + jj);

          Index B_ind = binarySearch(B_csrColInd, A_col, B_row_start, 
              B_row_end);
          Index B_val     = identity;
          if (B_ind != -1)
            B_val = __ldg(B_csrVal + B_ind);

          sum = add_op(sum, mul_op(A_val, B_val));
        }

        jj += 32;
        __syncwarp();
      }

      // Do shuffle reduce here
      #define FULL_MASK 0xffffffff
      for (int offset = 16; offset > 0; offset /= 2)
        sum = add_op(sum, __shfl_down_sync(FULL_MASK, sum, offset));

      // Then have lane_id == 0 atomicAdd it to val
      if (lane_id == 0)
        atomicAdd(val, sum);
      __syncwarp();
    }
  }

  template <typename T, typename a, typename b,
            typename MulOp, typename AddOp>
  __global__ void traceKernelTranspose( T*           val,
                                        T            identity,
                                        MulOp        mul_op,
                                        AddOp        add_op,
                                        Index        A_nrows,
                                        const Index* A_csrRowPtr,
                                        const Index* A_csrColInd,
                                        const a*     A_csrVal,
                                        const Index* B_csrRowPtr,
                                        const Index* B_csrColInd,
                                        const b*     B_csrVal )
  {
    Index thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    Index warp_id   = thread_id>>5;
    int lane_id     = thread_id & (32 - 1);
    Index row       = warp_id;

    if (row < A_nrows)
    {
      Index row_start = __ldg(A_csrRowPtr + row);
      Index row_end   = __ldg(A_csrRowPtr + row + 1);

      Index A_col = -1;
      T A_val     = identity;
      T sum       = identity;
      Index jj    = row_start + lane_id;

      for (Index jj_start = row_start; jj_start < row_end; jj_start += 32)
      {
        if (jj < row_end)
        {
          A_col = __ldg(A_csrColInd + jj);
          A_val = __ldg(A_csrVal    + jj);

          Index B_row_start = __ldg(B_csrRowPtr + A_col);
          Index B_row_end   = __ldg(B_csrRowPtr + A_col + 1);

          Index B_col_ind = binarySearch(B_csrColInd, warp_id, B_row_start, 
              B_row_end);
          Index B_val     = identity;
          if (B_col_ind != -1)
            B_val = __ldg(B_csrVal + B_col_ind);

          sum = add_op(sum, mul_op(A_val, B_val));
        }

        jj += 32;
        __syncwarp();
      }

      // Do shuffle reduce here
      #define FULL_MASK 0xffffffff
      for (int offset = 16; offset > 0; offset /= 2)
        sum = add_op(sum, __shfl_down_sync(FULL_MASK, sum, offset));

      // Then have lane_id == 0 atomicAdd it to val
      if (lane_id == 0)
        atomicAdd(val, sum);
      __syncwarp();
    }
  }
}  // backend
}  // graphblas

#endif  // GRB_BACKEND_CUDA_KERNELS_SPMV_HPP
