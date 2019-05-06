#ifndef GRAPHBLAS_BACKEND_CUDA_KERNELS_EWISEMULT_HPP_
#define GRAPHBLAS_BACKEND_CUDA_KERNELS_EWISEMULT_HPP_

namespace graphblas {
namespace backend {
// dense-dense dense no mask vector variant
// TODO(@ctcyang): add scmp, accum, repl, mask
// template <bool UseScmp, bool UseAccum, bool UseRepl,
template <typename W, typename T, typename U, typename V,
          typename AccumOp, typename MulOp>
__global__ void eWiseMultKernel(W*       w_val,
                                AccumOp  accum_op,
                                T        identity,
                                MulOp    mul_op,
                                U*       u_val,
                                V*       v_val,
                                Index    u_nvals) {
  Index row = blockIdx.x * blockDim.x + threadIdx.x;
  for (; row < u_nvals; row += blockDim.x * gridDim.x) {
    U u_val_t = u_val[row];
    V v_val_t = v_val[row];
    if (u_val_t == identity || v_val_t == identity)
      w_val[row] = identity;
    else
      w_val[row] = mul_op(u_val_t, v_val_t);
    __syncwarp();
  }
}

// dense-dense sparse mask vector variant
// TODO(@ctcyang): add scmp, accum, repl, mask
// template <bool UseScmp, bool UseAccum, bool UseRepl,
template <typename W, typename M, typename T, typename U, typename V,
          typename AccumOp, typename MulOp>
__global__ void eWiseMultKernel(Index*       w_ind,
                                W*           w_val,
                                const Index* mask_ind,
                                const M*     mask_val,
                                Index        mask_nvals,
                                AccumOp      accum_op,
                                T            identity,
                                MulOp        mul_op,
                                const U*     u_val,
                                const V*     v_val) {
  Index row = blockIdx.x * blockDim.x + threadIdx.x;
  for (; row < mask_nvals; row += blockDim.x * gridDim.x) {
    Index ind = mask_ind[row];
    M     m_t = mask_val[row];
    W     w_t = 0;
    if (m_t != 0) {
      U      u_t = u_val[ind];
      V      v_t = v_val[ind];
      w_t        = mul_op(u_t, v_t);
    }
    w_ind[row] = ind;
    w_val[row] = w_t;
    __syncwarp();
  }
}

// dense-dense dense mask vector variant
// TODO(@ctcyang): add scmp, accum, repl, mask
// template <bool UseScmp, bool UseAccum, bool UseRepl,
template <typename W, typename M, typename T, typename U, typename V,
          typename AccumOp, typename MulOp>
__global__ void eWiseMultKernel(W*       w_val,
                                AccumOp  accum_op,
                                const M* mask_val,
                                T        identity,
                                MulOp    mul_op,
                                U*       u_val,
                                V*       v_val,
                                Index    u_nvals) {
  Index row = blockIdx.x * blockDim.x + threadIdx.x;
  for (; row < u_nvals; row += blockDim.x * gridDim.x) {
    U u_val_t    = u_val[row];
    V v_val_t    = v_val[row];
    M mask_val_t = mask_val[row];

    if (mask_val_t == 0 || u_val_t == identity || v_val_t == identity)
      w_val[row] = identity;
    else
      w_val[row] = mul_op(u_val_t, v_val_t);
    __syncwarp();
  }
}

// sparse-dense dense no mask vector variant
// TODO(@ctcyang): add scmp, accum, repl, mask
// template <bool UseScmp, bool UseAccum, bool UseRepl,
template <typename W, typename T, typename U, typename V,
          typename AccumOp, typename MulOp>
__global__ void eWiseMultKernel(Index*       w_ind,
                                W*           w_val,
                                AccumOp      accum_op,
                                T            identity,
                                MulOp        mul_op,
                                const Index* u_ind,
                                const U*     u_val,
                                Index        u_nvals,
                                const V*     v_val,
                                bool         reverse) {
  Index row = blockIdx.x * blockDim.x + threadIdx.x;
  for (; row < u_nvals; row += blockDim.x * gridDim.x) {
    Index ind = u_ind[row];
    U     u_t = u_val[row];

    if (u_t != identity) {
      V      v_t = v_val[ind];
      w_val[row] = (reverse) ? mul_op(v_t, u_t) : mul_op(u_t, v_t);
    } else {
      w_val[row] = 0;
    }
    w_ind[row] = ind;
    __syncwarp();
  }
}

// sparse-dense dense mask vector variant
// TODO(@ctcyang): add scmp, accum, repl, mask
// template <bool UseScmp, bool UseAccum, bool UseRepl,
template <typename W, typename M, typename T, typename U, typename V,
          typename AccumOp, typename MulOp>
__global__ void eWiseMultKernel(Index*       w_ind,
                                W*           w_val,
                                const Index* mask_ind,
                                const M*     mask_val,
                                Index        mask_nvals,
                                AccumOp      accum_op,
                                T            identity,
                                MulOp        mul_op,
                                const Index* u_ind,
                                const U*     u_val,
                                Index        u_nvals,
                                const V*     v_val,
                                bool         reverse) {
  Index row = blockIdx.x * blockDim.x + threadIdx.x;
  for (; row < mask_nvals; row += blockDim.x * gridDim.x) {
    Index ind = mask_ind[row];
    M     m_t = mask_val[row];
    W     w_t = 0;

    if (m_t != 0) {
      V     v_t = v_val[ind];

      if (v_t != identity) {
        Index u_found = binarySearch(u_ind, ind, 0, u_nvals);
        if (u_found != -1) {
          U u_t = u_val[u_found];
          w_t   = (reverse) ? mul_op(v_t, u_t) : mul_op(u_t, v_t);
        }
      }
    }
    w_ind[row] = ind;
    w_val[row] = w_t;
    __syncwarp();
  }
}

// Multiply by scalar kernel
template <typename W, typename U, typename V,
          typename AccumOp, typename MulOp>
__global__ void eWiseMultKernel(W*      w_val,
                                AccumOp accum_op,
                                MulOp   mul_op,
                                U*      u_val,
                                Index   u_nvals,
                                V       val) {
  Index row = blockIdx.x * blockDim.x + threadIdx.x;
  for (; row < u_nvals; row += blockDim.x * gridDim.x) {
    U u_t      = u_val[row];
    w_val[row] = mul_op(u_t, val);
  }
}

// Elementwise multiply CSR value array by vector broadcast
template <typename c, typename a, typename b,
          typename AccumOp, typename MulOp>
__global__ void eWiseMultCSRKernel(c*       C_csrVal,
                                   AccumOp  accum_op,
                                   MulOp    mul_op,
                                   Index*   A_csrRowPtr,
                                   a*       A_csrVal,
                                   Index    A_nrows,
                                   const b* B_val) {
  Index thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  Index warp_id   = thread_id / 32;
  Index lane_id   = thread_id & (32 - 1);
  if (warp_id < A_nrows) {
    Index row_start = A_csrRowPtr[warp_id];
    Index row_end   = A_csrRowPtr[warp_id+1];
    b B_t = B_val[warp_id];

    Index ind = row_start + lane_id;
    for (Index ind_start = row_start; ind_start < row_end; ind_start += 32) {
      if (ind < row_end) {
        a A_t = A_csrVal[ind];
        c C_t = mul_op(A_t, B_t);
        C_csrVal[ind] = C_t;
      }
      ind += 32;
    }
  }
}

// Elementwise multiply CSC value array by vector broadcast
template <typename c, typename a, typename b,
          typename AccumOp, typename MulOp>
__global__ void eWiseMultCSCKernel(c*       C_cscVal,
                                   AccumOp  accum_op,
                                   MulOp    mul_op,
                                   Index*   A_cscColPtr,
                                   Index*   A_cscRowInd,
                                   a*       A_cscVal,
                                   Index    A_ncols,
                                   const b* B_val) {
  Index thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  Index warp_id   = thread_id / 32;
  Index lane_id   = thread_id & (32 - 1);
  if (warp_id < A_ncols) {
    Index col_start = A_cscColPtr[warp_id];
    Index col_end   = A_cscColPtr[warp_id+1];

    Index ind = col_start + lane_id;
    for (Index ind_start = col_start; ind_start < col_end; ind_start += 32) {
      if (ind < col_end) {
        Index B_ind = A_cscRowInd[ind];
        b B_t = B_val[B_ind];

        a A_t = A_cscVal[ind];
        c C_t = mul_op(A_t, B_t);
        C_cscVal[ind] = C_t;
      }
      ind += 32;
    }
  }
}
}  // namespace backend
}  // namespace graphblas

#endif  // GRAPHBLAS_BACKEND_CUDA_KERNELS_EWISEMULT_HPP_
