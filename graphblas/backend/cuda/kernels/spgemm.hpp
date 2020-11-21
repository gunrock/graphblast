#ifndef GRAPHBLAS_BACKEND_CUDA_KERNELS_SPGEMM_HPP_
#define GRAPHBLAS_BACKEND_CUDA_KERNELS_SPGEMM_HPP_

namespace graphblas {
namespace backend {
// Sparse matrix-Sparse matrix multiplication with sparse matrix mask
// Strategy:
// 1) Loop through mask using 1 warp/row
// 2) For each nonzero (row, col) of mask:
//    i)   initialize each thread to identity
//    ii)  compute dot-product A(row, :) x B(:, col)
//    iii) use warp on each nonzero at a time
//    iv)  tally up accumulated sum using warp reduction
//    v)   write to global memory C_csrVal
template <typename c, typename a, typename b, typename m,
          typename MulOp, typename AddOp>
__global__ void spgemmMaskedKernel(c*           C_csrVal,
                                   const Index* mask_csrRowPtr,
                                   const Index* mask_csrColInd,
                                   m*           mask_csrVal,
                                   MulOp        mul_op,
                                   AddOp        add_op,
                                   c            identity,
                                   const Index* A_csrRowPtr,
                                   const Index* A_csrColInd,
                                   const a*     A_csrVal,
                                   const Index* B_cscColPtr,
                                   const Index* B_cscRowInd,
                                   const b*     B_cscVal,
                                   Index        A_nrows) {
  Index thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  Index warp_id   = thread_id / 32;
  Index lane_id   = thread_id & (32 - 1);
  if (warp_id < A_nrows) {
    Index row_start = mask_csrRowPtr[warp_id];
    Index row_end   = mask_csrRowPtr[warp_id+1];

    // Entire warp works together on each nonzero
    for (Index edge = row_start; edge < row_end; ++edge) {
      m mask_val = mask_csrVal[edge];
      c accumulator = identity;
      if (mask_val) {
        // Load B bounds on which we must do binary search
        Index B_ind       = A_csrColInd[edge];
        Index B_col_start = B_cscColPtr[B_ind];
        Index B_col_end   = B_cscColPtr[B_ind+1];
        
        // Each thread iterates along row
        // Does binary search on B_row to try to find A_col
        // Adds result to accumulator if found
        Index ind = row_start + lane_id;
        for (Index ind_start = row_start; ind_start < row_end;
            ind_start += 32) {
          if (ind < row_end) {
            Index A_col = A_csrColInd[ind];
            Index B_row = binarySearch(B_cscRowInd, A_col, B_col_start,
                B_col_end);

            if (B_row != -1) {
              a A_t = A_csrVal[ind];
              b B_t = B_cscVal[B_row];
              c C_t = mul_op(A_t, B_t);
              accumulator = add_op(C_t, accumulator);
            }
          }
          ind += 32;
        }

        // Warp reduce for each edge
        for (int i = 1; i < 32; i *= 2)
          accumulator = add_op(__shfl_xor_sync(-1, accumulator, i),
              accumulator);
      }
      // Write to output
      if (lane_id == 0)
        C_csrVal[edge] = accumulator;
    }
  }
}

// This kernel and the following kernels are from the CUDA-based pseudocode from
// Naumov et al. AmgX: A library for GPU accelerated algebraic multigrid and
// preconditioned iterative methods (SIAM J. Sci. Comp., 2015).
__global__ csrgemm_count_kernel(int n, int *Ap, int *Ai, double *Av,
                                int *Pp, int *Pi, double *Pv,
                                int *Zp, int *Zi, double *Zv) {
  int row, col, j, k;
  // For each row of A in parallel.
  for (row = threadIdx.z + blockIdx.z * blockDim.z; row < n;
       row += blockDim.z * gridDim.z) {
    // For each column of A in parallel.
    for (j = Ap[row] + threadIdx.y + blockIdx.y * blockDim.y; j < Ap[row + 1];
         j += blockDim.y * gridDim.y) {
      // This represents the row of P.
      col = Ai[j];
      // For each column of P in parallel.
      for (k = Pp[col] + threadIdx.x + blockIdx.x * blockDim.x; k < Pp[col + 1];
           k += blockDim.x * gridDim.x) {
        // This represents the key.
        col_P = Pi[k];
        // Perform union to eliminate duplicate keys.
        hashTable[row].insert(col_P);
      }
    }
    Zp[row] = hashTable[row].size();
  }
}

__global__ csrgemm_compute_kernel(int n, int *Ap, int *Ai, double *Av,
                                  int *Pp, int *Pi, double *Pv,
                                  int *Zp, int *Zi, double *Zv) {
  int row, col, j, k;
  double val, val_P;
  // For each row of A in parallel.
  for (row = threadIdx.z + blockIdx.z*blockDim.z; row < n;
       row += blockDim.z * gridDim.z) {
    // For each col of A in parallel.
    for (j = Ap[row] + threadIdx.y + blockIdx.y * blockDim.y; j < Ap[row + 1];
         j+= blockDim.y * gridDim.y) {
      // This represents the row of P.
      col = Ai[j];
      val = Av[j];
      // For each col of P in parallel.
      for (k = Pp[col] + threadIdx.x + blockIdx.x * blockDim.x; k < Pp[col + 1];
           k += blockDim.x * gridDim.x) {
        // The next two variables represent the key and value respectively.
        col_P = Pi[k];
        val_P = Pv[k];
        // Perform union to reduce values on key conflict.
        hashTable[row].insert_by_key(col_P, val * val_P);
      }
    }
    Zi[row] = hashTable[row].export_keys()
    Zv[row] = hashTable[row].export_values();
  }
}
}  // namespace backend
}  // namespace graphblas

#endif  // GRAPHBLAS_BACKEND_CUDA_KERNELS_SPGEMM_HPP_
