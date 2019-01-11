#ifndef GRAPHBLAS_BACKEND_CUDA_KERNELS_UTIL_HPP_
#define GRAPHBLAS_BACKEND_CUDA_KERNELS_UTIL_HPP_

namespace graphblas {
namespace backend {

__device__ Index binarySearch(const Index* array,
                              Index        target,
                              Index        begin,
                              Index        end) {
  while (begin < end) {
    int mid  = begin + (end - begin) / 2;
    int item = array[mid];
    if (item == target)
      return mid;
    bool larger = (item > target);
    if (larger)
      end = mid;
    else
      begin = mid + 1;
  }
  return -1;
}

template <typename U>
__global__ void zeroKernel(U* u_val, U identity, Index u_nvals) {
  Index row = blockIdx.x*blockDim.x + threadIdx.x;
  for (; row < u_nvals; row += gridDim.x*blockDim.x) {
    u_val[row] = identity;
  }
}

template <typename U, typename T, typename M>
__global__ void zeroDenseIdentityKernel(const M* mask,
                                        T        identity,
                                        Index*   u_ind,
                                        U*       u_val,
                                        Index    u_nvals) {
  Index row = blockIdx.x*blockDim.x + threadIdx.x;

  for (; row < u_nvals; row += gridDim.x * blockDim.x) {
    Index ind = u_ind[row];
    M     val = mask[ind];
    if (val == 0)
      u_val[row] = identity;
  }
}

template <typename U>
__global__ void updateFlagKernel(Index*   d_flag,
                                 U        identity,
                                 const U* u_val,
                                 Index    u_nvals) {
  Index row = blockIdx.x*blockDim.x + threadIdx.x;

  for (; row < u_nvals; row += gridDim.x*blockDim.x) {
    U val = u_val[row];
    if (val == identity)
      d_flag[row] = 0;
    else
      d_flag[row] = 1;
  }
}

// sparse key-value variant
template <typename W, typename U>
__global__ void streamCompactSparseKernel(Index*       w_ind,
                                          W*           w_val,
                                          const Index* d_scan,
                                          U            identity,
                                          const Index* u_ind,
                                          const U*     u_val,
                                          Index        u_nvals) {
  Index row = blockIdx.x*blockDim.x + threadIdx.x;

  for (; row < u_nvals; row += gridDim.x*blockDim.x) {
    Index ind     = u_ind[row];
    U     val     = u_val[row];
    Index scatter = d_scan[row];

    if (val != identity) {
      w_ind[scatter] = ind;
      w_val[scatter] = val;
    }
  }
}

// sparse key-only variant
template <typename U>
__global__ void streamCompactSparseKernel(Index*       w_ind,
                                          const Index* d_scan,
                                          U            identity,
                                          const Index* u_ind,
                                          const U*     u_val,
                                          Index        u_nvals) {
  Index row = blockIdx.x*blockDim.x + threadIdx.x;

  for (; row < u_nvals; row += gridDim.x*blockDim.x) {
    Index ind     = u_ind[row];
    U val         = u_val[row];
    Index scatter = d_scan[row];

    if (val == identity)
      w_ind[scatter] = ind;
  }
}

// dense key-only variant
template <typename U>
__global__ void streamCompactDenseKernel(Index*       w_ind,
                                         const Index* d_scan,
                                         Index        identity,
                                         const U*     u_val,
                                         Index        u_nvals) {
  Index row = blockIdx.x*blockDim.x + threadIdx.x;

  for (; row < u_nvals; row += gridDim.x*blockDim.x) {
    Index scatter = d_scan[row];
    U val         = u_val[row];

    if (val == identity)
      w_ind[scatter] = row;
  }
}

// dense key-value variant
template <typename W, typename U>
__global__ void streamCompactDenseKernel(Index*       w_ind,
                                         W*           w_val,
                                         const Index* d_scan,
                                         U            identity,
                                         const U*     u_val,
                                         Index        u_nvals) {
  Index row = blockIdx.x*blockDim.x + threadIdx.x;

  for (; row < u_nvals; row += gridDim.x*blockDim.x) {
    Index scatter = d_scan[row];
    U val         = u_val[row];

    if (val != identity) {
      w_ind[scatter] = row;
      w_val[scatter] = val;
    }
  }
}

__global__ void indirectScanKernel(Index*       d_temp_nvals,
                                   const Index* A_csrRowPtr,
                                   const Index* u_ind,
                                   Index        u_nvals) {
  int gid = blockIdx.x*blockDim.x+threadIdx.x;
  Index length = 0;

  if (gid < u_nvals) {
    Index row   = u_ind[gid];
    Index start = A_csrRowPtr[row];
    Index end   = A_csrRowPtr[row + 1];
    length      = end-start;

    d_temp_nvals[gid] = length;
  }
}

__global__ void indirectGather(Index*       d_temp_nvals,
                               const Index* A_csrRowPtr,
                               const Index* u_ind,
                               Index        u_nvals) {
  int gid = blockIdx.x*blockDim.x+threadIdx.x;

  if (gid < u_nvals) {
    Index   row = u_ind[gid];
    Index start = A_csrRowPtr[row];
    d_temp_nvals[gid] = start;
  }
}

// key-only scatter
template <typename T>
__global__ void scatter(T*           w_val,
                        const Index* u_ind,
                        T            val,
                        Index        u_nvals) {
  int gid = blockIdx.x*blockDim.x+threadIdx.x;

  if (gid < u_nvals) {
    Index ind = u_ind[gid];
    w_val[ind] = val;
  }
}

// key-value scatter
template <typename T>
__global__ void scatter(T*           w_val,
                        const Index* u_ind,
                        const T*     u_val,
                        Index        u_nvals) {
  int gid = blockIdx.x*blockDim.x+threadIdx.x;

  if (gid < u_nvals) {
    Index ind = u_ind[gid];
    T     val = u_val[gid];

    w_val[ind] = val;
  }
}
}  // namespace backend
}  // namespace graphblas

#endif  // GRAPHBLAS_BACKEND_CUDA_KERNELS_UTIL_HPP_
