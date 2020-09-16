#ifndef GRAPHBLAS_BACKEND_CUDA_KERNELS_SCATTER_HPP_
#define GRAPHBLAS_BACKEND_CUDA_KERNELS_SCATTER_HPP_

namespace graphblas {
namespace backend {
// no mask vector constant variant for both sparse and dense
template <typename W, typename U, typename T>
__global__ void scatterKernel(W*       w_val,
                              Index    w_nvals,
                              U*       u_val,
                              Index    u_nvals,
                              T        val) {
  Index row = blockIdx.x * blockDim.x + threadIdx.x;
  for (; row < u_nvals; row += blockDim.x * gridDim.x) {
    Index ind = static_cast<Index>(u_val[row]);
    if (ind > 0 && ind < w_nvals)
      w_val[ind] = val;
    __syncwarp();
  }
}

// no mask vector indexed variant for both sparse and dense
template <typename W, typename V>
__global__ void scatterIndexedKernel(W*       w_val,
                                     Index    w_nvals,
                                     Index*   u_val,
                                     Index    u_nvals,
                                     V*       v_val) {
  Index row = blockIdx.x * blockDim.x + threadIdx.x;
  for (; row < u_nvals; row += blockDim.x * gridDim.x) {
    Index ind = u_val[row];
    V val = v_val[row];
    if (ind > 0 && ind < w_nvals)
      w_val[ind] = val;
    __syncwarp();
  }
}

// no mask vector indexed variant for both sparse and dense
template <typename W, typename V>
__global__ void scatterIndexedKernel(W*       w_val,
                                     Index    w_nvals,
                                     V*       v_val) {
  Index row = blockIdx.x * blockDim.x + threadIdx.x;
  for (; row < w_nvals; row += blockDim.x * gridDim.x) {
    V val = v_val[row];
    w_val[row] = val;
    __syncwarp();
  }
}
}  // namespace backend
}  // namespace graphblas

#endif  // GRAPHBLAS_BACKEND_CUDA_KERNELS_SCATTER_HPP_
