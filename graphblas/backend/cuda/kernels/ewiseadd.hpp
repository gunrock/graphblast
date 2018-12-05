#ifndef GRB_BACKEND_CUDA_KERNELS_EWISEADD_HPP
#define GRB_BACKEND_CUDA_KERNELS_EWISEADD_HPP

namespace graphblas
{
namespace backend
{
  // dense-dense dense no mask vector variant
  // TODO(@ctcyang): add scmp, accum, repl, mask
  //template <bool UseScmp, bool UseAccum, bool UseRepl,
  template <typename W, typename U, typename V,
            typename AccumOp, typename AddOp>
  __global__ void eWiseAddDenseDenseKernel( W*       w_val,
                                            AccumOp  accum_op,
                                            AddOp    add_op,
                                            U*       u_val,
                                            V*       v_val,
                                            Index    u_nvals )
  {
    Index row = blockIdx.x * blockDim.x + threadIdx.x;
    for (; row < u_nvals; row += blockDim.x * gridDim.x)
    {
      U u_val_t = u_val[row];
      V v_val_t = v_val[row];
      w_val[row] = add_op(u_val_t, v_val_t);
      __syncwarp();
    }
  }

  // sparse-dense dense no mask vector variant
  // TODO(@ctcyang): add scmp, accum, repl, mask
  //template <bool UseScmp, bool UseAccum, bool UseRepl,
  template <typename W, typename U,
            typename AccumOp, typename AddOp>
  __global__ void eWiseAddSparseDenseKernel( W*           w_val,
                                             AccumOp      accum_op,
                                             AddOp        add_op,
                                             const Index* u_ind,
                                             const U*     u_val,
                                             Index        u_nvals )
  {
    Index row = blockIdx.x * blockDim.x + threadIdx.x;
    for (; row < u_nvals; row += blockDim.x * gridDim.x)
    {
      Index ind = u_ind[row];
      U     u_t = u_val[row];

      W     w_t = w_val[ind];
      w_val[ind] = add_op(u_t, w_t);
      //printf("tid:%d, u_t:%f, v_t:%f, w_t:%f\n", row, u_t, w_t, w_val[ind]);
      __syncwarp();
    }
  }
}  // backend
}  // graphblas

#endif  // GRB_BACKEND_CUDA_KERNELS_EWISEADD_HPP
