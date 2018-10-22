#ifndef GRB_BACKEND_APSPIE_KERNELS_EWISEADD_HPP
#define GRB_BACKEND_APSPIE_KERNELS_EWISEADD_HPP

namespace graphblas
{
namespace backend
{
  // dense-dense dense no mask vector variant
  // TODO(@ctcyang): add scmp, accum, repl, mask
  //template <bool UseScmp, bool UseAccum, bool UseRepl,
  template <typename W, typename U, typename V,
            typename AccumOp, typename AddOp>
  __global__ void eWiseAddKernel( W*       w_val,
                                  AccumOp  accum_op,
                                  U        identity,
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
      if (u_val_t == identity)
        w_val[row] = v_val_t;
      else if (v_val_t == identity)
        w_val[row] = u_val_t;
      else
        w_val[row] = add_op(u_val_t, v_val_t);
      __syncwarp();
    }
  }

  // sparse-dense dense no mask vector variant
  // TODO(@ctcyang): add scmp, accum, repl, mask
  //template <bool UseScmp, bool UseAccum, bool UseRepl,
  template <typename W, typename U,
            typename AccumOp, typename AddOp>
  __global__ void eWiseAddKernel( W*           w_val,
                                  AccumOp      accum_op,
                                  U            identity,
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
      if (w_t == identity)
        w_val[ind] = u_t;
      else if (u_t != identity)
        w_val[ind] = add_op(u_t, w_t);
      __syncwarp();
    }
  }
}  // backend
}  // graphblas

#endif  // GRB_BACKEND_APSPIE_KERNELS_EWISEADD_HPP
