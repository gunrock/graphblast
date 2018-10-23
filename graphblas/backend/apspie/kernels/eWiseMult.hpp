#ifndef GRB_BACKEND_APSPIE_KERNELS_EWISEMULT_HPP
#define GRB_BACKEND_APSPIE_KERNELS_EWISEMULT_HPP

namespace graphblas
{
namespace backend
{
  // dense-dense dense no mask vector variant
  // TODO(@ctcyang): add scmp, accum, repl, mask
  //template <bool UseScmp, bool UseAccum, bool UseRepl,
  template <typename W, typename U, typename V,
            typename AccumOp, typename MulOp>
  __global__ void eWiseMultKernel( W*       w_val,
                                   AccumOp  accum_op,
                                   U        identity,
                                   MulOp    mul_op,
                                   U*       u_val,
                                   V*       v_val,
                                   Index    u_nvals )
  {
    Index row = blockIdx.x * blockDim.x + threadIdx.x;
    for (; row < u_nvals; row += blockDim.x * gridDim.x)
    {
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
  //template <bool UseScmp, bool UseAccum, bool UseRepl,
  template <typename W, typename U, typename V, typename M,
            typename AccumOp, typename MulOp>
  __global__ void eWiseMultKernel( Index*       w_ind,
                                   W*           w_val,
                                   const Index* mask_ind,
                                   const M*     mask_val,
                                   Index        mask_nvals,
                                   AccumOp      accum_op,
                                   U            identity,
                                   MulOp        mul_op,
                                   const U*     u_val,
                                   const V*     v_val )
  {
    Index row = blockIdx.x * blockDim.x + threadIdx.x;
    for (; row < mask_nvals; row += blockDim.x * gridDim.x)
    {
      Index ind = mask_ind[row];
      M     m_t = mask_val[row];
      W     w_t = 0;
      if (m_t != 0)
      {
        U      u_t = __ldg(u_val + ind);
        V      v_t = __ldg(v_val + ind);
        w_t        = mul_op(u_t, v_t);
      }
      w_ind[row] = ind;
      w_val[row] = w_t;
      __syncwarp();
    }
  }

  // dense-dense dense mask vector variant
  // TODO(@ctcyang): add scmp, accum, repl, mask
  //template <bool UseScmp, bool UseAccum, bool UseRepl,
  template <typename W, typename U, typename V, typename M,
            typename AccumOp, typename MulOp>
  __global__ void eWiseMultKernel( W*       w_val,
                                   AccumOp  accum_op,
                                   const M* mask_val,
                                   U        identity,
                                   MulOp    mul_op,
                                   U*       u_val,
                                   V*       v_val,
                                   Index    u_nvals )
  {
    Index row = blockIdx.x * blockDim.x + threadIdx.x;
    for (; row < u_nvals; row += blockDim.x * gridDim.x)
    {
      U u_val_t    = u_val[row];
      V v_val_t    = v_val[row];
      M mask_val_t = mask_val[row];

      if (mask_val_t == 0 | u_val_t == identity || v_val_t == identity)
        w_val[row] = identity;
      else
        w_val[row] = mul_op(u_val_t, v_val_t);
      __syncwarp();
    }
  }

  // dense-dense sparse mask vector variant
  // sparse-dense dense no mask vector variant
  // TODO(@ctcyang): add scmp, accum, repl, mask
  //template <bool UseScmp, bool UseAccum, bool UseRepl,
  template <typename W, typename U, typename V,
            typename AccumOp, typename MulOp>
  __global__ void eWiseMultKernel( Index*       w_ind,
                                   W*           w_val,
                                   AccumOp      accum_op,
                                   U            identity,
                                   MulOp        mul_op,
                                   const Index* u_ind,
                                   const U*     u_val,
                                   Index        u_nvals,
                                   const V*     v_val )
  {
    Index row = blockIdx.x * blockDim.x + threadIdx.x;
    for (; row < u_nvals; row += blockDim.x * gridDim.x)
    {
      Index ind = __ldg(u_ind + row);
      U     u_t = __ldg(u_val + row);

      if (u_t != identity)
      {
        V      v_t = __ldg(v_val + ind);
        w_val[row] = mul_op(u_t, v_t);
      }
      else
      {
        w_val[row] = 0;
      }
      w_ind[row] = ind;
      __syncwarp();
    }
  }

  // sparse-dense dense mask vector variant
  // TODO(@ctcyang): add scmp, accum, repl, mask
  //template <bool UseScmp, bool UseAccum, bool UseRepl,
  template <typename W, typename U, typename V, typename M,
            typename AccumOp, typename MulOp>
  __global__ void eWiseMultKernel( Index*       w_ind,
                                   W*           w_val,
                                   const Index* mask_ind,
                                   const M*     mask_val,
                                   Index        mask_nvals,
                                   AccumOp      accum_op,
                                   U            identity,
                                   MulOp        mul_op,
                                   const Index* u_ind,
                                   const U*     u_val,
                                   Index        u_nvals,
                                   const V*     v_val )
  {
    Index row = blockIdx.x * blockDim.x + threadIdx.x;
    for (; row < mask_nvals; row += blockDim.x * gridDim.x)
    {
      Index ind = mask_ind[row];
      M     m_t = mask_val[row];
      W     w_t = 0;

      if (m_t != 0)
      {
        V     v_t = __ldg(v_val + ind);

        if (v_t != identity)
        {
          Index u_found = binarySearch(u_ind, ind, 0, u_nvals);
          if (u_found != -1)
          {
            U u_t = __ldg(u_val + u_found);
            w_t   = mul_op(u_t, v_t);
          }
        }
      }
      w_ind[row] = ind;
      w_val[row] = w_t;
      __syncwarp();
    }
  }

  /*template <bool UseScmp, bool UseAccum, bool UseRepl,
            typename W, typename U, typename V, typename M,
            typename BinaryOpT, typename MulOp>
  __global__ void eWiseMultKernel( Index*           w_ind,
                                   W*               w_val,
                                   const M*         mask_val,
                                   const BinaryOpT* accum_op,
                                   U                identity,
                                   MulOp            mul_op,
                                   const Index*     u_ind,
                                   const U*         u_val,
                                   Index            u_nvals,
                                   const V*         v_val )
  {
    Index row = blockIdx.x*blockDim.x + threadIdx.x;

    for( ; row<u_nvals; row+=gridDim.x*blockDim.x )
    {
      Index ind = __ldg( u_ind+row );
      V     val = __ldg( v_val+ind );
      if( UseScmp^(val!=identity) )
      {
        w_ind[row] = -1;
        w_val[row] = identity;
      }
      else
      {
        w_ind[row] = ind;
        w_val[row] = __ldg(u_val+row);
      }
    }
  }*/

}  // backend
}  // graphblas

#endif  // GRB_BACKEND_APSPIE_KERNELS_EWISEMULT_HPP
