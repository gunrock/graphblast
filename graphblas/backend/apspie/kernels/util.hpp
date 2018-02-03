#ifndef GRB_BACKEND_APSPIE_KERNELS_UTIL_HPP
#define GRB_BACKEND_APSPIE_KERNELS_UTIL_HPP

#include <cuda.h>

namespace graphblas
{
namespace backend
{

  template <typename U>
  __global__ void zeroKernel( U* u_val, Index u_nvals )
  {
    Index row = blockIdx.x*blockDim.x + threadIdx.x;
    for( ; row<u_nvals; row+=gridDim.x*blockDim.x )
    {
      u_val[row] = (U)0;
    }
  }

  template <typename U>
  __global__ void updateFlagKernel( Index*   d_flag,
                                    U        identity,
                                    const U* u_val,
                                    Index    u_nvals )
  {
    Index row = blockIdx.x*blockDim.x + threadIdx.x;

    for( ; row<u_nvals; row+=gridDim.x*blockDim.x )
    {
      U val = __ldg( u_val+row );
      if( val==identity )
        d_flag[row] = 0;
      else
        d_flag[row] = 1;
    }
  }

  //KEYVALUE variant
  template <typename W, typename U>
  __global__ void streamCompactSparseKernel( Index*       w_ind,
                                             W*           w_val,
                                             const Index* d_scan,
                                             U            identity,
                                             const Index* u_ind,
                                             const U*     u_val,
                                             Index        u_nvals )
  {
    Index row = blockIdx.x*blockDim.x + threadIdx.x;

    for( ; row<u_nvals; row+=gridDim.x*blockDim.x )
    {
      Index ind     = __ldg( u_ind +row );
      U     val     = __ldg( u_val +row );
      Index scatter = __ldg( d_scan+row );

      if(val!=identity)
      {
        w_ind[scatter] = ind;
        w_val[scatter] = val;
      }
    }
  }

  // STRUCONLY variant
  template <typename U>
  __global__ void streamCompactSparseKernel( Index*       w_ind,
                                             const Index* d_scan,
                                             U            identity,
                                             const Index* u_ind,
                                             const U*     u_val,
                                             Index        u_nvals )
  {
    Index row = blockIdx.x*blockDim.x + threadIdx.x;

    for( ; row<u_nvals; row+=gridDim.x*blockDim.x )
    {
      Index ind     = __ldg( u_ind +row );
      U val         = __ldg( u_val+row );
      Index scatter = __ldg( d_scan+row );

      if( val==identity )
      {
        w_ind[scatter] = ind;
      }
    }
  }

  // STRUCONLY variant
  template <typename U>
  __global__ void streamCompactDenseKernel( Index*       w_ind,
                                            const Index* d_scan,
                                            Index        identity,
                                            const U*     u_val,
                                            Index        u_nvals )
  {
    Index row = blockIdx.x*blockDim.x + threadIdx.x;

    for( ; row<u_nvals; row+=gridDim.x*blockDim.x )
    {
      Index scatter = __ldg( d_scan+row );
      U val         = __ldg( u_val+row );

      if( val==identity )
      {
        w_ind[scatter] = row;
      }
    }
  }

  // KEYVALUE variant
  template <typename W, typename U>
  __global__ void streamCompactDenseKernel( Index*       w_ind,
                                            W*           w_val,
                                            const Index* d_scan,
                                            U            identity,
                                            const U*     u_val,
                                            Index        u_nvals )
  {
    Index row = blockIdx.x*blockDim.x + threadIdx.x;

    for( ; row<u_nvals; row+=gridDim.x*blockDim.x )
    {
      Index scatter = __ldg( d_scan+row );
      U val         = __ldg( u_val+row );

      if( val!=identity )
      {
        w_ind[scatter] = row;
        w_val[scatter] = val;
      }
    }
  }

  __global__ void indirectScanKernel( Index*       d_temp_nvals,
                                      const Index* A_csrRowPtr,
                                      const Index* u_ind,
                                      Index        u_nvals )
  {
    int gid = blockIdx.x*blockDim.x+threadIdx.x;
    Index length = 0;

    if( gid<u_nvals )
    {
      Index row   = __ldg( u_ind+gid );

      Index start = __ldg( A_csrRowPtr+row   );
      Index end   = __ldg( A_csrRowPtr+row+1 );
      length      = end-start;

      d_temp_nvals[gid] = length;
      //if( tid<10 ) printf("%d: %d = %d - %d\n", length, start, end);
    }
  }

  __global__ void indirectGather( Index*       d_temp_nvals,
                                  const Index* A_csrRowPtr,
                                  const Index* u_ind,
                                  Index        u_nvals )
  {
    int gid = blockIdx.x*blockDim.x+threadIdx.x;

    if( gid<u_nvals )
    {
      Index   row = __ldg( u_ind+gid );
      Index start = __ldg( A_csrRowPtr+row );
      d_temp_nvals[gid] = start;
    }
  }

  // STRUCONLY scatter
  template <typename T>
  __global__ void scatter( T*           w_val,
                           const Index* u_ind,
                           T            val,
                           Index        u_nvals )
  {
    int gid = blockIdx.x*blockDim.x+threadIdx.x;

    if( gid<u_nvals )
    {
      Index ind = __ldg(u_ind+gid);

      w_val[ind] = val;
    }
  }

  // KEYVALUE scatter
  template <typename T>
  __global__ void scatter( T*           w_val,
                           const Index* u_ind,
                           const T*     u_val,
                           Index        u_nvals )
  {
    int gid = blockIdx.x*blockDim.x+threadIdx.x;

    if( gid<u_nvals )
    {
      Index ind = __ldg(u_ind+gid);
      T     val = __ldg(u_val+gid);

      w_val[ind] = val;
    }
  }

}  // backend
}  // graphblas

#endif  // GRB_BACKEND_APSPIE_KERNELS_APPLY_HPP
