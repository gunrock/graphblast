#ifndef GRB_BACKEND_CUDA_KERNELS_ASSIGNSPARSE_HPP
#define GRB_BACKEND_CUDA_KERNELS_ASSIGNSPARSE_HPP

namespace graphblas
{
namespace backend
{

  // This is the key-value dense mask variant of assignSparse
  // Iterating along u_ind and u_val is better
  template <bool UseScmp, bool UseMask, bool UseAll,
            typename U, typename M, typename BinaryOpT>
  __global__ void assignSparseKernel( Index*    u_ind,
                                      U*        u_val,
                                      Index     u_nvals,
                                      const M*  mask_val,
                                      BinaryOpT accum_op,
                                      U         val,
                                      Index*    indices,
                                      Index     nindices )
  {
    Index row = blockIdx.x*blockDim.x + threadIdx.x;

    if( UseAll )
    {
      for( ; row<u_nvals; row+=gridDim.x*blockDim.x )
      {
        Index ind = u_ind[row];
        if( UseMask )
        {
          M m_val = mask_val[ind];

          // val is assigned if either:
          // 1) use structural complement and m_val is zero
          // 2) do not use structural complement and m_val is not zero
          if ((UseScmp && m_val == 0) || (!UseScmp && m_val != 0))
          {
            //printf("Success: %d\n", row);
            u_val[row] = val;
          }
        }
        else
        {
          u_val[row] = val;
        }
      }
    }
    else
    {
      if( row==0 )
      {
        printf( "Selective Indices SpVec Assign Constant Kernel\n" );
        printf( "Error: Feature not implemented yet!\n" );
      }
    }
  }

  // This is the struconly dense mask variant of assignSparse
  template <bool UseScmp, bool UseMask, bool UseAll,
            typename M, typename BinaryOpT>
  __global__ void assignSparseKernel( Index*    u_ind,
                                      Index     u_nvals,
                                      const M*  mask_val,
                                      BinaryOpT accum_op,
                                      Index     val,
                                      Index*    indices,
                                      Index     nindices )
  {
    Index row = blockIdx.x*blockDim.x + threadIdx.x;

    if( UseAll )
    {
      for( ; row<u_nvals; row+=gridDim.x*blockDim.x )
      {
        Index ind = u_ind[row];
        if( UseMask )
        {
          M m_val = mask_val[ind];

          // val is assigned if either:
          // 1) use structural complement and m_val is zero
          // 2) do not use structural complement and m_val is not zero
          if ((UseScmp && m_val == 0) || (!UseScmp && m_val != 0))
          {
            //printf("Success: %d\n", row);
            u_ind[row] = val;
          }
        }
        else
        {
          u_ind[row] = val;
        }
      }
    }
    else
    {
      if( row==0 )
      {
        printf( "Selective Indices SpVec Assign Constant Kernel\n" );
        printf( "Error: Feature not implemented yet!\n" );
      }
    }
  }

}  // backend
}  // graphblas

#endif  // GRB_BACKEND_CUDA_KERNELS_ASSIGNSPARSE_HPP
