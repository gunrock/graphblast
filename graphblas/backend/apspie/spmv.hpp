#ifndef GRB_BACKEND_APSPIE_SPMV_HPP
#define GRB_BACKEND_APSPIE_SPMV_HPP

#include <iostream>

#include <cuda.h>
#include <cusparse.h>

#include <moderngpu.cuh>

#include "graphblas/backend/apspie/SparseMatrix.hpp"
#include "graphblas/backend/apspie/DenseMatrix.hpp"
#include "graphblas/types.hpp"
#include "graphblas/util.hpp"

//#define TA     32
//#define TB     32
//#define NT     64

namespace graphblas
{
namespace backend
{
  template<typename c, int TB>
  __global__ void spmv_row_kernel( const Index A_nrows, 
      const Index B_ncols, const Index A_ncols, const Index A_nvals,
      const Index* A_csrRowPtr, const Index* A_csrColInd, const c* A_csrVal, 
      const c* B_denseVal, c* C_denseVal );
      //const c* B_denseVal, float4* C_denseVal );

  template<typename c, int TB>
  __global__ void spmv_col_kernel( const Index A_nrows, 
      const Index B_ncols, const Index A_ncols, const Index A_nvals,
      const Index* A_csrRowPtr, const Index* A_csrColInd, const c* A_csrVal, 
      const c* B_denseVal, c* C_denseVal );

  template<typename c, typename m, typename a, typename b>
  Info spmv( DenseMatrix<c>&        C,
             const SparseMatrix<m>& mask,
             const BinaryOp&        accum,
             const Semiring&        op,
             const SparseMatrix<a>& A,
             const DenseMatrix<b>&  B,
             const Descriptor&      desc )
  {
    Index A_nrows, A_ncols, A_nvals;
    Index B_nrows, B_ncols;
    Index C_nrows, C_ncols;

    A.nrows( A_nrows );
    A.ncols( A_ncols );
    A.nvals( A_nvals );
    B.nrows( B_nrows );
    B.ncols( B_ncols );
    C.nrows( C_nrows );
    C.ncols( C_ncols );

    // Dimension compatibility check
    if( (A_ncols != B_nrows) || (C_ncols != B_ncols) || (C_nrows != A_nrows ) )
    {
      std::cout << "Dim mismatch" << std::endl;
      std::cout << A_ncols << " " << B_nrows << std::endl;
      std::cout << C_ncols << " " << B_ncols << std::endl;
      std::cout << C_nrows << " " << A_nrows << std::endl;
      return GrB_DIMENSION_MISMATCH;
    }

    // Domain compatibility check
    // TODO: add domain compatibility check

    // Read descriptor
    Desc_value mode, ta, tb, nt;
    desc.get( GrB_MODE, mode );
    desc.get( GrB_TA  , ta );
    desc.get( GrB_TB  , tb );
    desc.get( GrB_NT  , nt );

    // Computation
    const int T        = static_cast<int>(ta);
    const int TB       = static_cast<int>(tb);
    const int NTHREADS = static_cast<int>(nt);
    const int NBLOCKS  = (T*A_nrows+NTHREADS-1)/NTHREADS;
    //CUDA( cudaDeviceSetCacheConfig( cudaFuncCachePreferL1 ) );
    if( mode == GrB_FIXEDROW )
      switch( TB ) {
        case 1:
          spmv_row_kernel<c,1><<<NBLOCKS,NTHREADS>>>( A_nrows, 
            B_ncols, A_ncols, A_nvals, A.d_csrRowPtr_, A.d_csrColInd_, A.d_csrVal_,
            B.d_denseVal_, C.d_denseVal_ );
          break;
        case 2:
          spmv_row_kernel<c,2><<<NBLOCKS,NTHREADS>>>( A_nrows, 
            B_ncols, A_ncols, A_nvals, A.d_csrRowPtr_, A.d_csrColInd_, A.d_csrVal_,
            B.d_denseVal_, C.d_denseVal_ );
          break;
        case 4:
          spmv_row_kernel<c,4><<<NBLOCKS,NTHREADS>>>( A_nrows, 
            B_ncols, A_ncols, A_nvals, A.d_csrRowPtr_, A.d_csrColInd_, A.d_csrVal_,
            B.d_denseVal_, C.d_denseVal_ );
          break;
        case 8:
          spmv_row_kernel<c,8><<<NBLOCKS,NTHREADS>>>( A_nrows, 
            B_ncols, A_ncols, A_nvals, A.d_csrRowPtr_, A.d_csrColInd_, A.d_csrVal_,
            B.d_denseVal_, C.d_denseVal_ );
          break;
        case 16:
          spmv_row_kernel<c,16><<<NBLOCKS,NTHREADS>>>( A_nrows, 
            B_ncols, A_ncols, A_nvals, A.d_csrRowPtr_, A.d_csrColInd_, A.d_csrVal_,
            B.d_denseVal_, C.d_denseVal_ );
          break;
        case 32:
          spmv_row_kernel<c,32><<<NBLOCKS,NTHREADS>>>( A_nrows, 
            B_ncols, A_ncols, A_nvals, A.d_csrRowPtr_, A.d_csrColInd_, A.d_csrVal_,
            B.d_denseVal_, C.d_denseVal_ );
          break;
      } else switch( TB ) {
        case 1:
          spmv_col_kernel<c,1><<<NBLOCKS,NTHREADS>>>( A_nrows, 
            B_ncols, A_ncols, A_nvals, A.d_csrRowPtr_, A.d_csrColInd_, A.d_csrVal_,
            B.d_denseVal_, C.d_denseVal_ );
          break;
        case 2:
          spmv_col_kernel<c,2><<<NBLOCKS,NTHREADS>>>( A_nrows, 
            B_ncols, A_ncols, A_nvals, A.d_csrRowPtr_, A.d_csrColInd_, A.d_csrVal_,
            B.d_denseVal_, C.d_denseVal_ );
          break;
        case 4:
          spmv_col_kernel<c,4><<<NBLOCKS,NTHREADS>>>( A_nrows, 
            B_ncols, A_ncols, A_nvals, A.d_csrRowPtr_, A.d_csrColInd_, A.d_csrVal_,
            B.d_denseVal_, C.d_denseVal_ );
          break;
        case 8:
          spmv_col_kernel<c,8><<<NBLOCKS,NTHREADS>>>( A_nrows, 
            B_ncols, A_ncols, A_nvals, A.d_csrRowPtr_, A.d_csrColInd_, A.d_csrVal_,
            B.d_denseVal_, C.d_denseVal_ );
          break;
        case 16:
          spmv_col_kernel<c,16><<<NBLOCKS,NTHREADS>>>( A_nrows, 
            B_ncols, A_ncols, A_nvals, A.d_csrRowPtr_, A.d_csrColInd_, A.d_csrVal_,
            B.d_denseVal_, C.d_denseVal_ );
          break;
        case 32:
          spmv_col_kernel<c,32><<<NBLOCKS,NTHREADS>>>( A_nrows, 
            B_ncols, A_ncols, A_nvals, A.d_csrRowPtr_, A.d_csrColInd_, A.d_csrVal_,
            B.d_denseVal_, C.d_denseVal_ );
          break;
      }

    //spmv_col_kernel<<<NBLOCKS,NTHREADS>>>( A_nrows, B_ncols, A_ncols, A_nvals,
    //  A.d_csrRowPtr_, A.d_csrColInd_, A.d_csrVal_, B.d_denseVal_, C.d_denseVal_ );
    C.need_update_ = true;
    return GrB_SUCCESS;
  }

  // Baseline implementation (row major) based on Bell/Garland 2008
  //
  template<typename c, int TB>
  __global__ void spmv_row_kernel( const Index A_nrows, 
      const Index B_ncols, const Index A_ncols, const Index A_nvals,
      const Index* A_csrRowPtr, const Index* A_csrColInd, const c* A_csrVal, 
      const c* B_denseVal, c* C_denseVal )
      //const c* B_denseVal, float4* C_denseVal )
  {
    float vals;
    float raws;

    int thread_id = blockDim.x*blockIdx.x+threadIdx.x; // global thrd idx
    int warp_id   = thread_id>>5;                      // global warp idx
    int lane      = thread_id & (32 - 1);
    int row, slab;

    // one warp per row
    // Note: Must reset this value every slab
    row = warp_id;
    //if( threadIdx.x==0 )
    //  printf("row:%d,slab:%d\n", row, slab);

    if( row < A_nrows ) {
      int row_start = __ldg(A_csrRowPtr+row);
      int row_end   = __ldg(A_csrRowPtr+row+1);

      // compute running sum per thread
      vals = 0.0;

      for( int jj=row_start+lane; jj<row_end; jj+=32 ) {
        int   col = A_csrColInd[jj];
        float val = A_csrVal[jj];

        raws = __ldg(B_denseVal+col);
        vals += val*raws;
        //printf("row:%d,tid:%d,vals_idx:%d\n",row,thread_id,(ii<<2)+slab);
        //printf("row:%d,col:%d,tid:%d,%.0f = %.0f * %.0f\n",row,col,thread_id, vals, val, raws);
      }

      // parallel reduction in register memory
      vals += __shfl_xor(vals, 16);
      vals += __shfl_xor(vals, 8 );
      vals += __shfl_xor(vals, 4 );
      vals += __shfl_xor(vals, 2 );
      vals += __shfl_xor(vals, 1 );

      // first thread writes the result
      if( lane==0 ) {
          C_denseVal[row] = vals;
      }
    }
  } // spmv_col_kernel

  // Baseline implementation (col major) based on Bell/Garland 2008
  //
  template<typename c, int TB>
  __global__ void spmv_col_kernel( const Index A_nrows, 
      const Index B_ncols, const Index A_ncols, const Index A_nvals,
      const Index* A_csrRowPtr, const Index* A_csrColInd, const c* A_csrVal, 
      const c* B_denseVal, c* C_denseVal )
  {
    float vals[TB];

    int thread_id = blockDim.x*blockIdx.x+threadIdx.x; // global thrd idx
    int warp_id   = thread_id>>5;                      // global warp idx
    int lane      = thread_id & (32 - 1);

    // one warp per row
    int row, slab;
    //if( threadIdx.x==0 )
    //  printf("row:%d\n", row);

    for( slab=0; slab<B_ncols; slab+=TB ) {
      row = warp_id;

      if( row < A_nrows ) {
        int row_start = __ldg(A_csrRowPtr+row);
        int row_end   = __ldg(A_csrRowPtr+row+1);

        // compute running sum per thread
        #pragma unroll
        for( int ii=0; ii<TB; ii++ )
          vals[ii] = 0.0;

        for( int jj=row_start+lane; jj<row_end; jj+=32 ) {
          //printf("row:%d,tid:%d,jj:%d,row_start:%d,row_end:%d,slab:%d\n", row, threadIdx.x, jj, row_start, row_end, slab);
          int   col = A_csrColInd[jj];
          float val = A_csrVal[jj];
          #pragma unroll
          for( int ii=0; ii<TB; ii++ )
            vals[ii] += val*__ldg(B_denseVal+col+A_nrows*(ii+slab));
        }
      }

      // parallel reduction in shared memory
      #pragma unroll
      for( int ii=0; ii<TB; ii++ ) {
        vals[ii] += __shfl_xor(vals[ii], 16);
        vals[ii] += __shfl_xor(vals[ii], 8 );
        vals[ii] += __shfl_xor(vals[ii], 4 );
        vals[ii] += __shfl_xor(vals[ii], 2 );
        vals[ii] += __shfl_xor(vals[ii], 1 );
      }

      // first thread writes the result
      if( lane==0 )
        #pragma unroll
        for( int ii=0; ii<TB; ii++ )
          C_denseVal[row+A_nrows*(ii+slab)] = vals[ii];
    }

    // Incomplete slab
    /*row = warp_id;

    if( row < A_nrows ) {
      int row_start = __ldg(A_csrRowPtr+row);
      int row_end   = __ldg(A_csrRowPtr+row+1);

      // compute running sum per thread
      #pragma unroll
      for( int ii=0; ii<TB; ii++ )
        vals[ii] = 0.0;
      for( int jj=row_start+lane; jj<row_end; jj+=32 ) { 
        //printf("row:%d,tid:%d,jj:%d,row_start:%d,row_end:%d,slab:%d\n", row, threadIdx.x, jj, row_start, row_end, slab);
        int   col = A_csrColInd[jj];
        float val = A_csrVal[jj];
        #pragma unroll
        for( int ii=0; ii<TB; ii++ )
          if( ii+slab<B_ncols )
            vals[ii] += val*__ldg(B_denseVal+col+A_nrows*(ii+slab));
      }
    }

    // parallel reduction in shared memory
    for( int offset = 16; offset > 0; offset /= 2 )
      #pragma unroll
      for( int ii=0; ii<TB; ii++ )
        vals[ii] += __shfl_down(vals[ii], offset);

    // first thread writes the result
    if( lane==0 )
      #pragma unroll
      for( int ii=0; ii<TB; ii++ )
        if( ii+slab<B_ncols )
          C_denseVal[row+A_nrows*(ii+slab)] = vals[ii];*/
  }

  /*template<typename c, typename a, typename b>
  Info cusparse_spmv( DenseMatrix<c>&        C,
                      const Semiring&        op,
                      const SparseMatrix<a>& A,
                      const DenseMatrix<b>&  B )
  {
    Index A_nrows, A_ncols, A_nvals;
    Index B_nrows, B_ncols;
    Index C_nrows, C_ncols;

    A.nrows( A_nrows );
    A.ncols( A_ncols );
    A.nvals( A_nvals );
    B.nrows( B_nrows );
    B.ncols( B_ncols );
    C.nrows( C_nrows );
    C.ncols( C_ncols );

    // Dimension compatibility check
    if( (A_ncols != B_nrows) || (C_ncols != B_ncols) || (C_nrows != A_nrows ) )
    {
      std::cout << "Dim mismatch" << std::endl;
      std::cout << A_ncols << " " << B_nrows << std::endl;
      std::cout << C_ncols << " " << B_ncols << std::endl;
      std::cout << C_nrows << " " << A_nrows << std::endl;
      return GrB_DIMENSION_MISMATCH;
    }

    // Domain compatibility check
    // TODO: add domain compatibility check

    // Computation
    cusparseHandle_t handle;
    cusparseCreate( &handle );
    cusparseSetPointerMode( handle, CUSPARSE_POINTER_MODE_HOST );

    cusparseMatDescr_t descr;
    cusparseCreateMatDescr( &descr );

    cusparseSetMatType( descr, CUSPARSE_MATRIX_TYPE_GENERAL );
    cusparseSetMatIndexBase( descr, CUSPARSE_INDEX_BASE_ZERO );
    cusparseStatus_t status;
    float alpha = 1.0;
    float beta  = 0.0;
    status = cusparseScsrmm( handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE, A_nrows, B_ncols, A_ncols, A_nvals,
        &alpha, descr, A.d_csrVal_, A.d_csrRowPtr_, A.d_csrColInd_, B.d_denseVal_,
        A_ncols,      // ldb = max(1,k) since op(A) = A
        &beta, C.d_denseVal_,
        A_nrows );    // ldc = max(1,m) since op(A) = A

    switch( status ) {
        case CUSPARSE_STATUS_SUCCESS:
            //std::cout << "SpMM successful!\n";
            break;
        case CUSPARSE_STATUS_NOT_INITIALIZED:
            std::cout << "Error: Library not initialized.\n";
            break;
        case CUSPARSE_STATUS_INVALID_VALUE:
            std::cout << "Error: Invalid parameters m, n, or nnz.\n";
            break;
        case CUSPARSE_STATUS_EXECUTION_FAILED:
            std::cout << "Error: Failed to launch GPU.\n";
            break;
        case CUSPARSE_STATUS_ALLOC_FAILED:
            std::cout << "Error: Resources could not be allocated.\n";
            break;
        case CUSPARSE_STATUS_ARCH_MISMATCH:
            std::cout << "Error: Device architecture does not support.\n";
            break;
        case CUSPARSE_STATUS_INTERNAL_ERROR:
            std::cout << "Error: An internal operation failed.\n";
            break;
        case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            std::cout << "Error: Matrix type not supported.\n";
    }

    C.need_update_ = true;  // Set flag that we need to copy data from GPU
    return GrB_SUCCESS;
  }

  template<typename c, typename a, typename b>
  Info mergepath_spmv( DenseMatrix<c>&        C,
                       const Semiring&        op,
                       const SparseMatrix<a>& A,
                       const DenseMatrix<b>&  B )
  {
    Index A_nrows, A_ncols, A_nvals;
    Index B_nrows, B_ncols;
    Index C_nrows, C_ncols;

    A.nrows( A_nrows );
    A.ncols( A_ncols );
    A.nvals( A_nvals );
    B.nrows( B_nrows );
    B.ncols( B_ncols );
    C.nrows( C_nrows );
    C.ncols( C_ncols );

    // Dimension compatibility check
    if( (A_ncols != B_nrows) || (C_ncols != B_ncols) || (C_nrows != A_nrows ) )
    {
      std::cout << "Dim mismatch" << std::endl;
      std::cout << A_ncols << " " << B_nrows << std::endl;
      std::cout << C_ncols << " " << B_ncols << std::endl;
      std::cout << C_nrows << " " << A_nrows << std::endl;
      return GrB_DIMENSION_MISMATCH;
    }

    // Domain compatibility check
    // TODO: add domain compatibility check

    // Computation
    mgpu::ContextPtr context = mgpu::CreateCudaDevice(0);
    CUDA( cudaDeviceSynchronize() );
    std::cout << "Success creating mgpu context\n";
    mgpu::SpmmCsrBinary( A.d_csrVal_, A.d_csrColInd_, A_nvals, A.d_csrRowPtr_, 
        A_nrows, B.d_denseVal_, true, C.d_denseVal_, (c) 0, mgpu::multiplies<c>(),
        mgpu::plus<c>(), B_nrows, *context );
    std::cout << "Finished SpmmCsrBinary\n";
    CUDA( cudaDeviceSynchronize() );

    C.need_update_ = true;  // Set flag that we need to copy data from GPU
    return GrB_SUCCESS;
  }*/

}  // backend
}  // graphblas

#endif  // GRB_BACKEND_APSPIE_SPMV_HPP
