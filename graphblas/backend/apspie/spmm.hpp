#ifndef GRB_BACKEND_APSPIE_SPMM_HPP
#define GRB_BACKEND_APSPIE_SPMM_HPP

#include <iostream>

#include <cuda.h>
#include <cusparse.h>

#include "graphblas/backend/apspie/SparseMatrix.hpp"
#include "graphblas/backend/apspie/DenseMatrix.hpp"
#include "graphblas/types.hpp"

//#define TA     32
//#define TB     32
//#define NT     64

namespace graphblas
{
namespace backend
{
	template<typename c, int TB>
	__global__ void spmv_csr_vector_row_kernel( const Index A_nrows, 
			const Index B_ncols, const Index A_ncols, const Index A_nvals,
			const Index* A_csrRowPtr, const Index* A_csrColInd, const c* A_csrVal, 
			const c* B_denseVal, c* C_denseVal );

	template<typename c, int TB>
	__global__ void spmv_csr_vector_col_kernel( const Index A_nrows, 
			const Index B_ncols, const Index A_ncols, const Index A_nvals,
			const Index* A_csrRowPtr, const Index* A_csrColInd, const c* A_csrVal, 
			const c* B_denseVal, c* C_denseVal );

  template<typename c, typename a, typename b>
	Info spmm( DenseMatrix<c>&        C,
             const Semiring&        op,
             const SparseMatrix<a>& A,
             const DenseMatrix<b>&  B,
				     const int TA,
				     const int TB,
				     const int NT,
				     const bool ROW_MAJOR	)
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
    const int T        = TA;
    const int NTHREADS = NT;
    const int NBLOCKS  = (T*A_nrows+NTHREADS-1)/NTHREADS;
    if( ROW_MAJOR )
			switch( TB ) {
				case 1:
          spmv_csr_vector_row_kernel<c,1><<<NBLOCKS,NTHREADS>>>( A_nrows, 
				    B_ncols, A_ncols, A_nvals, A.d_csrRowPtr, A.d_csrColInd, A.d_csrVal,
				    B.d_denseVal, C.d_denseVal );
					break;
				case 2:
          spmv_csr_vector_row_kernel<c,2><<<NBLOCKS,NTHREADS>>>( A_nrows, 
				    B_ncols, A_ncols, A_nvals, A.d_csrRowPtr, A.d_csrColInd, A.d_csrVal,
				    B.d_denseVal, C.d_denseVal );
					break;
				case 4:
          spmv_csr_vector_row_kernel<c,4><<<NBLOCKS,NTHREADS>>>( A_nrows, 
				    B_ncols, A_ncols, A_nvals, A.d_csrRowPtr, A.d_csrColInd, A.d_csrVal,
				    B.d_denseVal, C.d_denseVal );
					break;
				case 8:
          spmv_csr_vector_row_kernel<c,8><<<NBLOCKS,NTHREADS>>>( A_nrows, 
				    B_ncols, A_ncols, A_nvals, A.d_csrRowPtr, A.d_csrColInd, A.d_csrVal,
				    B.d_denseVal, C.d_denseVal );
					break;
				case 16:
          spmv_csr_vector_row_kernel<c,16><<<NBLOCKS,NTHREADS>>>( A_nrows, 
				    B_ncols, A_ncols, A_nvals, A.d_csrRowPtr, A.d_csrColInd, A.d_csrVal,
				    B.d_denseVal, C.d_denseVal );
					break;
				case 32:
          spmv_csr_vector_row_kernel<c,32><<<NBLOCKS,NTHREADS>>>( A_nrows, 
				    B_ncols, A_ncols, A_nvals, A.d_csrRowPtr, A.d_csrColInd, A.d_csrVal,
				    B.d_denseVal, C.d_denseVal );
					break;
			}
		else
			switch( TB ) {
				case 1:
          spmv_csr_vector_col_kernel<c,1><<<NBLOCKS,NTHREADS>>>( A_nrows, 
				    B_ncols, A_ncols, A_nvals, A.d_csrRowPtr, A.d_csrColInd, A.d_csrVal,
				    B.d_denseVal, C.d_denseVal );
					break;
				case 2:
          spmv_csr_vector_col_kernel<c,2><<<NBLOCKS,NTHREADS>>>( A_nrows, 
				    B_ncols, A_ncols, A_nvals, A.d_csrRowPtr, A.d_csrColInd, A.d_csrVal,
				    B.d_denseVal, C.d_denseVal );
					break;
				case 4:
          spmv_csr_vector_col_kernel<c,4><<<NBLOCKS,NTHREADS>>>( A_nrows, 
				    B_ncols, A_ncols, A_nvals, A.d_csrRowPtr, A.d_csrColInd, A.d_csrVal,
				    B.d_denseVal, C.d_denseVal );
					break;
				case 8:
          spmv_csr_vector_col_kernel<c,8><<<NBLOCKS,NTHREADS>>>( A_nrows, 
				    B_ncols, A_ncols, A_nvals, A.d_csrRowPtr, A.d_csrColInd, A.d_csrVal,
				    B.d_denseVal, C.d_denseVal );
					break;
				case 16:
          spmv_csr_vector_col_kernel<c,16><<<NBLOCKS,NTHREADS>>>( A_nrows, 
				    B_ncols, A_ncols, A_nvals, A.d_csrRowPtr, A.d_csrColInd, A.d_csrVal,
				    B.d_denseVal, C.d_denseVal );
					break;
				case 32:
          spmv_csr_vector_col_kernel<c,32><<<NBLOCKS,NTHREADS>>>( A_nrows, 
				    B_ncols, A_ncols, A_nvals, A.d_csrRowPtr, A.d_csrColInd, A.d_csrVal,
				    B.d_denseVal, C.d_denseVal );
          break;
			}

    //spmm_col_kernel<<<NBLOCKS,NTHREADS>>>( A_nrows, B_ncols, A_ncols, A_nvals,
    //  A.d_csrRowPtr, A.d_csrColInd, A.d_csrVal, B.d_denseVal, C.d_denseVal );

		C.need_update = true;
		return GrB_SUCCESS;
	}

	// Baseline implementation (row major) based on Bell/Garland 2008
	//
	template<typename c, int TB>
	__global__ void spmv_csr_vector_row_kernel( const Index A_nrows, 
			const Index B_ncols, const Index A_ncols, const Index A_nvals,
			const Index* A_csrRowPtr, const Index* A_csrColInd, const c* A_csrVal, 
			const c* B_denseVal, c* C_denseVal )
	{
		float vals[TB];

		int thread_id = blockDim.x*blockIdx.x+threadIdx.x; // global thrd idx
		int warp_id   = thread_id>>5;                      // global warp idx
		int lane      = thread_id & (32 - 1);
    int row, slab;

	  for( slab=0; slab<B_ncols-TB+1; slab+=TB ) {
		//for( slab=0; slab<min(B_ncols-TB+1,100*TB); slab+=TB )
    //const int slab = 0;

		// one warp per row
		// Note: Must reset this value every slab
      row = warp_id;
		  //if( threadIdx.x==0 )
      //  printf("row:%d,slab:%d\n", row, slab);

		  if( row < A_nrows ) {
        int row_start = __ldg(A_csrRowPtr+row);
			  int row_end   = __ldg(A_csrRowPtr+row+1);

			  // compute running sum per thread
        #pragma unroll
			  for( int ii=0; ii<TB; ii++ )
			    vals[ii] = 0.0;

			  for( int jj=row_start+lane; jj<row_end; jj+=32 ) {
				  //printf("row:%d,tid:%d,jj:%d,row_start:%d,row_end:%d\n", row, threadIdx.x, jj, row_start, row_end);
					//int   col = __ldg(A_csrColInd+jj);
					//float val = __ldg(A_csrVal+jj);
					int   col = A_csrColInd[jj];
					float val = A_csrVal[jj];
				  #pragma unroll
				  for( int ii=0; ii<TB; ii++ )
					  //printf("row:%d,tid:%d,vals_idx:%d\n",row,thread_id,ii+slab);
					  vals[ii] += val*__ldg(B_denseVal+col*B_ncols+ii+slab);
        }

			  // parallel reduction in register memory
				for( int offset = 16; offset > 0; offset /= 2 )
          #pragma unroll
					for( int ii=0; ii<TB; ii++ )
					  vals[ii] += __shfl_down(vals[ii], offset);

			  // first thread writes the result
			  if( lane==0 )
				  #pragma unroll
				  for( int ii=0; ii<TB; ii++ )
				    C_denseVal[row*B_ncols+ii+slab] = vals[ii];
		  }
		} // slab

  // Not unrolled last slab iteration
  //

		// one warp per row
		// Note: Must reset this value every slab
      row = warp_id;
		  //if( threadIdx.x==0 )
      //  printf("row:%d,slab:%d\n", row, slab);

		  if( row < A_nrows ) {
        int row_start = __ldg(A_csrRowPtr+row);
			  int row_end   = __ldg(A_csrRowPtr+row+1);

			  // compute running sum per thread
        #pragma unroll
			  for( int ii=0; ii<TB; ii++ )
			    vals[ii] = 0.0;

			  for( int jj=row_start+lane; jj<row_end; jj+=32 ) {
				  //printf("row:%d,tid:%d,jj:%d,row_start:%d,row_end:%d\n", row, threadIdx.x, jj, row_start, row_end);
					//int   col = __ldg(A_csrColInd+jj);
					//float val = __ldg(A_csrVal+jj);
					int   col = A_csrColInd[jj];
					float val = A_csrVal[jj];
          #pragma unroll
				  for( int ii=0; ii<TB; ii++ )
						if( ii+slab<B_ncols )
					    vals[ii] += val*__ldg(B_denseVal+col*B_ncols+ii+slab);
					    //vals[ii] += val*B_denseVal[col*B_ncols+ii+slab];
        }

			  // parallel reduction in register memory
				// TODO: need to accumulate to another variable (not vals[ii])
				for( int offset = 16; offset > 0; offset /= 2 )
          #pragma unroll
					for( int ii=0; ii<TB; ii++ )
					  vals[ii] += __shfl_down(vals[ii], offset);

			  // first thread writes the result
			  if( lane==0 )
					#pragma unroll
				  for( int ii=0; ii<TB; ii++ )
						if( ii+slab<B_ncols )
				    //printf("row:%d,col:%d,val:%f\n", row, ii, vals[ii]);
				      C_denseVal[row*B_ncols+ii+slab] = vals[ii];
		// incomplete slab
	    }
	}

	// Baseline implementation (col major) based on Bell/Garland 2008
	//
	template<typename c, int TB>
	__global__ void spmv_csr_vector_col_kernel( const Index A_nrows, 
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

    for( slab=0; slab<B_ncols-TB+1; slab+=TB ) {
      row = warp_id;

		  if( row < A_nrows ) {
        int row_start = A_csrRowPtr[row];
			  int row_end   = A_csrRowPtr[row+1];

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
					  vals[ii] += val*B_denseVal[col+A_nrows*(ii+slab)];
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
				  C_denseVal[row+A_nrows*(ii+slab)] = vals[ii];
		}

		// Incomplete slab
    row = warp_id;

		if( row < A_nrows ) {
      int row_start = A_csrRowPtr[row];
			int row_end   = A_csrRowPtr[row+1];

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
					  vals[ii] += val*B_denseVal[col+A_nrows*(ii+slab)];
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
				  C_denseVal[row+A_nrows*(ii+slab)] = vals[ii];
	}

  template<typename c, typename a, typename b>
  Info cusparse_spmm( DenseMatrix<c>&        C,
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

    cusparseSetMatType( descr, CUSPARSE_MATRIX_TYPE_GENERAL );
    cusparseSetMatIndexBase( descr, CUSPARSE_INDEX_BASE_ZERO );
    cusparseStatus_t status;
    float alpha = 1.0;
    float beta  = 0.0;
    status = cusparseScsrmm( handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE, A_nrows, B_ncols, A_ncols, A_nvals,
        &alpha, descr, A.d_csrVal, A.d_csrRowPtr, A.d_csrColInd, B.d_denseVal,
        A_ncols,      // ldb = max(1,k) since op(A) = A
        &beta, C.d_denseVal,
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

    C.need_update = true;  // Set flag that we need to copy data from GPU
    return GrB_SUCCESS;
  }
}  // backend
}  // graphblas

#endif  // GRB_BACKEND_APSPIE_SPMM_HPP
