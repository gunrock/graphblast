#ifndef GRB_BACKEND_APSPIE_SPMM_HPP
#define GRB_BACKEND_APSPIE_SPMM_HPP

#include <iostream>

#include <cuda.h>
#include <cusparse.h>

#include "graphblas/backend/apspie/SparseMatrix.hpp"
#include "graphblas/backend/apspie/DenseMatrix.hpp"
#include "graphblas/types.hpp"

namespace graphblas
{
namespace backend
{
	template<typename c>
	__global__ void spmm_kernel( const Index A_nrows, const Index B_ncols, 
			const Index A_ncols, const Index A_nvals, 
			const Index* A_csrRowPtr, const Index* A_csrColInd, const c* A_csrVal, 
			const c* B_denseVal, c* C_denseVal );

	// Naive implementation
  template<typename c, typename a, typename b>
	Info spmm( DenseMatrix<c>&        C,
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
    Info err;
    const int T        = 2;
    const int NTHREADS = 512;
    const int NBLOCKS  = T*A_nrows;
    spmm_kernel<<<NBLOCKS,NTHREADS>>>( A_nrows, B_ncols, A_ncols, A_nvals,
      A.d_csrRowPtr, A.d_csrColInd, A.d_csrVal, B.d_denseVal, C.d_denseVal );
    
	}

	template<typename c>
	__global__ void spmm_kernel( const Index A_nrows, const Index B_ncols, 
			const Index A_ncols, const Index A_nvals, 
			const Index* A_csrRowPtr, const Index* A_csrColInd, const c* A_csrVal, 
			const c* B_denseVal, c* C_denseVal )
	{
		const Index idx = blockIdx.x*blockDim.x + threadIdx.x;
    const int   idb = threadIdx.x;
		const int   T   = 2;
    const int   L_c = 4;
		const Index i   = idx/T;
    const int   idp = idb%T;
		const int   blk = 512;

		c sv[L_c];
		__shared__ c sdata[blk/T*L_c];
    if( i<A_nrows ) {
      sv[0] = 0.0; sv[1] = 0.0; sv[2] = 0.0; sv[3] = 0.0;
			const int max = A_csrRowPtr[i+1]-A_csrRowPtr[i];
			for( int j=0; j<max; j++ ) {
        Index ind = T*(j*A_nrows+i)+idp;
				c     val = A_csrVal[ind];
				Index col = A_csrColInd[ind];

				sv[0] += val*B_denseVal[col*A_ncols+0];
				sv[1] += val*B_denseVal[col*A_ncols+1];
				sv[2] += val*B_denseVal[col*A_ncols+2];
				sv[3] += val*B_denseVal[col*A_ncols+3];
			}
			if( idp!=0 ) {
		    sdata[i*L_c+0] = sv[0];
			  sdata[i*L_c+1] = sv[1];
			  sdata[i*L_c+2] = sv[2];
			  sdata[i*L_c+3] = sv[3];
      }
      __syncthreads();

			if( idp==0 ) {
				C_denseVal[0*A_ncols+i] = sdata[i*L_c+0]+sv[0];
			  C_denseVal[1*A_ncols+i] = sdata[i*L_c+1]+sv[1];
			  C_denseVal[2*A_ncols+i] = sdata[i*L_c+1]+sv[2];
			  C_denseVal[3*A_ncols+i] = sdata[i*L_c+1]+sv[3];
			}
		}
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
            std::cout << "SpMM successful!\n";
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
