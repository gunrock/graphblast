#ifndef GRB_MXM_BACKEND_APSPIE_HPP
#define GRB_MXM_BACKEND_APSPIE_HPP

#include <cusparse.h>

#include <graphblas/types.hpp>

namespace graphblas
{
namespace backend
{
	template <typename c, typename m, typename a, typename b,
					 MatrixType c_spar, MatrixType m_spar,
           MatrixType a_spar, MatrixType b_spar>
  Info mxm( Matrix<c,c_spar>&       C,
					  const Matrix<m,m_spar>& mask,
						const BinaryOp&         accum,
						const Semiring&         op,
						const Matrix<a,a_spar>& A,
						const Matrix<b,b_spar>& B,
						const Descriptor&       desc ); 

  template <typename c, typename a, typename b,
           MatrixType c_spar, MatrixType a_spar, MatrixType b_spar>
  Info mxm( Matrix<c,c_spar>&       C,
	  			  const Semiring&         op,
		  			const Matrix<a,a_spar>& A,
			  		const Matrix<b,b_spar>& B )
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
			/*std::cout << A_ncols << " " << B_nrows << std::endl;
			std::cout << C_ncols << " " << B_ncols << std::endl;
			std::cout << C_nrows << " " << A_nrows << std::endl;*/
      return GrB_DIMENSION_MISMATCH;
    }

		// Domain compatibility check
    // TODO: add domain compatibility check

    // Computation
		cusparseHandle_t cusparse_handle;
		cusparseCreate( &cusparse_handle );
		cusparseMatDescr_t cusparse_descr;
		cusparseCreateMatDescr( &cusparse_descr );
		cusparseStatus_t cusparse_status;
		float alpha = 1.0;
		float beta  = 0.0;
		cusparse_status = cusparseScsrmm( cusparse_handle,
			  CUSPARSE_OPERATION_NON_TRANSPOSE, A_nrows, B_ncols, A_ncols, A_nvals,
        &alpha, cusparse_descr, A.d_csrVal, A.d_csrRowPtr, A.d_csrColInd, 
				B.d_denseVal,
				A_ncols,      // ldb = max(1,k) since op(A) = A
				&beta, C.d_denseVal,
			  A_nrows );    // ldc = max(1,m) since op(A) = A

    switch( cusparse_status ) {
        case CUSPARSE_STATUS_SUCCESS:
            std::cout << "nnz count successful!\n";
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
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

		C.need_update = true;  // Set flag that we need to copy data from GPU
		return GrB_SUCCESS;
  }
}  // backend
}  // graphblas

#endif  // GRB_MXM_HPP
