#ifndef GRB_BACKEND_APSPIE_SPMM_HPP
#define GRB_BACKEND_APSPIE_SPMM_HPP

#include <iostream>

#include <cuda.h>
#include <cusparse.h>
#include <helper_math.h>

#include <moderngpu.cuh>

#include "graphblas/backend/apspie/SparseMatrix.hpp"
#include "graphblas/backend/apspie/DenseMatrix.hpp"
#include "graphblas/backend/apspie/kernels/spmm.hpp"
#include "graphblas/types.hpp"
#include "graphblas/util.hpp"

//#define TA     32
//#define TB     32
//#define NT     64

namespace graphblas
{
namespace backend
{
  template<typename c, typename m, typename a, typename b>
  Info spmm( DenseMatrix<c>&        C,
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

    dim3 NT;
    dim3 NB;
    NT.x = NTHREADS;
    NT.y = 1;
    NT.z = 1;
    NB.x = NBLOCKS;
    NB.y = (B_ncols+31)/32;
    NB.z = 1;

    //CUDA( cudaDeviceSetCacheConfig( cudaFuncCachePreferL1 ) );
    if( mode == GrB_FIXEDROW )
      switch( TB ) {
        case 1:
          spmmRowKernel3<c,1,false><<<NB,NT>>>( A_nrows, B_ncols, A_ncols, 
              A_nvals, A.d_csrRowPtr_, A.d_csrColInd_, A.d_csrVal_,
              B.d_denseVal_, C.d_denseVal_ );
          break;
        case 2:
          spmmRowKernel3<c,2,false><<<NB,NT>>>( A_nrows, B_ncols, A_ncols, 
              A_nvals, A.d_csrRowPtr_, A.d_csrColInd_, A.d_csrVal_,
              B.d_denseVal_, C.d_denseVal_ );
          break;
        case 4:
          spmmRowKernel3<c,4,false><<<NB,NT>>>( A_nrows, B_ncols, A_ncols, 
              A_nvals, A.d_csrRowPtr_, A.d_csrColInd_, A.d_csrVal_,
              B.d_denseVal_, C.d_denseVal_ );
          break;
        case 8:
          spmmRowKernel3<c,8,false><<<NB,NT>>>( A_nrows, B_ncols, A_ncols, 
              A_nvals, A.d_csrRowPtr_, A.d_csrColInd_, A.d_csrVal_,
              B.d_denseVal_, C.d_denseVal_ );
          break;
        case 16:
          spmmRowKernel3<c,16,false><<<NB,NT>>>( A_nrows, B_ncols, A_ncols, 
              A_nvals, A.d_csrRowPtr_, A.d_csrColInd_, A.d_csrVal_,
              B.d_denseVal_, C.d_denseVal_ );
          break;
        case 32:
          spmmRowKernel3<c,32,false><<<NB,NT>>>( A_nrows, B_ncols, A_ncols, 
              A_nvals, A.d_csrRowPtr_, A.d_csrColInd_, A.d_csrVal_,
              B.d_denseVal_, C.d_denseVal_ );
          break;
        break;
      }
		else if( mode == GrB_FIXEDROW2 )
			switch( TB ) {
        case 1:
          spmmRowKernel3<c,1,true><<<NB,NT>>>( A_nrows, B_ncols, A_ncols, 
              A_nvals, A.d_csrRowPtr_, A.d_csrColInd_, A.d_csrVal_,
              B.d_denseVal_, C.d_denseVal_ );
          break;
        case 2:
          spmmRowKernel3<c,2,true><<<NB,NT>>>( A_nrows, B_ncols, A_ncols, 
              A_nvals, A.d_csrRowPtr_, A.d_csrColInd_, A.d_csrVal_,
              B.d_denseVal_, C.d_denseVal_ );
          break;
        case 4:
          spmmRowKernel3<c,4,true><<<NB,NT>>>( A_nrows, B_ncols, A_ncols, 
              A_nvals, A.d_csrRowPtr_, A.d_csrColInd_, A.d_csrVal_,
              B.d_denseVal_, C.d_denseVal_ );
          break;
        case 8:
          spmmRowKernel3<c,8,true><<<NB,NT>>>( A_nrows, B_ncols, A_ncols, 
              A_nvals, A.d_csrRowPtr_, A.d_csrColInd_, A.d_csrVal_,
              B.d_denseVal_, C.d_denseVal_ );
          break;
        case 16:
          spmmRowKernel3<c,16,true><<<NB,NT>>>( A_nrows, B_ncols, A_ncols, 
              A_nvals, A.d_csrRowPtr_, A.d_csrColInd_, A.d_csrVal_,
              B.d_denseVal_, C.d_denseVal_ );
          break;
        case 32:
          spmmRowKernel3<c,32,true><<<NB,NT>>>( A_nrows, B_ncols, A_ncols, 
              A_nvals, A.d_csrRowPtr_, A.d_csrColInd_, A.d_csrVal_,
              B.d_denseVal_, C.d_denseVal_ );
          break;
        break;
      }
		else if( mode == GrB_FIXEDROW3 )
			switch( TB ) {
        case 1:
          spmmRowKernel2<c,1,false><<<NB,NT>>>( A_nrows, B_ncols, A_ncols, 
              A_nvals, A.d_csrRowPtr_, A.d_csrColInd_, A.d_csrVal_,
              B.d_denseVal_, C.d_denseVal_ );
          break;
        case 2:
          spmmRowKernel2<c,2,false><<<NB,NT>>>( A_nrows, B_ncols, A_ncols, 
              A_nvals, A.d_csrRowPtr_, A.d_csrColInd_, A.d_csrVal_,
              B.d_denseVal_, C.d_denseVal_ );
          break;
        case 4:
          spmmRowKernel2<c,4,false><<<NB,NT>>>( A_nrows, B_ncols, A_ncols, 
              A_nvals, A.d_csrRowPtr_, A.d_csrColInd_, A.d_csrVal_,
              B.d_denseVal_, C.d_denseVal_ );
          break;
        case 8:
          spmmRowKernel2<c,8,false><<<NB,NT>>>( A_nrows, B_ncols, A_ncols, 
              A_nvals, A.d_csrRowPtr_, A.d_csrColInd_, A.d_csrVal_,
              B.d_denseVal_, C.d_denseVal_ );
          break;
        case 16:
          spmmRowKernel2<c,16,false><<<NB,NT>>>( A_nrows, B_ncols, A_ncols, 
              A_nvals, A.d_csrRowPtr_, A.d_csrColInd_, A.d_csrVal_,
              B.d_denseVal_, C.d_denseVal_ );
          break;
        case 32:
          spmmRowKernel2<c,32,false><<<NB,NT>>>( A_nrows, B_ncols, A_ncols, 
              A_nvals, A.d_csrRowPtr_, A.d_csrColInd_, A.d_csrVal_,
              B.d_denseVal_, C.d_denseVal_ );
          break;
        break;
      }
		else if( mode == GrB_FIXEDROW4 )
			switch( TB ) {
        case 1:
          spmmRowKernel2<c,1,true><<<NB,NT>>>( A_nrows, B_ncols, A_ncols, 
              A_nvals, A.d_csrRowPtr_, A.d_csrColInd_, A.d_csrVal_,
              B.d_denseVal_, C.d_denseVal_ );
          break;
        case 2:
          spmmRowKernel2<c,2,true><<<NB,NT>>>( A_nrows, B_ncols, A_ncols, 
              A_nvals, A.d_csrRowPtr_, A.d_csrColInd_, A.d_csrVal_,
              B.d_denseVal_, C.d_denseVal_ );
          break;
        case 4:
          spmmRowKernel2<c,4,true><<<NB,NT>>>( A_nrows, B_ncols, A_ncols, 
              A_nvals, A.d_csrRowPtr_, A.d_csrColInd_, A.d_csrVal_,
              B.d_denseVal_, C.d_denseVal_ );
          break;
        case 8:
          spmmRowKernel2<c,8,true><<<NB,NT>>>( A_nrows, B_ncols, A_ncols, 
              A_nvals, A.d_csrRowPtr_, A.d_csrColInd_, A.d_csrVal_,
              B.d_denseVal_, C.d_denseVal_ );
          break;
        case 16:
          spmmRowKernel2<c,16,true><<<NB,NT>>>( A_nrows, B_ncols, A_ncols, 
              A_nvals, A.d_csrRowPtr_, A.d_csrColInd_, A.d_csrVal_,
              B.d_denseVal_, C.d_denseVal_ );
          break;
        case 32:
          spmmRowKernel2<c,32,true><<<NB,NT>>>( A_nrows, B_ncols, A_ncols, 
              A_nvals, A.d_csrRowPtr_, A.d_csrColInd_, A.d_csrVal_,
              B.d_denseVal_, C.d_denseVal_ );
          break;
        break;
      }

		CUDA( cudaDeviceSynchronize() );
    //spmmColKernel<<<NBLOCKS,NTHREADS>>>( A_nrows, B_ncols, A_ncols, A_nvals,
    //  A.d_csrRowPtr_, A.d_csrColInd_, A.d_csrVal_, B.d_denseVal_, C.d_denseVal_ );

    C.need_update_ = true;
    return GrB_SUCCESS;
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
  Info cusparse_spmm2( DenseMatrix<c>&        C,
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
    status = cusparseScsrmm2( handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
        A_nrows, B_ncols, A_ncols, A_nvals,
        &alpha, descr, 
        A.d_csrVal_, A.d_csrRowPtr_, A.d_csrColInd_, B.d_denseVal_,
        B_ncols,      // ldb = max(1,k) since op(A) = A
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
  Info mergepath_spmm( DenseMatrix<c>&        C,
                       const Semiring&        op,
                       const SparseMatrix<a>& A,
                       const DenseMatrix<b>&  B,
                       Descriptor&            desc )
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

    //std::cout << "A: " << A_nrows << " " << A_ncols << std::endl;
    //std::cout << "B: " << B_nrows << " " << B_ncols << std::endl;
    //std::cout << "C: " << C_nrows << " " << C_ncols << std::endl;

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

    // Temporarily for testing purposes
    //C.allocate();
    Desc_value tb, nt;
    desc.get( GrB_TB  , tb );
    desc.get( GrB_NT  , nt );

    // Computation
    const int TB       = static_cast<int>(tb);
    const int NT       = static_cast<int>(nt);

    // Computation
    //std::cout << "Success creating mgpu context\n";
    mgpu::SpmmCsrBinary( A.d_csrVal_, A.d_csrColInd_, A_nvals, A.d_csrRowPtr_, 
        A_nrows, B.d_denseVal_, false, C.d_denseVal_, (c) 0, 
        mgpu::multiplies<c>(), mgpu::plus<c>(), B_ncols, desc.d_limits_, 
        desc.d_carryin_, desc.d_carryout_, TB, NT, *desc.d_context_ );
    //std::cout << "Finished SpmmCsrBinary\n";

    C.need_update_ = true;  // Set flag that we need to copy data from GPU
    return GrB_SUCCESS;
  }

}  // backend
}  // graphblas

#endif  // GRB_BACKEND_APSPIE_SPMM_HPP
