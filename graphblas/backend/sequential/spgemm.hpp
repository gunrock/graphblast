#ifndef GRB_BACKEND_SEQUENTIAL_SPGEMM_HPP
#define GRB_BACKEND_SEQUENTIAL_SPGEMM_HPP

#include <iostream>

#include <mkl.h>

#include "graphblas/backend/sequential/SparseMatrix.hpp"
#include "graphblas/types.hpp"

//#define TA     32
//#define TB     32
//#define NT     64

namespace graphblas
{
namespace backend
{

  template<typename c, typename a, typename b>
  Info mkl_spgemm( SparseMatrix<c>&       C,
                   const Semiring&        op,
                   const SparseMatrix<a>& A,
                   const SparseMatrix<b>& B )
  {
    Index A_nrows, A_ncols, A_nvals;
    Index B_nrows, B_ncols, B_nvals;
    Index C_nrows, C_ncols, C_nvals;

    A.nrows( A_nrows );
    A.ncols( A_ncols );
    A.nvals( A_nvals );
    B.nrows( B_nrows );
    B.ncols( B_ncols );
    B.nvals( B_nvals );
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

    // SpGEMM Analyze
    int request = 1;
    int sort    = 7;
    int info    = 0;
    if( C.csrRowPtr==NULL )
        C.csrRowPtr = (Index*) malloc( (A_nrows+1)*sizeof(Index) );

    // Analyze
    // Note: one-based indexing
    mkl_scsrmultcsr((char*)"N", &request, &sort,
                    &A_nrows, &A_ncols, &B_ncols,
                    A.csrVal, A.csrColInd, A.csrRowPtr,
                    B.csrVal, B.csrColInd, B.csrRowPtr,
                    NULL, NULL, C.csrRowPtr,
                    NULL, &info);
    if( info!=0 ) {
      std::cout << "Error: code " << info << "\n";
    }

    C_nvals = C.csrRowPtr[A_nrows]-1;
    if( C_nvals >= C.nvals_ ) {
        free(C.csrColInd);
        free(C.csrVal);
        C.csrColInd = (Index*) malloc (C_nvals*sizeof(Index));
        C.csrVal    = (c*    ) malloc (C_nvals*sizeof(c    ));
    }

    // SpGEMM Compute
    request = 2;
    sort    = 8; // Sort output rows in C

    // Compute
    mkl_scsrmultcsr((char*)"N", &request, &sort,
                    &A_nrows, &A_ncols, &B_ncols,
                    A.csrVal, A.csrColInd, A.csrRowPtr,
                    B.csrVal, B.csrColInd, B.csrRowPtr,
                    C.csrVal, C.csrColInd, C.csrRowPtr,
                    NULL, &info);
    if( info!=0 ) {
      std::cout << "Error: code " << info << "\n";
    }

    C.nvals_ = C_nvals;     // Update nnz count for C
    return GrB_SUCCESS;
  }

  template<typename c, typename a, typename b>
  Info mkl_spgemm_analyze( SparseMatrix<c>&       C,
                           const Semiring&        op,
                           const SparseMatrix<a>& A,
                           const SparseMatrix<b>& B )
  {
  }

  template<typename c, typename a, typename b>
  Info mkl_spgemm_compute( SparseMatrix<c>&       C,
                           const Semiring&        op,
                           const SparseMatrix<a>& A,
                           const SparseMatrix<b>& B )
  {
    Index A_nrows, A_ncols, A_nvals;
    Index B_nrows, B_ncols, B_nvals;
    Index C_nrows, C_ncols;

    A.nrows( A_nrows );
    A.ncols( A_ncols );
    A.nvals( A_nvals );
    B.nrows( B_nrows );
    B.ncols( B_ncols );
    B.nvals( B_nvals );
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

    // SpGEMM Compute
    int request = 2;
    int sort    = 8; // Sort output rows in C
    int info    = 0;

    // Compute
    mkl_scsrmultcsr((char*)"N", &request, &sort,
                    &A_nrows, &A_ncols, &B_ncols,
                    A.csrVal, A.csrColInd, A.csrRowPtr,
                    B.csrVal, B.csrColInd, B.csrRowPtr,
                    C.csrVal, C.csrColInd, C.csrRowPtr,
                    NULL, &info);
    if( info!=0 ) {
      std::cout << "Error: code " << info << "\n";
    }

    C.nvals_ = C.csrRowPtr[A_nrows]-1;     // Update nnz count for C
  }
}  // backend
}  // graphblas

#endif  // GRB_BACKEND_SEQUENTIAL_SPGEMM_HPP
