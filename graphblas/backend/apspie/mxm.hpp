#ifndef GRB_BACKEND_APSPIE_MXM_HPP
#define GRB_BACKEND_APSPIE_MXM_HPP

#include <iostream>
#include <vector>

#include "graphblas/types.hpp"

#include "graphblas/backend/apspie/Matrix.hpp"
#include "graphblas/backend/apspie/SparseMatrix.hpp"
#include "graphblas/backend/apspie/DenseMatrix.hpp"
#include "graphblas/backend/apspie/spgemm.hpp"

namespace graphblas
{
namespace backend
{
  template <typename T>
  class Matrix;

  template <typename c, typename a, typename b>
  Info mxmTriple( SparseMatrix<c>&       C,
                  const SparseMatrix<c>& D,
                  const SparseMatrix<a>& A_t,
                  const SparseMatrix<b>& B,
                  const SparseMatrix<a>& A );

  template <typename c, typename a, typename b>
  Info mxm( Matrix<c>&       C,
            Matrix<c>&       D,
            const Matrix<a>& A_t,
            const Matrix<b>& B,
            const Matrix<a>& A )
  {
    Storage B_storage, A_storage, At_storage;
    B.getStorage( B_storage );
    A.getStorage( A_storage );
    A_t.getStorage( At_storage );

    Info err;
    if( B_storage == GrB_SPARSE && A_storage == GrB_SPARSE && 
        At_storage == GrB_SPARSE )
    {
      err = mxmTriple( C.sparse_, D.sparse_, 
                       A_t.sparse_, B.sparse_, A.sparse_ );
    }
    return err;
  }

  template <typename c, typename a, typename b>
  Info mxm( Matrix<c>&       C,
            const Matrix<a>& A,
            const Matrix<b>& B )
  {
    Storage B_storage, A_storage;
    A.getStorage( A_storage );
    B.getStorage( B_storage );

    Info err;
    if( B_storage == GrB_SPARSE && A_storage == GrB_SPARSE )
    {
      err = mxm( C.sparse_, A.sparse_, B.sparse_ );
    }
    return err;
  }
  
  // wrapper for switching between my spgemm and cusparse
  template <typename c, typename a, typename b>
  Info mxmTriple( SparseMatrix<c>&       C,
                  SparseMatrix<c>&       D,
                  const SparseMatrix<a>& A_t,
                  const SparseMatrix<b>& B,
                  const SparseMatrix<a>& A )
  {
    Info err;

    err = cusparse_spgemm( D,   B, A );
	  CUDA( cudaDeviceSynchronize() );
    err = cusparse_spgemm( C, A_t, D );
    //cusparse_spgemm2( D, A_t, B );
    //cusparse_spgemm2( C, B, A );

    return err;
  }

  // wrapper for switching between my spgemm and cusparse
  template <typename c, typename a, typename b>
  Info mxm( SparseMatrix<c>&       C,
            const SparseMatrix<a>& A,
            const SparseMatrix<b>& B )
  {
    Info err;

    err = cusparse_spgemm( C, A, B );
    //err = cusparse_spgemm2( C, A, B );

    return err;
  }

}  // backend
}  // graphblas

#endif  // GRB_BACKEND_APSPIE_MXM_HPP
