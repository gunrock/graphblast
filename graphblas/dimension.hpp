#ifndef GRB_DIMENSION_HPP
#define GRB_DIMENSION_HPP

#include <string>

#include "graphblas/Matrix.hpp"

namespace graphblas
{

  template <typename a, typename b>
  Info checkDimRowCol( Matrix<a>* A, Matrix<b>* B, const std::string& str )
  {
    if( A==NULL || B==NULL ) return GrB_SUCCESS;
    Index A_nrows, B_ncols;
    CHECK( A->nrows( &A_nrows ) );
    CHECK( B->ncols( &B_ncols ) );
    if( A_nrows!=B_ncols )
    {
      std::cout << str << std::endl;
      return GrB_DIMENSION_MISMATCH;
    }
    return GrB_SUCCESS;
  }

  template <typename a, typename b>
  Info checkDimRowRow( Matrix<a>* A, Matrix<b>* B, const std::string& str )
  {
    if( A==NULL || B==NULL ) return GrB_SUCCESS;
    Index A_nrows, B_nrows;
    CHECK( A->nrows( &A_nrows ) );
    CHECK( B->nrows( &B_nrows ) );
    if( A_nrows!=B_nrows )
    {
      std::cout << str << std::endl;
      return GrB_DIMENSION_MISMATCH;
    }
    return GrB_SUCCESS;
  }

  template <typename a, typename b>
  Info checkDimColCol( Matrix<a>* A, Matrix<b>* B, const std::string& str )
  {
    if( A==NULL || B==NULL ) return GrB_SUCCESS;
    Index A_ncols, B_ncols;
    CHECK( A->ncols( &A_ncols ) );
    CHECK( B->ncols( &B_ncols ) );
    if( A_ncols!=B_ncols )
    {
      std::cout << str << std::endl;
      return GrB_DIMENSION_MISMATCH;
    }
    return GrB_SUCCESS;
  }

}  // graphblas

#endif  // GRB_DIMENSION_HPP
