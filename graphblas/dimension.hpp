#ifndef GRB_DIMENSION_HPP
#define GRB_DIMENSION_HPP

#include <string>

#include "graphblas/Matrix.hpp"
#include "graphblas/Vector.hpp"

namespace graphblas
{

  template <typename a, typename b>
  Info checkDimRowCol( const Matrix<a>* A, 
                       const Matrix<b>* B, 
                       const std::string& str )
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
  Info checkDimRowRow( const Matrix<a>* A, 
                       const Matrix<b>* B, 
                       const std::string& str )
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
  Info checkDimColCol( const Matrix<a>* A, 
                       const Matrix<b>* B, 
                       const std::string& str )
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

  template <typename a, typename U>
  Info checkDimRowSize( const Matrix<a>*   A, 
                        const Vector<U>*   u, 
                        const std::string& str )
  {
    if( A==NULL || u==NULL ) return GrB_SUCCESS;
    Index A_nrows, u_size;
    CHECK( A->nrows( &A_nrows ) );
    CHECK(  u->size(  &u_size ) );
    if( A_nrows!=u_size )
    {
      std::cout << str << std::endl;
      return GrB_DIMENSION_MISMATCH;
    }
    return GrB_SUCCESS;
  }

  template <typename a, typename U>
  Info checkDimColSize( const Matrix<a>* A, 
                        const Vector<U>* u, 
                        const std::string& str )
  {
    if( A==NULL || u==NULL ) return GrB_SUCCESS;
    Index A_ncols, u_size;
    CHECK( A->ncols( &A_ncols ) );
    CHECK( u->size( &u_size ) );
    if( A_ncols!=u_size )
    {
      std::cout << str << std::endl;
      return GrB_DIMENSION_MISMATCH;
    }
    return GrB_SUCCESS;
  }

  template <typename U, typename W>
  Info checkDimSizeSize( const Vector<U>* u, 
                         const Vector<W>* w, 
                         const std::string& str )
  {
    if( u==NULL || w==NULL ) return GrB_SUCCESS;
    Index u_size, w_size;
    CHECK( u->size( &u_size ) );
    CHECK( w->size( &w_size ) );
    if( u_size!=w_size )
    {
      std::cout << str << std::endl;
      return GrB_DIMENSION_MISMATCH;
    }
    return GrB_SUCCESS;
  }

}  // graphblas

#endif  // GRB_DIMENSION_HPP
