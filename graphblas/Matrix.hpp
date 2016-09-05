#ifndef GRB_MATRIX_HPP
#define GRB_MATRIX_HPP

#include <graphblas/types.hpp>

// Opaque data members from the right backend
#define __GRB_BACKEND_MATRIX_HEADER <graphblas/backend/__GRB_BACKEND_ROOT/Matrix.hpp>
#include __GRB_BACKEND_MATRIX_HEADER
#undef __GRB_BACKEND_MATRIX_HEADER

namespace graphblas
{
  template <typename T>
  class Matrix
  {
    public:
    // Default Constructor, Standard Constructor and Assignment Constructor
    Matrix();
    Matrix( Index num_rows, Index num_cols );
    void operator=( Matrix& rhs );

    // Destructor
    ~Matrix() {};

    // C API Methods
    Info buildMatrix( const Index* row_ids, 
                      const Index* col_ids, 
                      const T* values, 
                      Index n, 
                      const Matrix* mask=NULL, 
                      const BinaryFunction* accum=NULL, 
                      const Descriptor* desc=NULL );

  	Info nnew( Index num_row, Index num_col ); // possibly unnecessary in C++
  	Info clear();
	  Info nrow( Index *m );
	  Info ncol( Index *n );
	  Info nnz( Index *s );

    private:
    // Data members that are same for all backends
    backend::Matrix<T> matrix;
  };

  template <typename T>  
  Matrix<T>::Matrix( Index num_row, 
                     Index num_col )
  {
	  backend::Matrix<T>( num_row, num_col );
  }

  template <typename T>
  Info Matrix<T>::buildMatrix( const Index* row_ids,
                               const Index* col_ids,
                               const T* values,
                               const Index n,
                               const Matrix* mask,
                               const BinaryFunction* accum,
                               const Descriptor* desc )
  {
	  return matrix.buildMatrix( row_ids, col_ids, values, n, mask, accum, desc );
  }
}

#endif  // GRB_MATRIX_HPP
