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
    Matrix( Index numRow, Index numCol );
    void operator=( Matrix& rhs );

    // Destructor
    ~Matrix() {};

    // C API Methods
	Info nnew( Index nrow, Index ncol );
	Info clear();
	Info nrows( Index *m );
	Info ncols( Index *n );
	Info nnz( Index *s );

    private:
    // Data members that are same for all backends
	Index numRows;
	Index numCols;
	Index numNnz;

  };

  template <typename T>  
  Matrix<T>::Matrix( Index numRow, Index numCol ) : numRows(numRow), numCols(numCol)
  {
	backend::Matrix<T>();
  }
}

#endif  // GRB_MATRIX_HPP
