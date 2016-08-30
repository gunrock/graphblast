#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <types.hpp>

// Opaque data members from the right backend
//#define __GRB_BACKEND_MATRIX_HEADER "backend/__GRB_BACKEND_ROOT/Matrix.hpp"
//#include __GRB_BACKEND_MATRIX_HEADER
//#undef __GRB_BACKEND_MATRIX_HEADER

namespace graphblas
{
  template <typename T>
  class Matrix
  {
    public:
    // Default Constructor, Standard Constructor and Assignment Constructor
    Matrix();
    Matrix( Index nrows, Index ncols );
    void operator=( Matrix& rhs );

    // Destructor
    ~Matrix();

    // C Methods
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
}

#endif  // MATRIX_HPP
