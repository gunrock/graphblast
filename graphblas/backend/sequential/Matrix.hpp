#ifndef GRB_BACKEND_SEQUENTIAL_HPP
#define GRB_BACKEND_SEQUENTIAL_HPP

#include <graphblas/backend/sequential/CooMatrix.hpp>

namespace graphblas
{
namespace backend
{
  template <typename T>
  class Matrix : public CooMatrix<T> 
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
                      const T *values, 
                      Index n,
                      const Matrix* mask,
                      const BinaryFunction* accum,
                      const Descriptor* desc );

    Info nnew( Index num_row, Index num_col ); // possibly unnecessary in C++
    Info clear();
    Info nrow( Index *m );
    Info ncol( Index *n );
    Info nnz( Index *s );

    private:
    // Data members that are same for all matrix formats 
    Index num_row_;
    Index num_col_;
    Index num_nnz_;

  };

  template <typename T>
  Matrix<T>::Matrix() : CooMatrix<T>() {};

  template <typename T>
  Matrix<T>::Matrix( Index num_row,
                     Index num_col ) : CooMatrix<T>( num_row, num_col ) {}


} // backend
} // graphblas

#endif  // GRB_BACKEND_SEQUENTIAL_HPP
