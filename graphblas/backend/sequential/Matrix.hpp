#ifndef GRB_BACKEND_SEQUENTIAL_HPP
#define GRB_BACKEND_SEQUENTIAL_HPP

#include <vector>

#include <graphblas/backend/sequential/CooMatrix.hpp>
#include <graphblas/backend/sequential/DenseMatrix.hpp>

namespace graphblas
{
namespace backend
{
  template <typename T>
  class Matrix : public CooMatrix<T> 
  {
    public:
    // Default Constructor, Standard Constructor and Assignment Constructor
    Matrix()
				: CooMatrix<T>() {}
    Matrix( const Index nrows, const Index ncols )
				: CooMatrix<T>( nrows, ncols ) {}
    void operator=( Matrix& rhs ) {}

    // Destructor
    ~Matrix() {};
      
    // C API Methods
    Info build( const std::vector<Index>& row_indices, 
                const std::vector<Index>& col_indices, 
                const std::vector<T>& values, 
                const Index nvals,
                const Matrix& mask,
                const BinaryOp& dup ) 
		{
            CooMatrix<T>::build( row_indices, col_indices, values, nvals, mask, dup );
		}

	Info build( const std::vector<T>& values )
        {
            DenseMatrix<T>::build( values );
        }

    Info nnew( const Index nrows, const Index ncols ) {} // possibly unnecessary in C++
		Info dup( Matrix& C ) {}
    Info clear() {}
    Info nrows( Index nrows__ ) {}
    Info ncols( Index ncols__ ) {}
    Info nvals( Index nvals__ ) {}

    private:
    // Data members that are same for all matrix formats 
    //Index nrows_;
    //Index ncols_;
    //Index nvals_;

  };


} // backend
} // graphblas

#endif  // GRB_BACKEND_SEQUENTIAL_HPP
