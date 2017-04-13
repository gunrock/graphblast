#ifndef GRB_MATRIX_HPP
#define GRB_MATRIX_HPP

#include <vector>

#include <graphblas/types.hpp>

// Opaque data members from the right backend
#define __GRB_BACKEND_MATRIX_HEADER <graphblas/backend/__GRB_BACKEND_ROOT/Matrix.hpp>
#include __GRB_BACKEND_MATRIX_HEADER
#undef __GRB_BACKEND_MATRIX_HEADER

namespace graphblas
{
  template <typename T, typename S=Sparse>
  class Matrix
  {
    public:
    // Default Constructor, Standard Constructor and Assignment Constructor
		//   -it's imperative to call constructor using matrix or the constructed object
		//     won't be tied to this outermost layer
    Matrix();
	  Matrix( const Index nrows, const Index ncols ) : matrix( nrows, ncols ) {}
    void operator=( Matrix& rhs );

    // Destructor
    ~Matrix() {};

    // C API Methods
    Info build( const std::vector<Index>& row_indices,
                const std::vector<Index>& col_indices,
                const std::vector<T>& values,
                const Index nvals,
                const Matrix& mask,
                const BinaryOp& dup );

    Info build( const std::vector<Index>& row_indices,
                const std::vector<Index>& col_indices,
                const std::vector<T>& values,
                const Index nvals );

		Info build( const std::vector<T>& values );

    Info nnew( const Index nrows, const Index ncols ); // possibly unnecessary in C++
    Info dup( Matrix& C );
    Info clear();
    Info nrows( const Index nrows__ );
    Info ncols( const Index ncols__ );
    Info nvals( const Index nvals__ );

    private:
    // Data members that are same for all backends
    backend::Matrix<T,S> matrix;
  };

  template <typename T, typename S>
  Info Matrix<T,S>::build( const std::vector<Index>& row_indices,
                         const std::vector<Index>& col_indices,
                         const std::vector<T>& values,
                         const Index nvals,
                         const Matrix& mask,
                         const BinaryOp& dup)
  {
	  return matrix.build( row_indices, col_indices, values, nvals, mask, dup );
  }

  template <typename T, typename S>
  Info Matrix<T,S>::build( const std::vector<Index>& row_indices,
                         const std::vector<Index>& col_indices,
                         const std::vector<T>& values,
                         const Index nvals )
  {
	  return matrix.build( row_indices, col_indices, values, nvals );
  }

	template <typename T, typename S>
	Info Matrix<T,S>::build( const std::vector<T>& values )
	{
		return matrix.build( values );
	}
}

#endif  // GRB_MATRIX_HPP
