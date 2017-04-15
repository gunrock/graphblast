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
  template <typename T>
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
		//
		// TODO: mask version
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

    Info print();

    Info nnew( const Index nrows, const Index ncols ); // possibly unnecessary in C++
    Info dup( Matrix& C );
    Info clear();
    Info nrows( Index& nrows ) const;
    Info ncols( Index& ncols ) const;
    Info nvals( Index& nvals ) const;

    private:
    // Data members that are same for all backends
    backend::Matrix<T> matrix;

		template <typename c, typename m, typename a, typename b>
    friend Info mxm( Matrix<c>&        C,
                     const Matrix<m>&  mask,
                     const BinaryOp&   accum,
                     const Semiring&   op,
                     const Matrix<a>&  A,
                     const Matrix<b>&  B,
                     const Descriptor& desc );

		template <typename c, typename a, typename b>
    friend Info mxm( Matrix<c>&       C,
                     const Semiring&  op,
                     const Matrix<a>& A,
                     const Matrix<b>& B );
	};

  template <typename T>
  Info Matrix<T>::build( const std::vector<Index>& row_indices,
                         const std::vector<Index>& col_indices,
                         const std::vector<T>& values,
                         const Index nvals,
                         const Matrix& mask,
                         const BinaryOp& dup)
  {
	  return matrix.build( row_indices, col_indices, values, nvals, mask, dup );
  }

  template <typename T>
  Info Matrix<T>::build( const std::vector<Index>& row_indices,
                         const std::vector<Index>& col_indices,
                         const std::vector<T>& values,
                         const Index nvals )
  {
	  return matrix.build( row_indices, col_indices, values, nvals );
  }

	template <typename T>
	Info Matrix<T>::build( const std::vector<T>& values )
	{
		return matrix.build( values );
	}

  template <typename T>
  Info Matrix<T>::print()
  {
    return matrix.print();
  }

	template <typename T>
  Info Matrix<T>::nrows( Index& nrows ) const
	{
    return matrix.nrows( nrows );
	}

	template <typename T>
  Info Matrix<T>::ncols( Index& ncols ) const
	{
    return matrix.ncols( ncols );
	}

	template <typename T>
  Info Matrix<T>::nvals( Index& nvals ) const
	{
    return matrix.nvals( nvals );
	}
}  // graphblas

#endif  // GRB_MATRIX_HPP
