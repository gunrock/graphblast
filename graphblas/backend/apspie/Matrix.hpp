#ifndef GRB_BACKEND_APSPIE_HPP
#define GRB_BACKEND_APSPIE_HPP

#include <vector>
//#include <iostream>

#include <graphblas/backend/apspie/SparseMatrix.hpp>
#include <graphblas/backend/apspie/DenseMatrix.hpp>

namespace graphblas
{
namespace backend
{
  template <typename T>
  class Matrix
  {
    public:
    // Default Constructor, Standard Constructor and Assignment Constructor
    Matrix() : nrows_(0), ncols_(0), mat_type_(Unknown) {}
    Matrix( const Index nrows, const Index ncols ) 
				: nrows_(nrows), ncols_(ncols), mat_type_(Unknown)
		{
			// Transfer nrows ncols to Sparse/DenseMatrix data member
		  sparse.nnew( nrows_, ncols_ );
			dense.nnew( nrows_, ncols_ );
		}
    void operator=( Matrix& rhs ) {}

    // Destructor
    ~Matrix() {};

    // C API Methods
    Info build( const std::vector<Index>& row_indices,
                const std::vector<Index>& col_indices,
                const std::vector<T>&     values,
                const Index nvals,
                const Matrix& mask,
                const BinaryOp& dup )
    {
      mat_type_ = Sparse;
      sparse.build( row_indices, col_indices, values, nvals, mask.sparse, dup );
    }

    Info build( const std::vector<Index>& row_indices,
                const std::vector<Index>& col_indices,
                const std::vector<T>&     values,
                const Index nvals )
    {
      mat_type_ = Sparse;
      sparse.build( row_indices, col_indices, values, nvals );
    }

    Info build( const std::vector<T>& values )
    {
			mat_type_ = Dense;
      dense.build( values );
    }

		// Mutators
    Info nnew( const Index nrows, const Index ncols ) {} // possibly unnecessary in C++
		Info set_storage( const Storage mat_type )
		{
      mat_type_ = mat_type;
		}
    Info dup( const Matrix& C ) {}
    Info clear() {}

		// Accessors
		Info print() const
		{
			if( mat_type_ == Sparse ) sparse.print();
			else dense.print();
		}
    Info nrows( Index& nrows ) const 
		{
		  if( mat_type_ == Sparse ) sparse.nrows( nrows );
			else dense.nrows( nrows );
		}
    Info ncols( Index& ncols ) const
		{
			if( mat_type_ == Sparse ) sparse.ncols( ncols );
			else dense.nrows( nrows );
		}
    Info nvals( Index& nvals ) const 
		{
			if( mat_type_ == Sparse ) sparse.nvals( nvals );
			else dense.nvals( nvals );
		}
		Info get_storage( Storage& mat_type ) const
		{
      mat_type = mat_type_;
		}

    private:
    Index nrows_;
    Index ncols_;

		SparseMatrix<T> sparse;
		DenseMatrix<T>  dense;

		// Keeps track of whether matrix is Sparse or Dense
		Storage mat_type_; 
  };

} // backend
} // graphblas

#endif  // GRB_BACKEND_APSPIE_HPP
