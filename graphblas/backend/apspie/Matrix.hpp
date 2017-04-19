#ifndef GRB_BACKEND_APSPIE_MATRIX_HPP
#define GRB_BACKEND_APSPIE_MATRIX_HPP

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
    // Default Constructor, Standard Constructor 
    // Replaces new in C++
    Matrix() : nrows_(0), ncols_(0), mat_type_(Unknown) {}
    Matrix( const Index nrows, const Index ncols );

    // Assignment Constructor
    // TODO: Replaces dup in C++
    void operator=( const Matrix& rhs ) {}

    // Destructor
    // TODO
    ~Matrix() {};

    // C API Methods
		//
		// Mutators
    Info build( const std::vector<Index>& row_indices,
                const std::vector<Index>& col_indices,
                const std::vector<T>&     values,
                const Index nvals,
                const Matrix& mask,
                const BinaryOp& dup );
    Info build( const std::vector<Index>& row_indices,
                const std::vector<Index>& col_indices,
                const std::vector<T>&     values,
                const Index nvals );
    Info build( const std::vector<T>& values );
		// Private method for setting storage type of matrix
		Info set_storage( const Storage mat_type );
    Info clear(); 

		// Accessors
    Info extractTuples( std::vector<Index>& row_indices,
                        std::vector<Index>& col_indices,
                        std::vector<T>&     values ) const;
		Info print() const;
    Info nrows( Index& nrows ) const;
    Info ncols( Index& ncols ) const;
    Info nvals( Index& nvals ) const; 
		Info get_storage( Storage& mat_type ) const;

    private:
    Index nrows_;
    Index ncols_;

		SparseMatrix<T> sparse;
		DenseMatrix<T>  dense;

		// Keeps track of whether matrix is Sparse or Dense
		Storage mat_type_;

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
  Matrix<T>::Matrix( const Index nrows, const Index ncols )
			: nrows_(nrows), ncols_(ncols), mat_type_(Unknown)
  {
		// Transfer nrows ncols to Sparse/DenseMatrix data member
		sparse.nnew( nrows_, ncols_ );
	  dense.nnew( nrows_, ncols_ );
	}

  template <typename T>
  Info Matrix<T>::build( const std::vector<Index>& row_indices,
                         const std::vector<Index>& col_indices,
                         const std::vector<T>&     values,
                         const Index nvals,
                         const Matrix& mask,
                         const BinaryOp& dup )
  {
    mat_type_ = Sparse;
    return sparse.build( row_indices, col_indices, values, nvals, mask.sparse, dup );
  }

  template <typename T>
  Info Matrix<T>::build( const std::vector<Index>& row_indices,
                         const std::vector<Index>& col_indices,
                         const std::vector<T>&     values,
                         const Index nvals )
  {
    mat_type_ = Sparse;
    return sparse.build( row_indices, col_indices, values, nvals );
  }

  template <typename T>
  Info Matrix<T>::build( const std::vector<T>& values )
  {
		mat_type_ = Dense;
    return dense.build( values );
  }

  template <typename T>
  Info Matrix<T>::extractTuples( std::vector<Index>& row_indices,
                                 std::vector<Index>& col_indices,
                                 std::vector<T>&     values ) const
  {
    if( mat_type_ == Sparse ) 
      return sparse.extractTuples( row_indices, col_indices, values );
    else if( mat_type_ == Dense ) 
      return dense.extractTuples( row_indices, col_indices, values );
    else
      return GrB_UNINITIALIZED_OBJECT;
  }

  // Private method that sets mat_type, clears and allocates
  template <typename T>
	Info Matrix<T>::set_storage( const Storage mat_type )
	{
    Info err;
    mat_type_ = mat_type;
    if( mat_type_ == Sparse ) {
      err = sparse.clear();
      err = sparse.allocate();
    } else {
      err = dense.clear();
      err = dense.allocate();
    }
    return err;
  }

  template <typename T>
  Info Matrix<T>::clear() 
  {
    Info err;
    mat_type_ = Unknown;
    err = sparse.clear();
    err = dense.clear();
    return err;
  }

  template <typename T>
	Info Matrix<T>::print() const
	{
		if( mat_type_ == Sparse ) return sparse.print();
	  else if( mat_type_ == Dense ) return dense.print();
    else return GrB_UNINITIALIZED_OBJECT;
	}

  template <typename T>
  Info Matrix<T>::nrows( Index& nrows ) const 
	{
		if( mat_type_ == Sparse ) return sparse.nrows( nrows );
	  else if( mat_type_ == Dense ) return dense.nrows( nrows );
    else return GrB_UNINITIALIZED_OBJECT;
	}

  template <typename T>
  Info Matrix<T>::ncols( Index& ncols ) const
	{
		if( mat_type_ == Sparse ) return sparse.ncols( ncols );
	  else if( mat_type_ == Dense ) return dense.ncols( ncols );
    else return GrB_UNINITIALIZED_OBJECT;
	}

  template <typename T>
  Info Matrix<T>::nvals( Index& nvals ) const 
	{
		if( mat_type_ == Sparse ) return sparse.nvals( nvals );
	  else if( mat_type_ == Dense ) return dense.nvals( nvals );
    else return GrB_UNINITIALIZED_OBJECT;
	}

  template <typename T>
	Info Matrix<T>::get_storage( Storage& mat_type ) const
	{
    mat_type = mat_type_;
	}
} // backend
} // graphblas

#endif  // GRB_BACKEND_APSPIE_HPP
