#ifndef GRB_BACKEND_APSPIE_MATRIX_HPP
#define GRB_BACKEND_APSPIE_MATRIX_HPP

#include <vector>
#include <iostream>

//#include "graphblas/backend/apspie/Descriptor.hpp"
#include "graphblas/backend/apspie/SparseMatrix.hpp"
#include "graphblas/backend/apspie/DenseMatrix.hpp"

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
    Matrix() : nrows_(0), ncols_(0), mat_type_(GrB_UNKNOWN) {}
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
    Info setStorage( const Storage mat_type );
    Info clear(); 
    Info print();

    // Accessors
    
    // Needs copy to cpu vector, so is not const
    Info extractTuples( std::vector<Index>& row_indices,
                        std::vector<Index>& col_indices,
                        std::vector<T>&     values );
    Info extractTuples( std::vector<T>& values );
    Info nrows( Index& nrows ) const;
    Info ncols( Index& ncols ) const;
    Info nvals( Index& nvals ) const; 
    Info getStorage( Storage& mat_type ) const;
    Info printStats() const;

    private:
    Index nrows_;
    Index ncols_;

    SparseMatrix<T> sparse_;
    DenseMatrix<T>  dense_;

    // Keeps track of whether matrix is GrB_SPARSE, GrB_DENSE or GrB_UNKNOWN
    Storage mat_type_;

    template <typename c, typename m, typename a, typename b>
    friend Info mxm( Matrix<c>&        C,
                     const Matrix<m>&  mask,
                     const BinaryOp&   accum,
                     const Semiring&   op,
                     const Matrix<a>&  A,
                     const Matrix<b>&  B,
                     Descriptor&       desc );

    // In <graphblas/backend/apspie/mxm.hpp>
    template <typename c, typename a, typename b>
    friend Info mxm( Matrix<c>&       C,
                     const Semiring&  op,
                     const Matrix<a>& A,
                     const Matrix<b>& B );

    // In <graphblas/backend/apspie/mxm.hpp>
    // For testing
    template <typename c, typename a, typename b>
    friend Info mxm( Matrix<c>&       C,
                     const Semiring&  op,
                     const Matrix<a>& A,
                     const Matrix<b>& B,
                     const int TA,
                     const int TB,
                     const int NT,
                     const bool ROW_MAJOR );

    template <typename c, typename m, typename a, typename b>
    friend Info mxv( Matrix<c>&        C,
                     const Matrix<m>&  mask,
                     const BinaryOp&   accum,
                     const Semiring&   op,
                     const Matrix<a>&  A,
                     const Matrix<b>&  B,
                     Descriptor&       desc );

    template <typename c, typename a, typename b>
    friend Info mxv( Matrix<c>&       C,
                     const Semiring&  op,
                     const Matrix<a>& A,
                     const Matrix<b>& B );

    template <typename c, typename a, typename b>
    friend Info mxv( Matrix<c>&       C,
                     const Semiring&  op,
                     const Matrix<a>& A,
                     const Matrix<b>& B,
                     const int TA,
                     const int TB,
                     const int NT,
                     const bool ROW_MAJOR );

    // In <graphblas/backend/apspie/mxm.hpp>
    // For testing
    template <typename c, typename a, typename b>
    friend Info mxmCompute( Matrix<c>&       C,
                     const Semiring&  op,
                     const Matrix<a>& A,
                     const Matrix<b>& B,
                     const int TA,
                     const int TB,
                     const int NT,
                     const bool ROW_MAJOR );
  };

  template <typename T>
  Matrix<T>::Matrix( const Index nrows, const Index ncols )
      : nrows_(nrows), ncols_(ncols), mat_type_(GrB_UNKNOWN)
  {
    // Transfer nrows ncols to Sparse/DenseMatrix data member
    sparse_.nnew( nrows_, ncols_ );
    dense_.nnew( nrows_, ncols_ );
  }

  template <typename T>
  Info Matrix<T>::build( const std::vector<Index>& row_indices,
                         const std::vector<Index>& col_indices,
                         const std::vector<T>&     values,
                         const Index nvals,
                         const Matrix& mask,
                         const BinaryOp& dup )
  {
    mat_type_ = GrB_SPARSE;
    return sparse_.build( row_indices, col_indices, values, nvals, mask.sparse_, dup );
  }

  template <typename T>
  Info Matrix<T>::build( const std::vector<Index>& row_indices,
                         const std::vector<Index>& col_indices,
                         const std::vector<T>&     values,
                         const Index nvals )
  {
    mat_type_ = GrB_SPARSE;
    return sparse_.build( row_indices, col_indices, values, nvals );
  }

  template <typename T>
  Info Matrix<T>::build( const std::vector<T>& values )
  {
    mat_type_ = GrB_DENSE;
    return dense_.build( values );
  }

  template <typename T>
  Info Matrix<T>::extractTuples( std::vector<Index>& row_indices,
                                 std::vector<Index>& col_indices,
                                 std::vector<T>&     values )
  {
    if( mat_type_ == GrB_SPARSE ) 
      return sparse_.extractTuples( row_indices, col_indices, values );
    else
      return GrB_UNINITIALIZED_OBJECT;
  }

  // Method that returns Tuples in dense_ format
  template <typename T>
  Info Matrix<T>::extractTuples( std::vector<T>& values )
  {
    if( mat_type_ == GrB_DENSE ) 
      return dense_.extractTuples( values );
    else
      return GrB_UNINITIALIZED_OBJECT;
  }

  // Private method that sets mat_type, clears and allocates
  template <typename T>
  Info Matrix<T>::setStorage( const Storage mat_type )
  {
    Info err;
		//if( mat_type_ != mat_type )
		//{
			if( mat_type == GrB_SPARSE ) {
				err = sparse_.clear();
				err = sparse_.allocate();
			  mat_type_ = mat_type;
			} else if (mat_type == GrB_DENSE ) {
				err = dense_.clear();
				err = dense_.allocate();
			  mat_type_ = mat_type;
			} else {
				return GrB_UNINITIALIZED_OBJECT;
			}
    //}
    return err;
  }

  template <typename T>
  Info Matrix<T>::clear() 
  {
    Info err;
    mat_type_ = GrB_UNKNOWN;
    err = sparse_.clear();
    err = dense_.clear();
    return err;
  }

  template <typename T>
  Info Matrix<T>::print()
  {
    if( mat_type_ == GrB_SPARSE ) return sparse_.print();
    else if( mat_type_ == GrB_DENSE ) return dense_.print();
    else return GrB_UNINITIALIZED_OBJECT;
  }

  template <typename T>
  Info Matrix<T>::nrows( Index& nrows ) const 
  {
    if( mat_type_ == GrB_SPARSE ) return sparse_.nrows( nrows );
    else if( mat_type_ == GrB_DENSE ) return dense_.nrows( nrows );
    else return GrB_UNINITIALIZED_OBJECT;
  }

  template <typename T>
  Info Matrix<T>::ncols( Index& ncols ) const
  {
    if( mat_type_ == GrB_SPARSE ) return sparse_.ncols( ncols );
    else if( mat_type_ == GrB_DENSE ) return dense_.ncols( ncols );
    else return GrB_UNINITIALIZED_OBJECT;
  }

  template <typename T>
  Info Matrix<T>::nvals( Index& nvals ) const 
  {
    if( mat_type_ == GrB_SPARSE ) return sparse_.nvals( nvals );
    else if( mat_type_ == GrB_DENSE ) return dense_.nvals( nvals );
    else return GrB_UNINITIALIZED_OBJECT;
  }

  template <typename T>
  Info Matrix<T>::getStorage( Storage& mat_type ) const
  {
    mat_type = mat_type_;
    return GrB_SUCCESS;
  }

  template <typename T>
  Info Matrix<T>::printStats() const
  {
    if( mat_type_ == GrB_SPARSE ) return sparse_.printStats();
    return GrB_UNINITIALIZED_OBJECT;
  }

} // backend
} // graphblas

#endif  // GRB_BACKEND_APSPIE_HPP
