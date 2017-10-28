#ifndef GRB_BACKEND_APSPIE_MATRIX_HPP
#define GRB_BACKEND_APSPIE_MATRIX_HPP

#include <vector>
#include <iostream>
#include <map>

#include "graphblas/backend/apspie/Vector.hpp"
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
    // Alternative to new in C++
    Matrix() : nrows_(0), ncols_(0), sparse_(0,0), dense_(0,0),
               mat_type_(GrB_SPARSE) {}
    Matrix( const Index nrows, const Index ncols )
        : nrows_(nrows), ncols_(ncols), sparse_(nrows,ncols), 
          dense_(nrows,ncols), mat_type_(GrB_SPARSE) {}

    // Destructor
    // TODO
    ~Matrix() {};

    // C API Methods

    // Mutators
    Info nnew( const Index nrows, const Index ncols );
    Info dup( const Matrix& rhs );
    Info build( const std::vector<Index>* row_indices,
                const std::vector<Index>* col_indices,
                const std::vector<T>*     values,
                Index                     nvals,
                const BinaryOp            dup ); 
    Info build( const std::vector<T>* values, Index nvals );
    Info buildFromVec( const Vector<T>& a );
    // Private method for setting storage type of matrix
    Info setStorage( const Storage mat_type );
    Info clear(); 
    Info print( bool forceUpdate = false );
    Info setNrows( const Index nrows );
    Info setNcols( const Index ncols );
    Info resize( const Index nrows, const Index ncols );
    template <typename U>
    Info fill( const Index axis, const Index nvals, const U start );
    template <typename U>
    Info fillAscending( const Index axis, const Index nvals, const U start );
    const T operator[]( const Index ind );
    Info check();

    // Accessors
    Info extractTuples( std::vector<Index>& row_indices,
                        std::vector<Index>& col_indices,
                        std::vector<T>&     values );
    Info extractTuples( std::vector<T>& values );
    Info getNrows( Index& nrows ) const;
    Info getNcols( Index& ncols ) const;
    Info getNvals( Index& nvals ) const; 
    Info getStorage( Storage& mat_type ) const;
    Info getRowPtr( Index& row_ptr, const Index row ) const;
    Info getColInd( Index& col_ind, const Index row_ptr ) const;
    Info getColInd( Index& col_ind, const Index offset, 
        const Index row ) const;
    Info getVal( T& col_val, const Index row_ptr ) const; 
    Info getVal( T& col_val, const Index offset, const Index row ) const;
    Info find( T& found, const Index target, const Index row ) const;
    Info checkRowReduce( const T sum, const Index row ) const;

    private:
    Index nrows_;
    Index ncols_;

    SparseMatrix<T> sparse_;
    DenseMatrix<T>  dense_;

    // Keeps track of whether matrix is Sparse or Dense
    Storage mat_type_;

  };

  template <typename T>
  Info Matrix<T>::nnew( const Index nrows, const Index ncols )
  {
    Info err;

    // Transfer nrows ncols to Sparse/DenseMatrix data member
    err = sparse_.nnew( nrows, ncols );
    err = dense_.nnew( nrows, ncols );
    return err;
  }

  template <typename T>
  Info Matrix<T>::dup( const Matrix& rhs )
  {
    mat_type_ = rhs.mat_type_;

    //std::cout << "Matrix type: " << (int) mat_type_ << "\n";

    if( mat_type_ == GrB_SPARSE )
      return sparse_.dup( rhs.sparse_ );
    else if( mat_type_ == GrB_SPARSE )
      return dense_.dup( rhs.dense_ );
    return GrB_PANIC;
  }

  // Not const to allow sorting
  template <typename T>
  Info Matrix<T>::build( const std::vector<Index>* row_indices,
                         const std::vector<Index>* col_indices,
                         const std::vector<T>*     values,
                         Index                     nvals,
                         const BinaryOp            dup )
  {
    mat_type_ = GrB_SPARSE;
    return sparse_.build( row_indices, col_indices, values, nvals, dup );
  }

  template <typename T>
  Info Matrix<T>::build( const std::vector<T>* values, Index nvals )
  {
    mat_type_ = GrB_DENSE;
    return dense_.build( values, nvals );
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
    mat_type_ = mat_type;
    if( mat_type_ == GrB_SPARSE ) {
      err = sparse_.clear();
      err = sparse_.allocate();
    } else if( mat_type_ == GrB_DENSE ) {
      err = dense_.clear();
      err = dense_.allocate();
    }
    return err;
  }

  template <typename T>
  Info Matrix<T>::clear() 
  {
    Info err;
    mat_type_ = GrB_SPARSE;
    err = sparse_.clear();
    err = dense_.clear();
    return err;
  }

  template <typename T>
  Info Matrix<T>::print( bool forceUpdate )
  {
    if( mat_type_ == GrB_SPARSE ) return sparse_.print( forceUpdate );
    else if( mat_type_ == GrB_DENSE ) return dense_.print();
    else return GrB_UNINITIALIZED_OBJECT;
  }

  template <typename T>
  Info Matrix<T>::setNrows( const Index nrows )
  {
    Info err;
    err = sparse_.setNrows( nrows );
    err = dense_.setNrows( nrows );
    return err;
  }

  template <typename T>
  Info Matrix<T>::setNcols( const Index ncols )
  {
    Info err;
    err = sparse_.setNcols( ncols );
    err = dense_.setNcols( ncols );
    return err;
  }

  template <typename T>
  Info Matrix<T>::resize( const Index nrows, const Index ncols )
  {
    if( mat_type_ == GrB_SPARSE ) return sparse_.resize( nrows, ncols );
    else return GrB_UNINITIALIZED_OBJECT;
  }

  template <typename T>
  template <typename U>
  Info Matrix<T>::fill( const Index axis, const Index nvals, const U start )
  {
    if( mat_type_ == GrB_SPARSE ) return sparse_.fill( axis, nvals, start );
    else return GrB_UNINITIALIZED_OBJECT;
  }

  template <typename T>
  template <typename U>
  Info Matrix<T>::fillAscending( const Index axis, const Index nvals, 
                                 const U start )
  {
    if( mat_type_ == GrB_SPARSE )
      return sparse_.fillAscending( axis, nvals, start );
    else return GrB_UNINITIALIZED_OBJECT;
  }

  template <typename T>
  const T Matrix<T>::operator[]( const Index ind )
  {
    if( mat_type_ == GrB_SPARSE )
      return sparse_[ind];
    else std::cout << "Error: operator[] not defined for dense matrices!\n";
    return 0.;
  }

  // Error checking function
  template <typename T>
  Info Matrix<T>::check()
  {
    if( mat_type_ == GrB_SPARSE )
      return sparse_.check();
    return GrB_UNINITIALIZED_OBJECT;
  }

  template <typename T>
  inline Info Matrix<T>::getNrows( Index& nrows ) const 
  {
    if( mat_type_ == GrB_SPARSE ) return sparse_.getNrows( nrows );
    else if( mat_type_ == GrB_DENSE ) return dense_.getNrows( nrows );
    else return GrB_UNINITIALIZED_OBJECT;
  }

  template <typename T>
  inline Info Matrix<T>::getNcols( Index& ncols ) const
  {
    if( mat_type_ == GrB_SPARSE ) return sparse_.getNcols( ncols );
    else if( mat_type_ == GrB_DENSE ) return dense_.getNcols( ncols );
    else return GrB_UNINITIALIZED_OBJECT;
  }

  template <typename T>
  inline Info Matrix<T>::getNvals( Index& nvals ) const 
  {
    if( mat_type_ == GrB_SPARSE ) return sparse_.getNvals( nvals );
    else if( mat_type_ == GrB_DENSE ) return dense_.getNvals( nvals );
    else return GrB_UNINITIALIZED_OBJECT;
  }

  template <typename T>
  inline Info Matrix<T>::getStorage( Storage& mat_type ) const
  {
    mat_type = mat_type_;
    return GrB_SUCCESS;
  }

  template <typename T>
  inline Info Matrix<T>::getRowPtr( Index& row_ptr, const Index row ) const
  {
    if( mat_type_ == GrB_SPARSE )
      return sparse_.getRowPtr( row_ptr, row );
    return GrB_UNINITIALIZED_OBJECT;
  }

  // Variant where row_ptr output is already known
  template <typename T>
  inline Info Matrix<T>::getColInd( Index& col_ind, const Index row_ptr ) const
  {
    if( mat_type_ == GrB_SPARSE ) 
      return sparse_.getColInd( col_ind, row_ptr );
    return GrB_UNINITIALIZED_OBJECT;
  }

  // Variant where only row_ind and offset are known
  template <typename T>
  inline Info Matrix<T>::getColInd( Index& col_ind, const Index offset, 
      const Index row ) const
  {
    if( mat_type_ == GrB_SPARSE ) 
      return sparse_.getColInd( col_ind, offset, row );
    return GrB_UNINITIALIZED_OBJECT;
  }

  // Variant where row_ptr output is already known
  template <typename T>
  inline Info Matrix<T>::getVal( T& col_val, const Index row_ptr ) const
  {
    if( mat_type_ == GrB_SPARSE ) 
      return sparse_.getVal( col_val, row_ptr );
    return GrB_UNINITIALIZED_OBJECT;
  }

  // Variant where only row_ind and offset are known
  template <typename T>
  inline Info Matrix<T>::getVal( T& col_val, const Index offset, 
      const Index row ) const
  {
    if( mat_type_ == GrB_SPARSE ) 
      return sparse_.getVal( col_val, offset, row );
    return GrB_UNINITIALIZED_OBJECT;
  }

  template <typename T>
  Info Matrix<T>::find( T& found, const Index target, const Index row )
      const
  {
    if( mat_type_ == GrB_SPARSE )
      return sparse_.find( found, target, row );
    return GrB_UNINITIALIZED_OBJECT;
  }

  template <typename T>
  Info Matrix<T>::checkRowReduce( const T sum, const Index row ) const
  {
    if( mat_type_ == GrB_SPARSE )
      return sparse_.checkRowReduce( sum, row );
    return GrB_UNINITIALIZED_OBJECT;
  }
}  // backend
}  // graphblas

#endif  // GRB_BACKEND_APSPIE_MATRIX_HPP
