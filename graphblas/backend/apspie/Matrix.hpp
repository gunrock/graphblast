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
    Matrix( Index nrows, Index ncols )
        : nrows_(nrows), ncols_(ncols), sparse_(nrows,ncols), 
          dense_(nrows,ncols), mat_type_(GrB_SPARSE) {}

    // Default Destructor is good enough for this layer
    ~Matrix() {}

    // C API Methods

    // Mutators
    Info nnew(  Index nrows, Index ncols );
    Info dup(   const Matrix* rhs );
    Info clear();
    Info nrows( Index* nrows_t );
    Info ncols( Index* ncols_t );
    Info nvals( Index* nvals_t );
    Info build( const std::vector<Index>* row_indices,
                const std::vector<Index>* col_indices,
                const std::vector<T>*     values,
                Index                     nvals,
                const BinaryOp            dup ); 
    Info build( const std::vector<T>* values, 
                Index nvals );

    // Handy methods
    void operator=( const Matrix* rhs );
    const T operator[]( const Index ind );
    Info print();
    Info check();
    Info set_nrows( Index nrows );
    Info set_ncols( Index ncols );
    Info resize( Index nrows,
                 Index ncols );
    Info set_storage( Storage  mat_type );
    Info get_storage( Storage* mat_type ) const;

    template <typename U>
    Info fill( Index axis,
               Index nvals,
               U     start );
    template <typename U>
    Info fill_ascending( Index axis,
                         Index nvals,
                         U     start );

    private:
    Index nrows_;
    Index ncols_;

    SparseMatrix<T> sparse_;
    DenseMatrix<T>  dense_;

    // Keeps track of whether matrix is Sparse or Dense
    Storage mat_type_;

  };

  template <typename T>
  Info Matrix<T>::nnew( Index nrows, Index ncols )
  {
    Info err;

    // Transfer nrows ncols to Sparse/DenseMatrix data member
    err = sparse_.nnew( nrows, ncols );
    err = dense_.nnew( nrows, ncols );
    return err;
  }

  template <typename T>
  Info Matrix<T>::dup( const Matrix* rhs )
  {
    mat_type_ = rhs->mat_type_;

    //std::cout << "Matrix type: " << (int) mat_type_ << "\n";

    if( mat_type_ == GrB_SPARSE )
      return sparse_.dup( rhs->sparse_ );
    else if( mat_type_ == GrB_SPARSE )
      return dense_.dup( rhs->dense_ );
    return GrB_PANIC;
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
  inline Info Matrix<T>::nrows( Index* nrows_t ) const 
  {
    if( mat_type_ == GrB_SPARSE ) return sparse_.nrows( nrows_t );
    else if( mat_type_ == GrB_DENSE ) return dense_.nrows( nrows_t );
    return GrB_UNINITIALIZED_OBJECT;
  }

  template <typename T>
  inline Info Matrix<T>::ncols( Index* ncols_t ) const
  {
    if( mat_type_ == GrB_SPARSE ) return sparse_.ncols( ncols_t );
    else if( mat_type_ == GrB_DENSE ) return dense_.ncols( ncols_t );
    return GrB_UNINITIALIZED_OBJECT;
  }

  template <typename T>
  inline Info Matrix<T>::nvals( Index* nvals_t ) const 
  {
    if( mat_type_ == GrB_SPARSE ) return sparse_.nvals( nvals_t );
    else if( mat_type_ == GrB_DENSE ) return dense_.nvals( nvals_t );
    return GrB_UNINITIALIZED_OBJECT;
  }

  // Option: Not const to allow sorting
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
  Info Matrix<T>::setElement( Index row_index,
                              Index col_index )
  {
    if( mat_type == GrB_SPARSE ) 
      return sparse_.setElement( row_index, col_index );
    else if( mat_type_ == GrB_DENSE )
      return dense_.setElement( row_index, col_index );
    return GrB_UNINITIALIZED_OBJECT;
  }

  template <typename T>
  Info Matrix<T>::extractElement( T*    val,
                                  Index row_index,
                                  Index col_index )
  {
    if( mat_type_ == GrB_SPARSE ) 
      return sparse_.extractElement( val, row_index, col_index );
    else if( mat_type_ == GrB_DENSE ) 
      return dense_.extractElement( val, row_index, col_index );
    return GrB_UNINITIALIZED_OBJECT;
  }

  template <typename T>
  Info Matrix<T>::extractTuples( std::vector<Index>* row_indices,
                                 std::vector<Index>* col_indices,
                                 std::vector<T>*     values,
                                 Index*              n )
  {
    if( mat_type_ == GrB_SPARSE ) 
      return sparse_.extractTuples( row_indices, col_indices, values, n );
    else
      return GrB_UNINITIALIZED_OBJECT;
  }

  template <typename T>
  Info Matrix<T>::extractTuples( std::vector<T>* values, Index* n )
  {
    if( mat_type_ == GrB_DENSE ) 
      return dense_.extractTuples( values, n );
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
  Info Matrix<T>::print( bool forceUpdate )
  {
    if( mat_type_ == GrB_SPARSE ) return sparse_.print( forceUpdate );
    else if( mat_type_ == GrB_DENSE ) return dense_.print();
    return GrB_UNINITIALIZED_OBJECT;
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
    return GrB_UNINITIALIZED_OBJECT;
  }

  template <typename T>
  template <typename U>
  Info Matrix<T>::fill( const Index axis, const Index nvals, const U start )
  {
    if( mat_type_ == GrB_SPARSE ) return sparse_.fill( axis, nvals, start );
    return GrB_UNINITIALIZED_OBJECT;
  }

  template <typename T>
  template <typename U>
  Info Matrix<T>::fillAscending( const Index axis, const Index nvals, 
                                 const U start )
  {
    if( mat_type_ == GrB_SPARSE )
      return sparse_.fillAscending( axis, nvals, start );
    return GrB_UNINITIALIZED_OBJECT;
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
