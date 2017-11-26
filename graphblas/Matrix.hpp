#ifndef GRB_MATRIX_HPP
#define GRB_MATRIX_HPP

#include <vector>

#include "graphblas/types.hpp"
#include "graphblas/Descriptor.hpp"

// Opaque data members from the right backend
#define __GRB_BACKEND_MATRIX_HEADER <graphblas/backend/__GRB_BACKEND_ROOT/Matrix.hpp>
#include __GRB_BACKEND_MATRIX_HEADER
#undef __GRB_BACKEND_MATRIX_HEADER

namespace graphblas
{
  template <typename T1, typename T2, typename T3>
  class BinaryOp;

  template <typename T>
  class Matrix
  {
    public:
    // Default Constructor, Standard Constructor (Replaces new in C++)
    //   -it's imperative to call constructor using matrix_ or the constructed 
    //     object won't be tied to this outermost layer
    Matrix() : matrix_() {}
    Matrix( Index nrows, Index ncols ) : matrix_( nrows, ncols ) {}

    // Default Destructor is good enough for this layer
    ~Matrix() {}

    // C API Methods
    Info nnew(  Index nrows, Index ncols );
    Info dup(   const Matrix* rhs );
    Info clear();
    Info nrows( Index* nrows_ ) const;
    Info ncols( Index* ncols_ ) const;
    Info nvals( Index* nvals_ ) const;
    //template <typename BinaryOpT>
    Info build( const std::vector<Index>* row_indices,
                const std::vector<Index>* col_indices,
                const std::vector<T>*     values,
                Index                     nvals,
                const BinaryOp<T,T,T>*    dup );
    //            const BinaryOpT*          dup );
    Info build( const std::vector<T>*     values,
                Index                     nvals );
    Info setElement(     Index row_index,
                         Index col_index );
    Info extractElement( T*    val,
                         Index row_index,
                         Index col_index );
    Info extractTuples(  std::vector<Index>* row_indices,
                         std::vector<Index>* col_indices,
                         std::vector<T>*     values,
                         Index*              n );
    Info extractTuples(  std::vector<T>* values, 
                         Index* n );

    // Handy methods
    void operator=( const Matrix* rhs );
    const T operator[]( Index ind );
    Info print( bool force_update = false );
    Info check();
    Info setNrows( Index nrows );
    Info setNcols( Index ncols );
    Info resize(   Index nrows, 
                   Index ncols );
    Info setStorage( Storage  mat_type ); 
    Info getStorage( Storage* mat_type ) const;

    template <typename U>
    Info fill( Index axis, 
               Index nvals, 
               U     start );
    template <typename U>
    Info fillAscending( Index axis, 
                        Index nvals, 
                        U     start );

    private:
    // Data members that are same for all backends
    backend::Matrix<T> matrix_;

		template <typename c, typename a, typename b, typename m,
							typename BinaryOpT,     typename SemiringT>
		friend Info mxm( Matrix<c>*        C,
										 const Matrix<m>*  mask,
										 const BinaryOpT*  accum,
										 const SemiringT*  op,
										 const Matrix<a>*  A,
										 const Matrix<b>*  B,
										 const Descriptor* desc );

  };

  template <typename T>
  Info Matrix<T>::nnew( Index nrows, Index ncols )
  {
    if( nrows==0 || ncols==0 ) return GrB_INVALID_VALUE;
    matrix_.nnew( nrows, ncols );
  }

  template <typename T>
  Info Matrix<T>::dup( const Matrix* rhs )
  {
    if( rhs==NULL ) return GrB_NULL_POINTER;
    return matrix_.dup( rhs->matrix_ );
  }

  template <typename T>
  Info Matrix<T>::clear()
  {
    return matrix_.clear();
  }

  template <typename T>
  Info Matrix<T>::nrows( Index* nrows ) const
  {
    if( nrows==NULL ) return GrB_NULL_POINTER;
    return matrix_.nrows( nrows );
  }

  template <typename T>
  Info Matrix<T>::ncols( Index* ncols ) const
  {
    if( ncols==NULL ) return GrB_NULL_POINTER;
    return matrix_.ncols( ncols );
  }

  template <typename T>
  Info Matrix<T>::nvals( Index* nvals ) const
  {
    if( nvals==NULL ) return GrB_NULL_POINTER;
    return matrix_.nvals( nvals );
  }

  template <typename T>
  //template <typename BinaryOpT>
  Info Matrix<T>::build( const std::vector<Index>* row_indices,
                         const std::vector<Index>* col_indices,
                         const std::vector<T>*     values,
                         Index                     nvals,
                         const BinaryOp<T,T,T>*    dup )
  {
    if( row_indices==NULL || col_indices==NULL || values==NULL )
      return GrB_NULL_POINTER;
    const backend::BinaryOp<T,T,T>* dup_t = (dup==NULL) ? NULL : &dup->op_;
    //    : &((BinaryOp<T,T,T>*)dup)->op_;
    return matrix_.build( row_indices, col_indices, values, nvals, dup_t );
  }

  template <typename T>
  Info Matrix<T>::build( const std::vector<T>* values, Index nvals )
  {
    return matrix_.build( values, nvals );
  }

  template <typename T>
  Info Matrix<T>::setElement( Index row_index,
                              Index col_index )
  {
    return matrix_.setElement( row_index, col_index );
  }

  template <typename T>
  Info Matrix<T>::extractElement( T*    val,
                                  Index row_index,
                                  Index col_index )
  {
    if( val==NULL ) return GrB_NULL_POINTER;
    return matrix_.extractElement( val, row_index, col_index );
  }

  template <typename T>
  Info Matrix<T>::extractTuples( std::vector<Index>* row_indices,
                                 std::vector<Index>* col_indices,
                                 std::vector<T>*     values,
                                 Index*              n )
  {
    if( row_indices==NULL || col_indices==NULL || values==NULL || n==NULL )
      return GrB_NULL_POINTER;
    return matrix_.extractTuples( row_indices, col_indices, values, n );
  }
  
  template <typename T>
  Info Matrix<T>::extractTuples( std::vector<T>* values, 
                                 Index*          n )
  {
    if( values==NULL ) return GrB_NULL_POINTER;
    return matrix_.extractTuples( values, n );
  }

  // Handy methods
  template <typename T>
  void Matrix<T>::operator=( const Matrix* rhs )
  {
    if( rhs==NULL ) return;
    matrix_.dup( rhs->matrix_ );
  }

  template <typename T>
  const T Matrix<T>::operator[]( Index ind )
  {
    return matrix_[ind];
  }

  template <typename T>
  Info Matrix<T>::print( bool force_update )
  {
    return matrix_.print( force_update );
  }

  template <typename T>
  Info Matrix<T>::check()
  {
    return matrix_.check();
  }

  template <typename T>
  Info Matrix<T>::setNrows( Index nrows )
  {
    return matrix_.setNrows( nrows );
  }

  template <typename T>
  Info Matrix<T>::setNcols( Index ncols )
  {
    return matrix_.setNcols( ncols );
  }

  template <typename T>
  Info Matrix<T>::resize( Index nrows, 
                          Index ncols )
  {
    return matrix_.resize( nrows, ncols );
  }

  template <typename T>
  Info Matrix<T>::setStorage( Storage mat_type )
  {
    return matrix_.setStorage( mat_type );
  }

  template <typename T>
  Info Matrix<T>::getStorage( Storage* mat_type ) const
  {
    if( mat_type==NULL ) return GrB_NULL_POINTER;
    return matrix_.getStorage( mat_type );
  }

  template <typename T>
  template <typename U>
  Info Matrix<T>::fill( Index axis, 
                        Index nvals, 
                        U     start )
  {
    return matrix_.fill( axis, nvals, start );
  }

	template <typename T>
  template <typename U>
	Info Matrix<T>::fillAscending( Index axis, 
                                 Index nvals, 
                                 U     start )
  {
    return matrix_.fillAscending( axis, nvals, start );
  }

}  // graphblas

#endif  // GRB_MATRIX_HPP
