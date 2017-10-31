#ifndef GRB_VECTOR_HPP
#define GRB_VECTOR_HPP

#include <vector>

#include "graphblas/types.hpp"
#include "graphblas/util.hpp"

// Opaque data members from the right backend
#define __GRB_BACKEND_VECTOR_HEADER <graphblas/backend/__GRB_BACKEND_ROOT/Vector.hpp>
#include __GRB_BACKEND_VECTOR_HEADER
#undef __GRB_BACKEND_VECTOR_HEADER

namespace graphblas
{
  template <typename T>
  class Vector
  {
    public:
    Vector() : vector_() {}
    Vector( Index nsize ) : vector_( nsize ) {}

    // Default Destructor is good enough for this layer
    ~Vector() {}

    // C API Methods
    // Note: extractTuples no longer an accessor for GPU version
    Info nnew(  Index nsize );
    Info dup(   const Vector* rhs );
    Info clear();
    Info size(  Index* nsize_ ) const;
    Info nvals( Index* nvals_ ) const;
    Info build( const std::vector<Index>* indices,
                const std::vector<T>*     values,
                Index                     nvals,
                const BinaryOp<T,T,T>*    dup );
    Info build( const std::vector<T>* values,
                Index                 nvals );
    Info setElement(     T val, 
                         Index index );
    Info extractElement( T*    val, 
                         Index index );
    Info extractTuples(  std::vector<Index>* indices,
                         std::vector<T>*     values,
                         Index*              n );
    Info extractTuples(  std::vector<T>* values, 
                         Index*          n );

    // Some handy methods
    void operator=(      Vector* rhs );
    const T& operator[]( Index ind );
    Info resize(         Index nvals );
    Info fill(           Index vals );
    Info print(          bool force_update = false );
    Info countUnique(    Index* count );
    Info setStorage( Storage  vec_type );
    Info getStorage( Storage* vec_type ) const;

    private:
    backend::Vector<T> vector_;

    /*template <typename m, typename U, typename a, typename BinaryOp, 
              typename Semiring>
    friend Info vxm( const Vector<m>*  mask,
                     const BinaryOp*   accum,
                     const Semiring*   op,
                     const Vector<U>*  u,
                     const Matrix<a>*  A,
                     const Descriptor* desc );*/
  };

  template <typename T>
  Info Vector<T>::nnew( Index nsize )
  {
    return vector_.nnew( nsize );
  }

  template <typename T>
  Info Vector<T>::dup( const Vector* rhs )
  {
    return vector_.dup( &rhs->vector_ );
  }

  template <typename T>
  Info Vector<T>::clear()
  {
    return vector_.clear();
  }

  template <typename T>
  Info Vector<T>::size( Index* nsize_ ) const
  {
    if( nsize_==NULL ) return GrB_NULL_POINTER;
    return vector_.size( nsize_ );
  }

  template <typename T>
  Info Vector<T>::nvals( Index* nvals_ ) const
  {
    if( nvals_==NULL ) return GrB_NULL_POINTER;
    return vector_.nvals( nvals_ );
  }

  template <typename T>
  Info Vector<T>::build( const std::vector<Index>* indices,
                         const std::vector<T>*     values,
                         Index                     nvals,
                         const BinaryOp<T,T,T>*    dup )
  {
    if( indices==NULL || values==NULL || dup==NULL )
      return GrB_NULL_POINTER;
    return vector_.build( indices, values, nvals, dup );
  }

  template <typename T>
  Info Vector<T>::build( const std::vector<T>* values,
                         Index                 nvals )
  {
    if( values==NULL ) return GrB_NULL_POINTER;
    return vector_.build( values, nvals );
  }

  template <typename T>
  Info Vector<T>::setElement( T val, Index index )
  {
    return vector_.setElement( val, index );
  }
  
  template <typename T>
  Info Vector<T>::extractElement( T* val, Index index )
  {
    if( val==NULL ) return GrB_NULL_POINTER;
    return vector_.extractElement( val, index );
  }

  template <typename T>
  Info Vector<T>::extractTuples( std::vector<Index>* indices,
                                 std::vector<T>*     values,
                                 Index*              n )
  {
    if( indices==NULL || values==NULL || n==NULL ) return GrB_NULL_POINTER;
    return vector_.extractTuples( indices, values, n );
  }

  template <typename T>
  Info Vector<T>::extractTuples( std::vector<T>* values, 
                                 Index*          n )
  {
    if( values==NULL || n==NULL ) return GrB_NULL_POINTER;
    return vector_.extractTuples( values, n );
  }

  template <typename T>
  void Vector<T>::operator=( Vector* rhs )
  {
    if( rhs==NULL ) return;
    vector_.dup( &rhs->vector_ );
  }

  template <typename T>
  const T& Vector<T>::operator[]( Index ind )
  {
    return vector_[ind];
  }

  // Copies the val to arrays kresize_ratio x bigger than capacity
  template <typename T>
  Info Vector<T>::resize( Index nvals )
  {
    return vector_.resize( nvals );
  }

  template <typename T>
  Info Vector<T>::fill( Index nvals )
  {
    return vector_.fill( nvals );
  }

  template <typename T>
  Info Vector<T>::print( bool force_update )
  {
    return vector_.print( force_update );
  }

  // Count number of unique numbers
  template <typename T>
  Info Vector<T>::countUnique( Index* count )
  {
    if( count==NULL ) return GrB_NULL_POINTER;
    return vector_.countUnique( count );
  }

  template <typename T>
  Info Vector<T>::setStorage( Storage vec_type )
  {
    return vector_.setStorage( vec_type );
  }
  
  template <typename T>
  Info Vector<T>::getStorage( Storage* vec_type ) const
  {
    if( vec_type==NULL ) return GrB_NULL_POINTER;
    return vector_.getStorage( vec_type );
  }

}  // graphblas

#endif  // GRB_VECTOR_HPP
