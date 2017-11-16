#ifndef GRB_BACKEND_APSPIE_VECTOR_HPP
#define GRB_BACKEND_APSPIE_VECTOR_HPP

#include <vector>
#include <iostream>
#include <unordered_set>

#include <cuda.h>
#include <cuda_runtime.h>

#include "graphblas/types.hpp"
#include "graphblas/util.hpp"

#include "graphblas/backend/apspie/apspie.hpp"

namespace graphblas
{
namespace backend
{
  template <typename T>
  class SparseVector;

  template <typename T>
  class DenseVector;
  
  template <typename T1, typename T2, typename T3>
  class BinaryOp;

  template <typename T>
  class Vector
  {
    public:
    Vector() : nvals_(0), sparse_(0), dense_(0), vec_type_(GrB_UNKNOWN) {}
    Vector( Index nvals )
        : nvals_(nvals), sparse_(nvals), dense_(nvals), 
          vec_type_(GrB_UNKNOWN) {}

    // Default destructor is good enough for this layer
    ~Vector() {}

    // C API Methods
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

    // private method for allocation
    const T& operator[]( Index ind );
    Info resize( Index nvals );
    Info fill( Index vals );
    Info print( bool forceUpdate = false );
    Info countUnique( Index* count );
    Info setStorage( Storage  vec_type );
    Info getStorage( Storage* vec_type ) const;
    template <typename VectorT>
    VectorT* getVector() const;
    Info convert();
    Info sparse2dense();
    Info dense2sparse();

    private: 
    Index           nvals_;      // 3 ways to set: (1) dup  (2) build 
                                 //                (3) resize
                                 // Note: not set by nnew()
    SparseVector<T> sparse_;
    DenseVector<T>  dense_;
    Storage         vec_type_;
  };

  template <typename T>
  Info Vector<T>::nnew( Index nsize )
  {
    Info err;
    err = sparse_.nnew( nsize );
    err = dense_.nnew( nsize );
    return err;
  }

  template <typename T>
  Info Vector<T>::dup( const Vector* rhs )
  {
    vec_type_ = rhs->vec_type_;
    if( vec_type_ == GrB_SPARSE )
      return sparse_.dup( &rhs->sparse_ );
    else if( vec_type_ == GrB_SPARSE )
      return dense_.dup( &rhs->dense_ );
    return GrB_PANIC;
  }

  template <typename T>
	Info Vector<T>::clear()
  {
    Info err;
    vec_type_ = GrB_UNKNOWN;
    err = sparse_.clear();
    err = dense_.clear();
    return err;
  }

  template <typename T>
	Info Vector<T>::size( Index* nsize_ ) const
  {
    if( vec_type_ == GrB_SPARSE ) return sparse_.size( nsize_ );
    else if( vec_type_ == GrB_DENSE ) return dense_.size( nsize_ );
    else return GrB_UNINITIALIZED_OBJECT;
  }
  
  template <typename T>
	Info Vector<T>::nvals( Index* nvals_ ) const
  {
    if( vec_type_ == GrB_SPARSE ) return sparse_.nvals( nvals_ );
    else if( vec_type_ == GrB_DENSE ) return dense_.nvals( nvals_ );
    else return GrB_UNINITIALIZED_OBJECT;
  }

  template <typename T>
  Info Vector<T>::build( const std::vector<Index>* indices,
                         const std::vector<T>*     values,
                         Index                     nvals,
                         const BinaryOp<T,T,T>*    dup )
  {
    vec_type_ = GrB_SPARSE;
    return sparse_.build( indices, values, nvals, dup );
  }

  template <typename T>
  Info Vector<T>::build( const std::vector<T>* values,
                         Index                 nvals )
  {
    vec_type_ = GrB_DENSE;
    return dense_.build( values, nvals );
  }

  template <typename T>
	Info Vector<T>::setElement( T     val,
	           									Index index )
  {
    if( vec_type_ == GrB_SPARSE ) return sparse_.setElement( val, index );
    else if( vec_type_ == GrB_DENSE ) return dense_.setElement( val, index );
    else return GrB_UNINITIALIZED_OBJECT;
  }

  template <typename T>
	Info Vector<T>::extractElement( T*    val,
											            Index index )
  {
    if( vec_type_ == GrB_SPARSE ) 
      return sparse_.extractElement( val, index );
    else if( vec_type_ == GrB_DENSE ) 
      return dense_.extractElement( val, index );
    else return GrB_UNINITIALIZED_OBJECT;
  }

  template <typename T>
	Info Vector<T>::extractTuples( std::vector<Index>* indices,
											           std::vector<T>*     values,
											           Index*              n )
  {
    if( vec_type_ == GrB_SPARSE )
      return sparse_.extractTuples( indices, values, n );
    else if( vec_type_ == GrB_DENSE )
      return dense_.extractTuples( indices, values, n );
    else return GrB_UNINITIALIZED_OBJECT;
  }

  template <typename T>
	Info Vector<T>::extractTuples( std::vector<T>* values,
											           Index*          n )
  {
    if( vec_type_ == GrB_SPARSE )
      return sparse_.extractTuples( values, n );
    else if( vec_type_ == GrB_DENSE )
      return dense_.extractTuples( values, n );
    else return GrB_UNINITIALIZED_OBJECT;
  }

  // Handy methods:
  template <typename T>
  const T& Vector<T>::operator[]( Index ind )
  {
    if( vec_type_ == GrB_SPARSE )
      return sparse_[ind];
    else if( vec_type_ == GrB_DENSE )
      return dense_[ind];
    else return GrB_UNINITIALIZED_OBJECT;
  }

  // Copies the val to arrays kresize_ratio x bigger than capacity
  template <typename T>
  Info Vector<T>::resize( Index nvals )
  {
    if( vec_type_ == GrB_SPARSE )
      return sparse_.resize(nvals );
    else if( vec_type_ == GrB_DENSE )
      return dense_.resize( nvals );
    else return GrB_UNINITIALIZED_OBJECT;
  }

  template <typename T>
  Info Vector<T>::fill( const Index nvals )
  {
    if( vec_type_ == GrB_SPARSE )
      return sparse_.fill(nvals );
    else if( vec_type_ == GrB_DENSE )
      return dense_.fill( nvals );
    else return GrB_UNINITIALIZED_OBJECT;
  }

  template <typename T>
  Info Vector<T>::print( bool forceUpdate )
  {
    if( vec_type_ == GrB_SPARSE )
      return sparse_.print( forceUpdate );
    else if( vec_type_ == GrB_DENSE )
      return dense_.print( forceUpdate );
    else return GrB_UNINITIALIZED_OBJECT;
  }

  // Count number of unique numbers
  template <typename T>
  Info Vector<T>::countUnique( Index* count )
  {
    return GrB_SUCCESS;
  }

  // Private method that sets mat_type, clears and allocates
  template <typename T>
  Info Vector<T>::setStorage( Storage vec_type )
  {
    Info err;
    vec_type_ = vec_type;
    if( vec_type_ == GrB_SPARSE ) {
      err = sparse_.clear();
      err = sparse_.allocate();
    } else if( vec_type_ == GrB_DENSE ) {
      err = dense_.clear();
      err = dense_.allocate();
    }
    return err;
  }

  template <typename T>
  inline Info Vector<T>::getStorage( Storage* vec_type ) const
  {
    *vec_type = vec_type_;
    return GrB_SUCCESS;
  }

  template <typename T>
  template <typename VectorT>
  VectorT* Vector<T>::getVector() const
  {
    if( vec_type_ == GrB_SPARSE )     return &sparse_;
    else if( vec_type_ == GrB_DENSE ) return &dense_;
    return NULL;
  }

  // Check if necessary to convert sparse-to-dense or dense-to-sparse
	// a) if more elements than GrB_THRESHOLD, convert SpVec->DeVec
	// b) if less elements than GrB_THRESHOLD, convert DeVec->SpVec
  template <typename T>
  Info Vector<T>::convert()
  {
    Index nvals_t;
    Index nsize_t;
    CHECK( nvals( &nvals_t ) );
    CHECK( size(  &nsize_t ) );
    nvals_ = nvals_t;

    if( vec_type_ == GrB_SPARSE && nvals_t/nsize_t > GrB_THRESHOLD )
      CHECK( sparse2dense() );
    else if( vec_type_ == GrB_DENSE && nvals_t/nsize_t <= GrB_THRESHOLD )
      CHECK( dense2sparse() );
    else if( vec_type_ == GrB_UNKNOWN ) return GrB_UNINITIALIZED_OBJECT;
    return GrB_SUCCESS;
  }

  template <typename T>
  Info Vector<T>::sparse2dense()
  {
    if( vec_type_==GrB_DENSE ) return GrB_INVALID_OBJECT;

    // 1. Initialize memory
    // 2. Call scatter

    CHECK( setStorage( GrB_DENSE ) );
    return GrB_SUCCESS;
  }

  template <typename T>
  Info Vector<T>::dense2sparse()
  {
    if( vec_type_==GrB_SPARSE ) return GrB_INVALID_OBJECT;

    // 1. Initialize memory
    // 2. Run kernel

    CHECK( setStorage(GrB_DENSE) );
    return GrB_SUCCESS;
  }

}  // backend
}  // graphblas

#endif  // GRB_BACKEND_APSPIE_VECTOR_HPP
