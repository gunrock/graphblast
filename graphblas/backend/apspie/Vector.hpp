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
    Vector() 
        : nsize_(0), nvals_(0), sparse_(0), dense_(0), vec_type_(GrB_UNKNOWN) {}
    Vector( Index nsize )
        : nsize_(nsize), nvals_(0), sparse_(nsize), dense_(nsize), 
          vec_type_(GrB_UNKNOWN) {}

    // Default destructor is good enough for this layer
    ~Vector() {}

    // C API Methods
    Info nnew(  Index nsize_t );
    Info dup(   const Vector* rhs );
    Info clear();
    Info size(  Index* nsize_t );
    Info nvals( Index* nvals_t );
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
    Info fill( T val );
    Info fillAscending( Index nvals );
    Info print( bool forceUpdate = false );
    Info countUnique( Index* count );
    Info setStorage( Storage  vec_type );
    Info getStorage( Storage* vec_type ) const;
    Info convert( T identity, int tol );
    Info sparse2dense( T identity );
    Info dense2sparse( T identity, int tol );

    private: 
    Index           nsize_;
    Index           nvals_;
    SparseVector<T> sparse_;
    DenseVector<T>  dense_;
    Storage         vec_type_;
  };

  // nsize_ is not modified, because it only gets modified in size()
  template <typename T>
  Info Vector<T>::nnew( Index nsize_t )
  {
    CHECK( sparse_.nnew(nsize_t) );
    CHECK(  dense_.nnew(nsize_t) );
    return GrB_SUCCESS;
  }

  template <typename T>
  Info Vector<T>::dup( const Vector* rhs )
  {
    vec_type_ = rhs->vec_type_;
    if( vec_type_ == GrB_SPARSE )
      return sparse_.dup( &rhs->sparse_ );
    else if( vec_type_ == GrB_DENSE )
      return dense_.dup( &rhs->dense_ );
    std::cout << "Error: Failed to call dup!\n";
    return GrB_UNINITIALIZED_OBJECT;
  }

  template <typename T>
	Info Vector<T>::clear()
  {
    vec_type_ = GrB_UNKNOWN;
    nvals_    = 0;
    CHECK( sparse_.clear() );
    CHECK(  dense_.clear() );
    return GrB_SUCCESS;
  }

  // Calls size_ from SparseVector or DenseVector
  // Updates nsize_ with the latest value
  template <typename T>
	Info Vector<T>::size( Index* nsize_t )
  {
    Index nsize;
    if(      vec_type_ == GrB_SPARSE ) CHECK( sparse_.size(&nsize) );
    else if( vec_type_ == GrB_DENSE  ) CHECK(  dense_.size(&nsize) );
    else nsize = nsize_;

    // Update nsize_ with latest value
    nsize_   = nsize;
    *nsize_t = nsize;
    return GrB_SUCCESS;
  }
  
  template <typename T>
	Info Vector<T>::nvals( Index* nvals_t )
  {
    Index new_nvals;
    if(      vec_type_ == GrB_SPARSE ) CHECK( sparse_.nvals(&new_nvals) );
    else if( vec_type_ == GrB_DENSE  ) CHECK(  dense_.nvals(&new_nvals) );
    else new_nvals = nvals_;

    // Update nvals_ with latest value;
    nvals_   = new_nvals;
    *nvals_t = new_nvals;
    return GrB_SUCCESS;
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
    if(      vec_type_ == GrB_SPARSE ) return sparse_.setElement( val, index );
    else if( vec_type_ == GrB_DENSE  ) return  dense_.setElement( val, index );
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

  // Fill constant value
  template <typename T>
  Info Vector<T>::fill( T val )
  {
    if( vec_type_!=GrB_DENSE )
      CHECK( setStorage(GrB_DENSE) );
    return dense_.fill( val );
  }

  // Fill ascending
  template <typename T>
  Info Vector<T>::fillAscending( Index nvals )
  {
    if( vec_type_ != GrB_DENSE )
      CHECK( setStorage( GrB_DENSE ) );
    return dense_.fillAscending( nvals );
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

  // Private method that sets mat_type, and tries to allocate
  template <typename T>
  Info Vector<T>::setStorage( Storage vec_type )
  {
    vec_type_ = vec_type;
    if(        vec_type_ == GrB_SPARSE ) {
      //CHECK( sparse_.clear()         );
      CHECK( sparse_.allocate(nsize_));
    } else if( vec_type_ == GrB_DENSE ) {
      //CHECK( dense_.clear()          );
      CHECK( dense_.allocate(nsize_) );
    }
    return GrB_SUCCESS;
  }

  template <typename T>
  inline Info Vector<T>::getStorage( Storage* vec_type ) const
  {
    *vec_type = vec_type_;
    return GrB_SUCCESS;
  }

  // Check if necessary to convert sparse-to-dense or dense-to-sparse
	// a) if more elements than GrB_THRESHOLD, convert SpVec->DeVec
	// b) if less elements than GrB_THRESHOLD, convert DeVec->SpVec
  template <typename T>
  Info Vector<T>::convert( T identity, int tol )
  {
    Index nvals_t;
    Index nsize_t;
    if( vec_type_ == GrB_SPARSE )
    {
      CHECK( sparse_.nvals(&nvals_t) );
      CHECK(  sparse_.size(&nsize_t) );
    }
    else if( vec_type_ == GrB_DENSE )
    {
      CHECK( sparse_.nvals(&nvals_t) );
      CHECK(  sparse_.size(&nsize_t) );
    }
    else return GrB_UNINITIALIZED_OBJECT;

    nvals_ = nvals_t;
    nsize_ = nsize_t;

    if( vec_type_ == GrB_SPARSE && nvals_t/nsize_t > GrB_THRESHOLD )
      CHECK( sparse2dense( identity ) );
    else if( vec_type_ == GrB_DENSE && nvals_t/nsize_t <= GrB_THRESHOLD )
      CHECK( dense2sparse( identity, tol ) );
    return GrB_SUCCESS;
  }

  template <typename T>
  Info Vector<T>::sparse2dense( T identity )
  {
    if( vec_type_==GrB_DENSE ) return GrB_INVALID_OBJECT;

    // 1. Initialize memory
    // 2. Call scatter

    CHECK( setStorage( GrB_DENSE ) );
    return GrB_SUCCESS;
  }

  template <typename T>
  Info Vector<T>::dense2sparse( T identity, int tol )
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
