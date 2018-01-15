#ifndef GRB_BACKEND_APSPIE_DENSEVECTOR_HPP
#define GRB_BACKEND_APSPIE_DENSEVECTOR_HPP

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

  template <typename T1, typename T2, typename T3>
  class BinaryOp;

  template <typename T>
  class DenseVector
  {
    public:
    DenseVector() : nvals_(0), h_val_(NULL), d_val_(NULL), need_update_(0) {}

    DenseVector( Index nsize )
        : nvals_(nsize), h_val_(NULL), d_val_(NULL), need_update_(0)
    {
      allocate();
    }

    // Need to write Default Destructor
    ~DenseVector();

    // C API Methods
    Info nnew(  Index nsize );
    Info dup(   const DenseVector* rhs );
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

    // handy methods
    const T& operator[]( Index ind );
    Info resize( Index nsize );
    Info fill( T val );
    Info fillAscending( Index vals );
    Info print( bool forceUpdate = false );
    Info countUnique( Index* count );
    Info allocate();  
    Info cpuToGpu();
    Info gpuToCpu( bool forceUpdate = false );
    Info swap( DenseVector* rhs );

    private:
    // Note nsize_ is understood to be the same as nvals_, so it is omitted
    Index nvals_; // 6 ways to set: (1) Vector (2) nnew (3) dup (4) build 
                  //                (5) resize (6) allocate
    T*    h_val_;
    T*    d_val_;

    bool  need_update_; // set to true by changing DenseVector
                       // set to false by gpuToCpu()
  };

  template <typename T>
  DenseVector<T>::~DenseVector()
  {
    if( h_val_!=NULL ) free(h_val_);
    if( d_val_!=NULL ) CUDA( cudaFree(d_val_) );
  }

  template <typename T>
  Info DenseVector<T>::nnew( Index nsize )
  {
    nvals_ = nsize;
    CHECK( allocate() );
    return GrB_SUCCESS;
  }

  template <typename T>
  Info DenseVector<T>::dup( const DenseVector* rhs )
  {
    nvals_ = rhs->nvals_;

    if( d_val_==NULL || h_val_==NULL )
      CHECK( allocate() );

    //std::cout << "copying " << nrows_+1 << " rows\n";
    //std::cout << "copying " << nvals_+1 << " rows\n";

    CUDA( cudaMemcpy( d_val_, rhs->d_val_, nvals_*sizeof(T),
        cudaMemcpyDeviceToDevice ) );

    need_update_ = true;
    return GrB_SUCCESS; 
  }

  template <typename T>
  Info DenseVector<T>::clear()
  {
    CHECK( fill((T) 0) );
    return GrB_SUCCESS;
  }

  template <typename T>
  Info DenseVector<T>::size( Index* nsize_t ) const
  {
    *nsize_t = nvals_;
    return GrB_SUCCESS;
  }

  template <typename T>
  Info DenseVector<T>::nvals( Index* nvals_t ) const
  {
    *nvals_t = nvals_;
    return GrB_SUCCESS;
  }

  template <typename T>
  Info DenseVector<T>::build( const std::vector<Index>* indices,
                              const std::vector<T>*     values,
                              Index                     nvals,
                              const BinaryOp<T,T,T>*    dup )
  {
    std::cout << "DeVec Build Using Sparse Indices\n";
    std::cout << "Error: Feature not implemented yet!\n";
    return GrB_SUCCESS;
  }

  template <typename T>
  Info DenseVector<T>::build( const std::vector<T>* values,
                              Index                 nvals )
  {
    if( nvals > nvals_ )
      return GrB_INDEX_OUT_OF_BOUNDS;
    if( d_val_==NULL || h_val_==NULL  )
      return GrB_UNINITIALIZED_OBJECT;

    for( Index i=0; i<nvals; i++ )
      h_val_[i] = (*values)[i];

    CHECK( cpuToGpu() );

    return GrB_SUCCESS;
  }

  template <typename T>
  Info DenseVector<T>::setElement( T     val,
                                   Index index )
  {
    CHECK( gpuToCpu() );
    h_val_[index] = val;
    CHECK( cpuToGpu() );
    return GrB_SUCCESS;
  }

  template <typename T>
  Info DenseVector<T>::extractElement( T*    val,
                                       Index index )
  {
    std::cout << "DeVec ExtractElement\n";
    std::cout << "Error: Feature not implemented yet!\n";
    return GrB_SUCCESS;
  }

  template <typename T>
  Info DenseVector<T>::extractTuples( std::vector<Index>* indices,
                                      std::vector<T>*     values,
                                      Index*              n )
  {
    std::cout << "DeVec ExtractTuples into Sparse Indices\n";
    std::cout << "Error: Feature not implemented yet!\n";
    return GrB_SUCCESS;
  }

  template <typename T>
  Info DenseVector<T>::extractTuples( std::vector<T>* values,
                                      Index*          n )
  {
    CHECK( gpuToCpu() );
    values->clear();

    if( *n>nvals_ )
    {
      std::cout << "Error: DeVec Too many tuples requested!\n";
      return GrB_UNINITIALIZED_OBJECT;
    }
    if( *n<nvals_ ) 
    {
      std::cout << "Error: DeVec Insufficient space!\n";
      //return GrB_INSUFFICIENT_SPACE;
    }

    for( Index i=0; i<*n; i++ )
      values->push_back( h_val_[i]);
    
    return GrB_SUCCESS;
  }

  template <typename T>
  const T& DenseVector<T>::operator[]( Index ind )
  {
    CHECKVOID( gpuToCpu() );
    if( ind>=nvals_ ) std::cout << "Error: Index out of bounds!\n";

    return h_val_[ind];
  }

  // Copies the val to arrays kresize_ratio x bigger than capacity
  template <typename T>
  Info DenseVector<T>::resize( Index nsize )
  {
    T* h_tempVal = h_val_;
    T* d_tempVal = d_val_;

    // Compute how much to copy
    Index to_copy = min(nsize, nvals_);    

    nvals_ = nsize;
    h_val_ = (T*) malloc( nvals_*sizeof(T) );
    if( h_tempVal!=NULL )
      memcpy( h_val_, h_tempVal, to_copy*sizeof(T) );

    CUDA( cudaMalloc( &d_val_, nvals_*sizeof(T)) );
    if( d_tempVal!=NULL )
      CUDA( cudaMemcpy( d_val_, d_tempVal, to_copy*sizeof(T), 
          cudaMemcpyDeviceToDevice) );
    nvals_ = nsize;

    free( h_tempVal );
    CUDA( cudaFree(d_tempVal) );

    return GrB_SUCCESS;
  }

  template <typename T>
  Info DenseVector<T>::fill( T val )
  {
    for( Index i=0; i<nvals_; i++ )
      h_val_[i] = val;

    CHECK( cpuToGpu() );
    return GrB_SUCCESS;
  }

  template <typename T>
  Info DenseVector<T>::fillAscending( Index nvals )
  {
    for( Index i=0; i<nvals; i++ )
      h_val_[i] = i;

    CHECK( cpuToGpu() );
    return GrB_SUCCESS;
  }

  template <typename T>
  Info DenseVector<T>::print( bool forceUpdate )
  {
    CUDA( cudaDeviceSynchronize() );
    CHECK( gpuToCpu(forceUpdate) );
    printArray( "val", h_val_, std::min(nvals_,40) );
    return GrB_SUCCESS;
  }

  // Count number of unique numbers
  template <typename T>
  Info DenseVector<T>::countUnique( Index* count )
  {
    CHECK( gpuToCpu() );
    std::unordered_set<Index> unique;
    for( Index block=0; block<nvals_; block++ )
    {
      if( unique.find( h_val_[block] )==unique.end() )
      {
        unique.insert( h_val_[block] );
        //std::cout << "Inserting " << array[block] << std::endl;
      }
    }
    *count = unique.size();

    return GrB_SUCCESS;
  }

  // Allocate just enough (different from CPU impl since kcap_ratio=1.)
  template <typename T>
  Info DenseVector<T>::allocate()
  {
    // Host malloc
    if( nvals_>0 && h_val_ == NULL )
      h_val_ = (T*) malloc(nvals_*sizeof(T));
    else
    {
      //std::cout << "Error: DeVec Host allocation unsuccessful!\n";
      //return GrB_UNINITIALIZED_OBJECT;
    }

    // GPU malloc
    if( nvals_>0 && d_val_ == NULL )
      CUDA( cudaMalloc( &d_val_, nvals_*sizeof(T)) );
    else
    {
      //std::cout << "Error: DeVec Device allocation unsuccessful!\n";
      //return GrB_UNINITIALIZED_OBJECT;
    }

    if( h_val_==NULL || d_val_==NULL )
    {
      std::cout << "Error: DeVec Out of memory!\n";
      //return GrB_OUT_OF_MEMORY;
    }

    return GrB_SUCCESS;
  }

  // Copies graph to GPU
  template <typename T>
  Info DenseVector<T>::cpuToGpu()
  {
    CUDA( cudaMemcpy( d_val_, h_val_, nvals_*sizeof(T),
        cudaMemcpyHostToDevice ) );
    return GrB_SUCCESS;
  }

  // Copies graph to CPU
  template <typename T>
  Info DenseVector<T>::gpuToCpu( bool forceUpdate )
  {
    if( need_update_ || forceUpdate )
    {
      CUDA( cudaMemcpy( h_val_, d_val_, nvals_*sizeof(T),
          cudaMemcpyDeviceToHost ) );
      CUDA( cudaDeviceSynchronize() );
    }
    need_update_ = false;
    return GrB_SUCCESS;
  }

  template <typename T>
  Info DenseVector<T>::swap( DenseVector* rhs )
  {
    // Change member scalars
    Index temp_nvals = nvals_;
    nvals_ = rhs->nvals_;
    rhs->nvals_ = temp_nvals;

    // Only need to change GPU pointers
    T*     temp_val_ = d_val_;
    d_val_           = rhs->d_val_;
    rhs->d_val_      = temp_val_;

    need_update_     = true;
    rhs->need_update_= true;

    return GrB_SUCCESS;
  }

}  // backend
}  // graphblas

#endif  // GRB_BACKEND_APSPIE_DENSEVECTOR_HPP
