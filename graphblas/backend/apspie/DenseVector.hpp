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

  template <typename T, typename T1, typename T2>
  class BinaryOp;

  template <typename T>
  class DenseVector
  {
    public:
    DenseVector() : nvals_(0), ncapacity_(0), h_val_(NULL), d_val_(NULL), 
          need_update_(0) {}

    DenseVector( Index nvals )
        : nvals_(nvals), ncapacity_(0), h_val_(NULL), d_val_(NULL), 
          need_update_(0)
    {
      allocate(nvals_);
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
                const BinaryOp<T>*        dup );
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
    Info resize( Index nvals );
    Info fill( Index vals );
    Info print( bool forceUpdate = false );
    Info countUnique( Index* count );
 
    private:
    Info allocate( Index nvals );  
    Info cpuToGpu();
    Info gpuToCpu( bool forceUpdate = false );

    private:
    Index nvals_;      // 3 ways to set: (1) dup (2) build (3) nnew (4) resize
                       //                (5) allocate
    Index ncapacity_;
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
    return GrB_SUCCESS;
  }

  template <typename T>
  Info DenseVector<T>::dup( const DenseVector* rhs )
  {
    Info err;
    nvals_ = rhs->nvals_;

    if( d_val_==NULL && h_val_==NULL )
      err = allocate( rhs->nvals_ );
    if( err != GrB_SUCCESS ) return err;

    //std::cout << "copying " << nrows_+1 << " rows\n";
    //std::cout << "copying " << nvals_+1 << " rows\n";

    CUDA( cudaMemcpy( d_val_, rhs->d_val_, nvals_*sizeof(T),
        cudaMemcpyDeviceToDevice ) );

    need_update_ = true;
    return err; 
  }

  template <typename T>
  Info DenseVector<T>::clear()
  {
    if( h_val_ ) {
      free( h_val_ );
      h_val_ = NULL;
    }

    if( d_val_ ) {
      CUDA( cudaFree(d_val_) );
      d_val_ = NULL;
    }
    ncapacity_ = 0;

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
                              const BinaryOp<T>*        dup )
  {
    return GrB_SUCCESS;
  }

  template <typename T>
  Info DenseVector<T>::build( const std::vector<T>* values,
                              Index                 nvals )
  {
    Info err;

    if( d_val_==NULL && h_val_==NULL  )
      err = allocate( nvals );
    else while( ncapacity_ < nvals )
      err = resize( nvals );
    nvals_ = nvals;
    if( err != GrB_SUCCESS ) return err;

    for( Index i=0; i<nvals; i++ )
      h_val_[i] = (*values)[i];

    err = cpuToGpu();

    return err;
  }

  template <typename T>
  Info DenseVector<T>::setElement( T     val,
                                   Index index )
  {
    return GrB_SUCCESS;
  }

  template <typename T>
  Info DenseVector<T>::extractElement( T*    val,
                                       Index index )
  {
    return GrB_SUCCESS;
  }

  template <typename T>
  Info DenseVector<T>::extractTuples( std::vector<Index>* indices,
                                      std::vector<T>*     values,
                                      Index*              n )
  {
    return GrB_SUCCESS;
  }

  template <typename T>
  Info DenseVector<T>::extractTuples( std::vector<T>* values,
                                      Index*          n )
  {
    Info err = gpuToCpu();
    values->clear();

    if( *n>nvals_ )
    {
      err = GrB_UNINITIALIZED_OBJECT;
      *n = nvals_;
    }
    if( *n<nvals_ ) 
      err = GrB_INSUFFICIENT_SPACE;

    for( Index i=0; i<*n; i++ )
      values->push_back( h_val_[i]);
    
    return err;
  }

  template <typename T>
  const T& DenseVector<T>::operator[]( Index ind )
  {
    gpuToCpu();
    if( ind>=nvals_ ) std::cout << "Error: index out of bounds!\n";

    return h_val_[ind];
  }

  // Copies the val to arrays kresize_ratio x bigger than capacity
  template <typename T>
  Info DenseVector<T>::resize( Index nvals )
  {
    T* h_tempVal = h_val_;
    T* d_tempVal = d_val_;

    // Compute how much to copy
    Index to_copy = min(nvals, nvals_);    

    ncapacity_ = nvals;
    h_val_ = (T*) malloc( ncapacity_*sizeof(T) );
    if( h_tempVal!=NULL )
      memcpy( h_val_, h_tempVal, to_copy*sizeof(T) );

    CUDA( cudaMalloc( &d_val_, ncapacity_*sizeof(T)) );
    if( d_tempVal!=NULL )
      CUDA( cudaMemcpy( d_val_, d_tempVal, to_copy*sizeof(T), 
          cudaMemcpyDeviceToDevice) );
    nvals_ = nvals;

    free( h_tempVal );
    CUDA( cudaFree(d_tempVal) );

    return GrB_SUCCESS;
  }

  template <typename T>
  Info DenseVector<T>::fill( Index nvals )
  {
    for( Index i=0; i<nvals; i++ )
      h_val_[i] = i;

    Info err = cpuToGpu();
    return err;
  }

  template <typename T>
  Info DenseVector<T>::print( bool forceUpdate )
  {
    CUDA( cudaDeviceSynchronize() );
    Info err = gpuToCpu( forceUpdate );
    printArray( "val", h_val_, std::min(nvals_,40) );
    return GrB_SUCCESS;
  }

  // Count number of unique numbers
  template <typename T>
  Info DenseVector<T>::countUnique( Index* count )
  {
    Info err = gpuToCpu();
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

    return err;
  }

  // Private methods:
  template <typename T>
  Info DenseVector<T>::allocate( Index nvals )
  {
    // Allocate just enough (different from CPU impl since kcap_ratio=1.)
    //ncapacity_ = nvals_;
    ncapacity_ = nvals;

    // Host malloc
    if( nvals!=0 && h_val_ == NULL )
      h_val_ = (T*) malloc(ncapacity_*sizeof(T));
    else
      return GrB_UNINITIALIZED_OBJECT;

    // GPU malloc
    if( nvals!=0 && d_val_ == NULL )
      CUDA( cudaMalloc( &d_val_, ncapacity_*sizeof(T)) );
    else
      return GrB_UNINITIALIZED_OBJECT;


    if( h_val_==NULL || d_val_==NULL )
      return GrB_OUT_OF_MEMORY;

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

}  // backend
}  // graphblas

#endif  // GRB_BACKEND_APSPIE_DENSEVECTOR_HPP
