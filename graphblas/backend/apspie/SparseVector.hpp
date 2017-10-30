#ifndef GRB_BACKEND_APSPIE_SPARSEVECTOR_HPP
#define GRB_BACKEND_APSPIE_SPARSEVECTOR_HPP

#include <vector>
#include <iostream>
#include <unordered_set>

#include <cuda.h>
#include <cuda_runtime.h>

#include "graphblas/types.hpp"
#include "graphblas/util.hpp"

#include "graphblas/backend/apspie/apspie.hpp"
#include "graphblas/backend/apspie/Matrix.hpp"

namespace graphblas
{
namespace backend
{
  template <typename T>
  class DenseVector;

  template <typename T>
  class SparseVector
  {
    public:
    SparseVector() : nsize_(0), nvals_(0), ncapacity_(0), h_ind_(NULL), 
          h_val_(NULL), d_ind_(NULL), d_val_(NULL), need_update_(0) {}

    SparseVector( Index nvals )
        : nsize_(0), nvals_(nvals), ncapacity_(0), h_ind_(NULL), h_val_(NULL), 
          d_ind_(NULL), d_val_(NULL), need_update_(0)
    {
      allocate(nvals);
    }

    // Need to write Default Destructor
    ~SparseVector();

    // C API Methods
    Info nnew(  Index nvals );
    Info dup(   const SparseVector* rhs );
    Info clear();
    Info size(  Index* nsize_t  ) const;
    Info nvals( Index* nvals_t ) const;
    Info build( const std::vector<Index>* indices,
                const std::vector<T>*     values,
                Index                     nvals,
                const BinaryOp*           dup );
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
    Index  nsize_;      // 3 ways to set: (1) dup (2) build (3) nnew (4) resize
                        //                (5) allocate
    Index  nvals_;
    Index  ncapacity_;
    Index* h_ind_;
    T*     h_val_;
    Index* d_ind_;
    T*     d_val_;

    bool  need_update_; // set to true by changing SparseVector
                       // set to false by gpuToCpu()
  };

  template <typename T>
  SparseVector<T>::~SparseVector()
  {
    if( h_ind_!=NULL ) free(h_ind_);
    if( h_val_!=NULL ) free(h_val_);
    if( d_ind_!=NULL ) CUDA( cudaFree(d_ind_) );
    if( d_ind_!=NULL ) CUDA( cudaFree(d_val_) );
  }

  template <typename T>
  Info SparseVector<T>::nnew( Index nvals )
  {
    nvals_ = nvals;
    return GrB_SUCCESS;
  }

  template <typename T>
  Info SparseVector<T>::dup( const SparseVector* rhs )
  {
    Info err;
    nvals_ = rhs->nvals_;

    if( d_ind_==NULL && h_ind_==NULL && d_val_==NULL && h_val_==NULL )
      err = allocate( rhs->nvals_ );
    if( err != GrB_SUCCESS ) return err;

    //std::cout << "copying " << nrows_+1 << " rows\n";
    //std::cout << "copying " << nvals_+1 << " rows\n";

    CUDA( cudaMemcpy( d_ind_, rhs->d_ind_, nvals_*sizeof(Index),
        cudaMemcpyDeviceToDevice ) );
    CUDA( cudaMemcpy( d_val_, rhs->d_val_, nvals_*sizeof(T),
        cudaMemcpyDeviceToDevice ) );

    need_update_ = true;
    return err; 
  }

  template <typename T>
  Info SparseVector<T>::clear()
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
  Info SparseVector<T>::size( Index* nsize_t ) const
  {
    if( nsize_t==NULL ) return GrB_NULL_POINTER;
    *nsize_t = nsize_;
    return GrB_SUCCESS;
  }

  template <typename T>
  Info SparseVector<T>::nvals( Index* nvals_t ) const
  {
    if( nvals_t==NULL ) return GrB_NULL_POINTER;
    *nvals_t = nvals_;
    return GrB_SUCCESS;
  }

  template <typename T>
  Info SparseVector<T>::build( const std::vector<Index>* indices,
                               const std::vector<T>*     values,
                               Index                     nvals,
                               const BinaryOp*           dup )
  {
    Info err;

    if( d_ind_==NULL && h_ind_==NULL && d_val_==NULL && h_val_==NULL  )
      err = allocate( nvals );
    else while( ncapacity_ < nvals )
      err = resize( nvals );
    nvals_ = nvals;
    if( err != GrB_SUCCESS ) return err;

    for( Index i=0; i<nvals; i++ )
    {
      h_ind_[i] = (*indices)[i];
      h_val_[i] = (*values) [i];
    }

    err = cpuToGpu();

    return err;
  }

  template <typename T>
  Info SparseVector<T>::build( const std::vector<T>* values,
                               Index                 nvals )
  {
    return GrB_SUCCESS;
  }

  template <typename T>
  Info SparseVector<T>::setElement( T     val,
                                    Index index )
  {
    return GrB_SUCCESS;
  }

  template <typename T>
  Info SparseVector<T>::extractElement( T*    val,
                                        Index index )
  {
    return GrB_SUCCESS;
  }

  template <typename T>
  Info SparseVector<T>::extractTuples( std::vector<Index>* indices,
                                       std::vector<T>*     values,
                                       Index*              n )
  {
    Info err = gpuToCpu();
    indices->clear();
    values->clear();

    if( n==NULL ) return GrB_NULL_POINTER;
    if( *n>nvals_ )
    {
      err = GrB_UNINITIALIZED_OBJECT;
      *n  = nvals_;
    }
    else if( *n<nvals_ ) 
      err = GrB_INSUFFICIENT_SPACE;

    for( Index i=0; i<*n; i++ )
    {
      indices->push_back(h_ind_[i] );
      values->push_back( h_val_[i] );
    }
   
    return err;
  }

  template <typename T>
  Info SparseVector<T>::extractTuples( std::vector<T>* values,
                                       Index*          n )
  {
    return GrB_SUCCESS;
  }

  // If ind is found, then return the value at that ind
  // Else if ind is not found, return 0 of type T
  template <typename T>
  const T& SparseVector<T>::operator[]( Index ind )
  {
    gpuToCpu();
    if( ind>=nvals_ ) std::cout << "Error: index out of bounds!\n";

    for( Index i=0; i<nvals_; i++ )
      if( h_ind_[i]==ind )
        return h_val_[i];
    return T(0);
  }

  // Copies the val to arrays kresize_ratio x bigger than capacity
  template <typename T>
  Info SparseVector<T>::resize( Index nvals )
  {
    Index* h_temp_ind = h_ind_;
    T*     h_temp_val = h_val_;
    Index* d_temp_ind = d_ind_;
    T*     d_temp_val = d_val_;

    // Compute how much to copy
    Index to_copy = min(nvals, nvals_);    

    ncapacity_ = nvals;
    h_ind_ = (Index*) malloc( ncapacity_*sizeof(Index) );
    h_val_ = (T*)     malloc( ncapacity_*sizeof(T) );
    if( h_temp_ind!=NULL )
      memcpy( h_ind_, h_temp_ind, to_copy*sizeof(Index) );
    if( h_temp_val!=NULL )
      memcpy( h_val_, h_temp_val, to_copy*sizeof(T) );

    CUDA( cudaMalloc( &d_ind_, ncapacity_*sizeof(Index)) );
    CUDA( cudaMalloc( &d_val_, ncapacity_*sizeof(T)) );
    if( d_temp_ind!=NULL )
      CUDA( cudaMemcpy( d_ind_, d_temp_ind, to_copy*sizeof(Index), 
          cudaMemcpyDeviceToDevice) );
    if( d_temp_val!=NULL )
      CUDA( cudaMemcpy( d_val_, d_temp_val, to_copy*sizeof(T), 
          cudaMemcpyDeviceToDevice) );
    nvals_ = nvals;

    free( h_temp_ind );
    free( h_temp_val );
    CUDA( cudaFree(d_temp_ind) );
    CUDA( cudaFree(d_temp_val) );

    return GrB_SUCCESS;
  }

  template <typename T>
  Info SparseVector<T>::fill( Index nvals )
  {
    for( Index i=0; i<nvals; i++ )
      h_val_[i] = i;

    Info err = cpuToGpu();
    return err;
  }

  template <typename T>
  Info SparseVector<T>::print( bool forceUpdate )
  {
    CUDA( cudaDeviceSynchronize() );
    Info err = gpuToCpu( forceUpdate );
    printArray( "ind", h_ind_, std::min(nvals_,40) );
    printArray( "val", h_val_, std::min(nvals_,40) );
    return GrB_SUCCESS;
  }

  // Count number of unique numbers
  template <typename T>
  Info SparseVector<T>::countUnique( Index* count )
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
    count = unique.size();

    return err;
  }

  // Private methods:
  template <typename T>
  Info SparseVector<T>::allocate( Index nvals )
  {
    // Allocate just enough (different from CPU impl since kcap_ratio=1.)
    ncapacity_ = nvals;

    // Host malloc
    if( nvals!=0 && h_ind_==NULL && h_val_==NULL )
    {
      h_ind_ = (Index*) malloc(ncapacity_*sizeof(Index));
      h_val_ = (T*)     malloc(ncapacity_*sizeof(T));
    }
    else
      return GrB_UNINITIALIZED_OBJECT;

    // GPU malloc
    if( nvals!=0 && d_ind_==NULL && d_val_==NULL )
    {
      CUDA( cudaMalloc( &d_ind_, ncapacity_*sizeof(Index)) );
      CUDA( cudaMalloc( &d_val_, ncapacity_*sizeof(T)) );
    }
    else
      return GrB_UNINITIALIZED_OBJECT;


    if( h_ind_==NULL || h_val_==NULL || d_ind_==NULL || d_val_==NULL )
      return GrB_OUT_OF_MEMORY;

    return GrB_SUCCESS;
  }

  // Copies graph to GPU
  template <typename T>
  Info SparseVector<T>::cpuToGpu()
  {
    CUDA( cudaMemcpy( d_ind_, h_ind_, nvals_*sizeof(Index),
        cudaMemcpyHostToDevice ) );
    CUDA( cudaMemcpy( d_val_, h_val_, nvals_*sizeof(T),
        cudaMemcpyHostToDevice ) );
    return GrB_SUCCESS;
  }

  // Copies graph to CPU
  template <typename T>
  Info SparseVector<T>::gpuToCpu( bool forceUpdate )
  {
    if( need_update_ || forceUpdate )
    {
      CUDA( cudaMemcpy( h_ind_, d_ind_, nvals_*sizeof(Index),
          cudaMemcpyDeviceToHost ) );
      CUDA( cudaMemcpy( h_val_, d_val_, nvals_*sizeof(T),
          cudaMemcpyDeviceToHost ) );
      CUDA( cudaDeviceSynchronize() );
    }
    need_update_ = false;
    return GrB_SUCCESS;
  }

}  // backend
}  // graphblas

#endif  // GRB_BACKEND_APSPIE_SPARSEVECTOR_HPP
