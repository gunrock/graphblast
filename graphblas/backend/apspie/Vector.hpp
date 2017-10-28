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
#include "graphblas/backend/apspie/Matrix.hpp"

namespace graphblas
{
namespace backend
{
  template <typename T>
  class Vector
  {
    public:
    Vector() : nvals_(0), ncapacity_(0), h_val_(NULL), d_val_(NULL), 
               need_update_(0)
    {
      allocate(nvals_);
    }

    Vector( const Index nvals )
        : nvals_(nvals), ncapacity_(0), h_val_(NULL), d_val_(NULL), 
          need_update_(0)
    {
      allocate(nvals_);
    }

    // C API Methods
    Info dup( const Vector& rhs );

    Info build( const std::vector<T>& values,
                const Index     nvals );

    // No longer an accessor for GPU version
    Info extract( std::vector<T>& values );

    // Mutators
    Info nnew( const Index nvals );
    // private method for allocation
    Info allocate( const Index nvals );  
    Info resize( const Index nvals );
    Info fill( const Index vals );
    Info clear();
    Info print( bool forceUpdate = false );
    Info cpuToGpu();
    Info gpuToCpu( bool forceUpdate = false );
    Info countUnique( Index& count );
    const T& operator[]( const Index ind );

    // Accessors
    Info getNvals( Index& nvals ) const;
 
    // Accessors

    private:
    Index nvals_;      // 3 ways to set: (1) dup (2) build (3) nnew (4) resize
                       //                (5) allocate
    Index ncapacity_;
    T*    h_val_;
    T*    d_val_;

    bool  need_update_; // set to true by changing Vector
                       // set to false by gpuToCpu()
  };

  template <typename T>
  Info Vector<T>::dup( const Vector& rhs )
  {
    Info err;
    nvals_ = rhs.nvals_;

    if( d_val_==NULL && h_val_==NULL )
      err = allocate( rhs.nvals_ );
    if( err != GrB_SUCCESS ) return err;

    //std::cout << "copying " << nrows_+1 << " rows\n";
    //std::cout << "copying " << nvals_+1 << " rows\n";

    CUDA( cudaMemcpy( d_val_, rhs.d_val_, nvals_*sizeof(T),
        cudaMemcpyDeviceToDevice ) );

    need_update_ = true;
    return err; 
  }

  template <typename T>
  Info Vector<T>::build( const std::vector<T>& values,
                         const Index           nvals )
  {
    Info err;

    if( d_val_==NULL && h_val_==NULL  )
      err = allocate( nvals );
    else while( ncapacity_ < nvals )
      err = resize( nvals );
    nvals_ = nvals;
    if( err != GrB_SUCCESS ) return err;

    for( Index i=0; i<nvals; i++ )
      h_val_[i] = values[i];

    err = cpuToGpu();

    return err;
  }

  template <typename T>
  Info Vector<T>::extract( std::vector<T>& values )
  {
    Info err = gpuToCpu();
    values.clear();

    for( Index i=0; i<nvals_; i++ )
      values.push_back( h_val_[i]);

    return err;
  }

  template <typename T>
  Info Vector<T>::nnew( const Index nvals )
  {
    nvals_ = nvals;
    return GrB_SUCCESS;
  }

  template <typename T>
  Info Vector<T>::allocate( const Index nvals )
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

  // Copies the val to arrays kresize_ratio x bigger than capacity
  template <typename T>
  Info Vector<T>::resize( const Index nvals )
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
  Info Vector<T>::fill( const Index nvals )
  {
    for( Index i=0; i<nvals; i++ )
      h_val_[i] = i;

    Info err = cpuToGpu();
    return err;
  }

  template <typename T>
  Info Vector<T>::clear()
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
  Info Vector<T>::print( bool forceUpdate )
  {
    CUDA( cudaDeviceSynchronize() );
    Info err = gpuToCpu( forceUpdate );
    printArray( "val", h_val_, std::min(nvals_,40) );
    return GrB_SUCCESS;
  }

  // Copies graph to GPU
  template <typename T>
  Info Vector<T>::cpuToGpu()
  {
    CUDA( cudaMemcpy( d_val_, h_val_, nvals_*sizeof(T),
        cudaMemcpyHostToDevice ) );
    return GrB_SUCCESS;
  }

  // Copies graph to CPU
  template <typename T>
  Info Vector<T>::gpuToCpu( bool forceUpdate )
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

  // Count number of unique numbers
  template <typename T>
  Info Vector<T>::countUnique( Index& count )
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

  template <typename T>
  const T& Vector<T>::operator[]( const Index ind )
  {
    gpuToCpu();
    if( ind>=nvals_ ) std::cout << "Error: index out of bounds!\n";

    return h_val_[ind];
  }

  template <typename T>
  inline Info Vector<T>::getNvals( Index& nvals ) const
  {
    nvals = nvals_;
    return GrB_SUCCESS;
  }

}  // backend
}  // graphblas

#endif  // GRB_BACKEND_APSPIE_VECTOR_HPP
