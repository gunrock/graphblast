#ifndef GRB_BACKEND_APSPIE_REDUCE_HPP
#define GRB_BACKEND_APSPIE_REDUCE_HPP

#include <iostream>

#include "graphblas/backend/apspie/Descriptor.hpp"
#include "graphblas/backend/apspie/SparseMatrix.hpp"
#include "graphblas/backend/apspie/DenseMatrix.hpp"
#include "graphblas/backend/apspie/operations.hpp"

#include <cub.cuh>

namespace graphblas
{
namespace backend
{
  // Dense vector variant
  template <typename T, typename U>
  Info reduceInner( T*                     val,
									  const BinaryOp<U,U,U>* accum,
									  const Monoid<U>*       op,
									  const DenseVector<U>*  u,
									  Descriptor*            desc )
  {
    // Nasty bug! Must point d_val at desc->d_buffer_ only after it gets 
    // possibly resized!
    CHECK( desc->resize(sizeof(T), "buffer") );
    T* d_val = (T*) desc->d_buffer_;
    size_t temp_storage_bytes = 0;

    if( !desc->split() )
      CUDA( cub::DeviceReduce::Reduce(NULL, temp_storage_bytes, u->d_val_, 
          d_val, u->nvals_, mgpu::plus<T>(), op->identity()) );
    else
      temp_storage_bytes = desc->d_temp_size_;

    CHECK( desc->resize(temp_storage_bytes, "temp") );
    if( desc->debug() )
    {
      std::cout << temp_storage_bytes << " <= " << desc->d_temp_size_ << 
          std::endl;
    }

    CUDA( cub::DeviceReduce::Reduce(desc->d_temp_, temp_storage_bytes, 
        u->d_val_, d_val, u->nvals_, mgpu::plus<T>(), op->identity()) );
    CUDA( cudaMemcpy(val, d_val, sizeof(T), cudaMemcpyDeviceToHost) );

    // If doing reduce on DenseVector, then we might as well write the nnz
    // to the internal variable
    DenseVector<U>* u_t = const_cast<DenseVector<U>*>(u);
    u_t->nnz_ = *val;
    return GrB_SUCCESS;
  }

  // Sparse vector variant
  template <typename T, typename U>
  Info reduceInner( T*                     val,
									  const BinaryOp<U,U,U>* accum,
									  const Monoid<U>*       op,
									  const SparseVector<U>* u,
									  Descriptor*            desc )
  {
    if( desc->struconly() )
    {
      *val = u->nvals_;
    }
    else
    {
      T* d_val = (T*) desc->d_buffer_;
      desc->resize(sizeof(T), "buffer");
      size_t temp_storage_bytes = 0;

      if( !desc->split() )
        cub::DeviceReduce::Reduce( NULL, temp_storage_bytes, u->d_val_, d_val, 
            u->nvals_, mgpu::plus<T>(), op->identity() );
      else
        temp_storage_bytes = desc->d_temp_size_;

      desc->resize( temp_storage_bytes, "temp" );
      cub::DeviceReduce::Reduce( desc->d_temp_, temp_storage_bytes, u->d_val_, 
          d_val, u->nvals_, mgpu::plus<T>(), op->identity() );

      CUDA( cudaMemcpy(val, d_val, sizeof(T), cudaMemcpyDeviceToHost) );
    }

    return GrB_SUCCESS;
  }
}  // backend
}  // graphblas

#endif  // GRB_BACKEND_APSPIE_REDUCE_HPP
