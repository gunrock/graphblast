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
  template <typename T, typename U>
  Info reduceInner( T*                     val,
									  const BinaryOp<U,U,U>* accum,
									  const Monoid<U>*       op,
									  const DenseVector<U>*  u,
									  Descriptor*            desc )
  {
    T* d_val = (T*) desc->d_buffer_;
    desc->resize(sizeof(T), "buffer");
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Reduce( NULL, temp_storage_bytes, u->d_val_, d_val, 
        u->nvals_, mgpu::plus<T>(), op->identity() );

    desc->resize( temp_storage_bytes, "temp" );
    cub::DeviceReduce::Reduce( desc->d_temp_, temp_storage_bytes, u->d_val_, 
        d_val, u->nvals_, mgpu::plus<T>(), op->identity() );
    CUDA( cudaMemcpy(val, d_val, sizeof(T), cudaMemcpyDeviceToHost) );

    return GrB_SUCCESS;
  }

  template <typename T, typename U>
  Info reduceInner( T*                     val,
									  const BinaryOp<U,U,U>* accum,
									  const Monoid<U>*       op,
									  const SparseVector<U>* u,
									  Descriptor*            desc )
  {
    T* d_val = (T*) desc->d_buffer_;
    desc->resize(sizeof(T), "buffer");
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Reduce( NULL, temp_storage_bytes, u->d_val_, d_val, 
        u->nvals_, mgpu::plus<T>(), op->identity() );

    desc->resize( temp_storage_bytes, "temp" );
    cub::DeviceReduce::Reduce( desc->d_temp_, temp_storage_bytes, u->d_val_, 
        d_val, u->nvals_, mgpu::plus<T>(), op->identity() );
    CUDA( cudaMemcpy(val, d_val, sizeof(T), cudaMemcpyDeviceToHost) );

    return GrB_SUCCESS;
  }
}  // backend
}  // graphblas

#endif  // GRB_BACKEND_APSPIE_REDUCE_HPP
