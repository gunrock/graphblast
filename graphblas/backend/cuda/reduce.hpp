#ifndef GRAPHBLAS_BACKEND_CUDA_REDUCE_HPP_
#define GRAPHBLAS_BACKEND_CUDA_REDUCE_HPP_

#include <cub.cuh>

#include <iostream>

namespace graphblas {
namespace backend {

// Dense vector variant
template <typename T, typename U,
          typename BinaryOpT, typename MonoidT>
Info reduceInner(T*                     val,
                 BinaryOpT              accum,
                 MonoidT                op,
                 const DenseVector<U>*  u,
                 Descriptor*            desc) {
  // Nasty bug! Must point d_val at desc->d_buffer_ only after it gets
  // possibly resized!
  CHECK(desc->resize(sizeof(T), "buffer"));
  T* d_val = reinterpret_cast<T*>(desc->d_buffer_);
  size_t temp_storage_bytes = 0;

  if (u->nvals_ == 0)
    return GrB_INVALID_OBJECT;

  if (!desc->split())
    CUDA_CALL(cub::DeviceReduce::Reduce(NULL, temp_storage_bytes, u->d_val_,
        d_val, u->nvals_, op, op.identity()));
  else
    temp_storage_bytes = desc->d_temp_size_;

  CHECK(desc->resize(temp_storage_bytes, "temp"));
  if (desc->debug()) {
    std::cout << "u_nvals: " << u->nvals_ << std::endl;
    std::cout << temp_storage_bytes << " <= " << desc->d_temp_size_ <<
        std::endl;
  }

  CUDA_CALL(cub::DeviceReduce::Reduce(desc->d_temp_, temp_storage_bytes,
      u->d_val_, d_val, u->nvals_, op, op.identity()));
  CUDA_CALL(cudaMemcpy(val, d_val, sizeof(T), cudaMemcpyDeviceToHost));

  // If doing reduce on DenseVector, then we might as well write the nnz
  // to the internal variable
  //DenseVector<U>* u_t = const_cast<DenseVector<U>*>(u);
  //u_t->nnz_ = *val;
  return GrB_SUCCESS;
}

// Sparse vector variant
template <typename T, typename U,
          typename BinaryOpT, typename MonoidT>
Info reduceInner(T*                     val,
                 BinaryOpT              accum,
                 MonoidT                op,
                 const SparseVector<U>* u,
                 Descriptor*            desc) {
  if (desc->struconly()) {
    *val = u->nvals_;
  } else {
    T* d_val = reinterpret_cast<T*>(desc->d_buffer_);
    desc->resize(sizeof(T), "buffer");
    size_t temp_storage_bytes = 0;

    if (u->nvals_ == 0)
      return GrB_INVALID_OBJECT;

    if (!desc->split())
      CUDA_CALL(cub::DeviceReduce::Reduce(NULL, temp_storage_bytes, u->d_val_,
          d_val, u->nvals_, op, op.identity()));
    else
      temp_storage_bytes = desc->d_temp_size_;

    desc->resize(temp_storage_bytes, "temp");
    CUDA_CALL(cub::DeviceReduce::Reduce(desc->d_temp_, temp_storage_bytes,
        u->d_val_, d_val, u->nvals_, op, op.identity()));

    CUDA_CALL(cudaMemcpy(val, d_val, sizeof(T), cudaMemcpyDeviceToHost));
  }

  return GrB_SUCCESS;
}

// TODO(@ctcyang): Dense matrix variant
template <typename W, typename a, typename M,
          typename BinaryOpT,     typename MonoidT>
Info reduceInner(DenseVector<W>*       w,
                 const Vector<M>*      mask,
                 BinaryOpT             accum,
                 MonoidT               op,
                 const DenseMatrix<a>* A,
                 Descriptor*           desc) {
  std::cout << "Error: Dense reduce matrix-to-vector not implemented yet!\n";
  return GrB_SUCCESS;
}

// Sparse matrix variant
template <typename W, typename a, typename M,
          typename BinaryOpT,     typename MonoidT>
Info reduceInner(DenseVector<W>*        w,
                 const Vector<M>*       mask,
                 BinaryOpT              accum,
                 MonoidT                op,
                 const SparseMatrix<a>* A,
                 Descriptor*            desc) {
  // TODO(@ctcyang): Structure-only optimization uses CSR row pointers
  if (desc->struconly()) {
  } else {
    // Cannot use mgpu, because BinaryOps and Monoids do not satisfy
    // first_argument_type requirement for mgpu ops
    // mgpu::SegReduceCsr( A->d_csrVal_, A->d_csrRowPtr_,
    //     static_cast<int>(A->nvals_), static_cast<int>(A->nrows_),
    //     true, w->d_val_, op.identity(), op, *desc->d_context_ );

    // Use CUB
    size_t temp_storage_bytes = 0;

    if (A->nrows_ == 0)
      return GrB_INVALID_OBJECT;
    
    if (!desc->split())
      CUDA_CALL(cub::DeviceSegmentedReduce::Reduce(NULL, temp_storage_bytes,
          A->d_csrVal_, w->d_val_, A->nrows_, A->d_csrRowPtr_,
          A->d_csrRowPtr_+1, op, op.identity()));
    else
      temp_storage_bytes = desc->d_temp_size_;
    desc->resize(temp_storage_bytes, "temp");
    CUDA_CALL(cub::DeviceSegmentedReduce::Reduce(desc->d_temp_,
        temp_storage_bytes, A->d_csrVal_, w->d_val_, A->nrows_,
        A->d_csrRowPtr_, A->d_csrRowPtr_+1, op, op.identity()));
    w->nnz_ = A->nrows_;
  }
  w->need_update_ = true;

  return GrB_SUCCESS;
}

// TODO(@ctcyang): Dense matrix variant
template <typename T, typename a,
          typename BinaryOpT,     typename MonoidT>
Info reduceInner(T*                    val,
                 BinaryOpT             accum,
                 MonoidT               op,
                 const DenseMatrix<a>* A,
                 Descriptor*           desc) {
  std::cout << "Error: Dense reduce matrix-to-scalar not implemented yet!\n";
  return GrB_SUCCESS;
}

// Sparse matrix variant
template <typename T, typename a,
          typename BinaryOpT,     typename MonoidT>
Info reduceInner(T*                     val,
                 BinaryOpT              accum,
                 MonoidT                op,
                 const SparseMatrix<a>* A,
                 Descriptor*            desc) {
  if (desc->struconly()) {
    *val = A->nvals_;
  } else {
    T* d_val = reinterpret_cast<T*>(desc->d_buffer_);
    desc->resize(sizeof(T), "buffer");
    size_t temp_storage_bytes = 0;

    if (A->nvals_ == 0)
      return GrB_INVALID_OBJECT;

    if (!desc->split())
      CUDA_CALL(cub::DeviceReduce::Reduce(NULL, temp_storage_bytes,
          A->d_csrVal_, d_val, A->nvals_, op, op.identity()));
    else
      temp_storage_bytes = desc->d_temp_size_;

    desc->resize(temp_storage_bytes, "temp");
    CUDA_CALL(cub::DeviceReduce::Reduce(desc->d_temp_, temp_storage_bytes,
        A->d_csrVal_, d_val, A->nvals_, op, op.identity()));

    CUDA_CALL(cudaMemcpy(val, d_val, sizeof(T), cudaMemcpyDeviceToHost));
  }

  return GrB_SUCCESS;
}

}  // namespace backend
}  // namespace graphblas

#endif  // GRAPHBLAS_BACKEND_CUDA_REDUCE_HPP_
