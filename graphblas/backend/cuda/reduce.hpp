#ifndef GRAPHBLAS_BACKEND_CUDA_REDUCE_HPP_
#define GRAPHBLAS_BACKEND_CUDA_REDUCE_HPP_

#include <cub.cuh>

#include <iostream>

namespace graphblas {
namespace backend {

template <typename T, typename U,
          typename BinaryOpT, typename MonoidT>
Info reduceCommon(T*          val,
                  BinaryOpT   accum,
                  MonoidT     op,
                  U*          d_val,
                  Index       nvals,
                  Descriptor* desc) {
  // Nasty bug! Must point d_temp_val at desc->d_buffer_ only after it gets
  // possibly resized!
  CHECK(desc->resize(sizeof(T), "buffer"));
  T* d_temp_val = reinterpret_cast<T*>(desc->d_buffer_);
  size_t temp_storage_bytes = 0;

  if (nvals == 0)
    return GrB_INVALID_OBJECT;

  CUDA_CALL(cub::DeviceReduce::Reduce(NULL, temp_storage_bytes, d_val,
      d_temp_val, nvals, op, op.identity()));

  CHECK(desc->resize(temp_storage_bytes, "temp"));

  if (desc->debug()) {
    std::cout << "nvals: " << nvals << std::endl;
    std::cout << temp_storage_bytes << " <= " << desc->d_temp_size_ <<
        std::endl;
  }

  CUDA_CALL(cub::DeviceReduce::Reduce(desc->d_temp_, temp_storage_bytes,
      d_val, d_temp_val, nvals, op, op.identity()));
  CUDA_CALL(cudaMemcpy(val, d_temp_val, sizeof(T), cudaMemcpyDeviceToHost));

  // If doing reduce on DenseVector, then we might as well write the nnz
  // to the internal variable
  //DenseVector<U>* u_t = const_cast<DenseVector<U>*>(u);
  //u_t->nnz_ = *val;
  return GrB_SUCCESS;
}

// Dense vector variant
template <typename T, typename U,
          typename BinaryOpT, typename MonoidT>
Info reduceInner(T*                     val,
                 BinaryOpT              accum,
                 MonoidT                op,
                 const DenseVector<U>*  u,
                 Descriptor*            desc) {
  return reduceCommon(val, accum, op, u->d_val_, u->nvals_, desc);
}

// Sparse vector variant
template <typename T, typename U,
          typename BinaryOpT, typename MonoidT>
Info reduceInner(T*                     val,
                 BinaryOpT              accum,
                 MonoidT                op,
                 const SparseVector<U>* u,
                 Descriptor*            desc) {
  if (desc->struconly())
    *val = u->nvals_;
  else
    return reduceCommon(val, accum, op, u->d_val_, u->nvals_, desc);
  return GrB_SUCCESS;
}

// Sparse matrix-to-Scalar variant
template <typename T, typename a,
          typename BinaryOpT, typename MonoidT>
Info reduceInner(T*                     val,
                 BinaryOpT              accum,
                 MonoidT                op,
                 const SparseMatrix<a>* A,
                 Descriptor*            desc) {
  if (desc->struconly())
    *val = A->nvals_;
  else
    return reduceCommon(val, accum, op, A->d_csrVal_, A->nvals_, desc);
  return GrB_SUCCESS;
}

// TODO(@ctcyang): Dense matrix-to-Dense matrix variant
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

// Sparse matrix-to-Dense vector variant
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
}  // namespace backend
}  // namespace graphblas

#endif  // GRAPHBLAS_BACKEND_CUDA_REDUCE_HPP_
