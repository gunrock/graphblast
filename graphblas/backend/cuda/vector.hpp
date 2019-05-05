#ifndef GRAPHBLAS_BACKEND_CUDA_VECTOR_HPP_
#define GRAPHBLAS_BACKEND_CUDA_VECTOR_HPP_

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <iostream>
#include <algorithm>

#include "graphblas/types.hpp"
#include "graphblas/util.hpp"
#include "graphblas/backend/cuda/kernels/kernels.hpp"

namespace graphblas {
namespace backend {

template <typename T>
class SparseVector;

template <typename T>
class DenseVector;

template <typename T>
class Vector {
 public:
  Vector()
      : nsize_(0), nvals_(0), sparse_(0), dense_(0), vec_type_(GrB_UNKNOWN),
        ratio_(0) {}
  explicit Vector(Index nsize)
      : nsize_(nsize), nvals_(0), sparse_(nsize), dense_(nsize),
        vec_type_(GrB_UNKNOWN), ratio_(0) {}

  // Default destructor is good enough for this layer
  ~Vector() {}

  // C API Methods
  Info nnew(Index nsize_t);
  Info dup(const Vector* rhs);
  Info clear();
  Info size(Index* nsize_t);
  Info nvals(Index* nvals_t);
  template <typename BinaryOpT>
  Info build(const std::vector<Index>* indices,
             const std::vector<T>*     values,
             Index                     nvals,
             BinaryOpT                 dup);
  Info build(const std::vector<T>* values,
             Index                 nvals);
  Info build(Index* indices,
             T*     values,
             Index nvals);
  Info build(T*    values,
             Index nvals);
  Info setElement(T val, Index index);
  Info extractElement(T* val, Index index);
  Info extractTuples(std::vector<Index>* indices,
                     std::vector<T>*     values,
                     Index*              n);
  Info extractTuples(std::vector<T>* values, Index* n);

  // private method for allocation
  const T& operator[](Index ind);
  Info resize(Index nvals);
  Info fill(T val);
  Info fillAscending(Index nvals);
  Info print(bool force_update = false);
  Info countUnique(Index* count);
  inline Info setStorage(Storage  vec_type);
  inline Info getStorage(Storage* vec_type) const;
  Info convert(T identity, float switchpoint, Descriptor* desc);
  Info sparse2dense(T identity, Descriptor* desc = NULL);
  Info dense2sparse(T identity, Descriptor* desc);
  Info swap(Vector* rhs);

 private:
  Index           nsize_;
  Index           nvals_;
  SparseVector<T> sparse_;
  DenseVector<T>  dense_;
  Storage         vec_type_;

  float           ratio_;
};

// nsize_ is not modified, because it only gets modified in size()
template <typename T>
Info Vector<T>::nnew(Index nsize_t) {
  CHECK(sparse_.nnew(nsize_t));
  CHECK(dense_.nnew(nsize_t));
  return GrB_SUCCESS;
}

template <typename T>
Info Vector<T>::dup(const Vector* rhs) {
  vec_type_ = rhs->vec_type_;
  if (vec_type_ == GrB_SPARSE)
    return sparse_.dup(&rhs->sparse_);
  else if (vec_type_ == GrB_DENSE)
    return dense_.dup(&rhs->dense_);
  std::cout << "Error: Failed to call dup!\n";
  return GrB_UNINITIALIZED_OBJECT;
}

template <typename T>
Info Vector<T>::clear() {
  vec_type_ = GrB_UNKNOWN;
  nvals_    = 0;
  CHECK(sparse_.clear());
  CHECK(dense_.clear());
  return GrB_SUCCESS;
}

// Calls size_ from SparseVector or DenseVector
// Updates nsize_ with the latest value
template <typename T>
Info Vector<T>::size(Index* nsize_t) {
  Index nsize;
  if (vec_type_ == GrB_SPARSE)
    CHECK(sparse_.size(&nsize));
  else if (vec_type_ == GrB_DENSE)
    CHECK(dense_.size(&nsize));
  else
    nsize = nsize_;

  // Update nsize_ with latest value
  nsize_   = nsize;
  *nsize_t = nsize;
  return GrB_SUCCESS;
}

template <typename T>
Info Vector<T>::nvals(Index* nvals_t) {
  Index new_nvals;
  if (vec_type_ == GrB_SPARSE)
    CHECK(sparse_.nvals(&new_nvals));
  else if (vec_type_ == GrB_DENSE)
    CHECK(dense_.nvals(&new_nvals));
  else
    new_nvals = nvals_;

  // Update nvals_ with latest value;
  nvals_   = new_nvals;
  *nvals_t = new_nvals;
  return GrB_SUCCESS;
}

template <typename T>
template <typename BinaryOpT>
Info Vector<T>::build(const std::vector<Index>* indices,
                      const std::vector<T>*     values,
                      Index                     nvals,
                      BinaryOpT                 dup) {
  vec_type_ = GrB_SPARSE;
  return sparse_.build(indices, values, nvals, dup);
}

template <typename T>
Info Vector<T>::build(const std::vector<T>* values,
                      Index                 nvals) {
  vec_type_ = GrB_DENSE;
  return dense_.build(values, nvals);
}

template <typename T>
Info Vector<T>::build(Index* indices,
                      T*     values,
                      Index nvals) {
  vec_type_ = GrB_SPARSE;
  return sparse_.build(indices, values, nvals);
}

template <typename T>
Info Vector<T>::build(T*    values,
                      Index nvals) {
  vec_type_ = GrB_DENSE;
  return dense_.build(values, nvals);
}

template <typename T>
Info Vector<T>::setElement(T val, Index index) {
  if (vec_type_ == GrB_SPARSE) return sparse_.setElement(val, index);
  else if (vec_type_ == GrB_DENSE) return  dense_.setElement(val, index);
  return GrB_UNINITIALIZED_OBJECT;
}

template <typename T>
Info Vector<T>::extractElement(T* val, Index index) {
  if (vec_type_ == GrB_SPARSE)
    return sparse_.extractElement(val, index);
  else if (vec_type_ == GrB_DENSE)
    return dense_.extractElement(val, index);
  return GrB_UNINITIALIZED_OBJECT;
}

template <typename T>
Info Vector<T>::extractTuples(std::vector<Index>* indices,
                              std::vector<T>*     values,
                              Index*              n) {
  if (vec_type_ == GrB_SPARSE)
    return sparse_.extractTuples(indices, values, n);
  else if (vec_type_ == GrB_DENSE)
    return dense_.extractTuples(indices, values, n);
  return GrB_UNINITIALIZED_OBJECT;
}

template <typename T>
Info Vector<T>::extractTuples(std::vector<T>* values,
                              Index*          n) {
  if (vec_type_ == GrB_SPARSE) {
    CHECK(sparse2dense(0.f));
    return dense_.extractTuples(values, n);
  } else if (vec_type_ == GrB_DENSE) {
    return dense_.extractTuples(values, n);
  }
  return GrB_UNINITIALIZED_OBJECT;
}

// Handy methods:
template <typename T>
const T& Vector<T>::operator[](Index ind) {
  if (vec_type_ == GrB_SPARSE)
    return sparse_[ind];
  else if (vec_type_ == GrB_DENSE)
    return dense_[ind];
  return GrB_UNINITIALIZED_OBJECT;
}

// Copies the val to arrays kresize_ratio x bigger than capacity
template <typename T>
Info Vector<T>::resize(Index nvals) {
  if (vec_type_ == GrB_SPARSE)
    return sparse_.resize(nvals);
  else if (vec_type_ == GrB_DENSE)
    return dense_.resize(nvals);
  return GrB_UNINITIALIZED_OBJECT;
}

// Fill constant value
template <typename T>
Info Vector<T>::fill(T val) {
  if (vec_type_ != GrB_DENSE)
    CHECK(setStorage(GrB_DENSE));
  return dense_.fill(val);
}

// Fill ascending
template <typename T>
Info Vector<T>::fillAscending(Index nvals) {
  if (vec_type_ != GrB_DENSE)
    CHECK(setStorage(GrB_DENSE));
  return dense_.fillAscending(nvals);
}

template <typename T>
Info Vector<T>::print(bool force_update) {
  if (vec_type_ == GrB_SPARSE)
    return sparse_.print(force_update);
  else if (vec_type_ == GrB_DENSE)
    return dense_.print(force_update);
  std::cout << "Error: Vector not initialized!\n";
  return GrB_SUCCESS;
}

// Count number of unique numbers
template <typename T>
Info Vector<T>::countUnique(Index* count) {
  return GrB_SUCCESS;
}

// Private method that sets mat_type, and tries to allocate
template <typename T>
inline Info Vector<T>::setStorage(Storage vec_type) {
  vec_type_ = vec_type;
  if (vec_type_ == GrB_SPARSE)
    CHECK(sparse_.allocate());
  else if (vec_type_ == GrB_DENSE)
    CHECK(dense_.allocate());
  return GrB_SUCCESS;
}

template <typename T>
inline Info Vector<T>::getStorage(Storage* vec_type) const {
  *vec_type = vec_type_;
  return GrB_SUCCESS;
}

// Check if necessary to convert sparse-to-dense or dense-to-sparse
// a) if more elements than desc->switchpoint(), convert SpVec->DeVec
// b) if less elements than desc->switchpoint(), convert DeVec->SpVec
template <typename T>
Info Vector<T>::convert(T identity, float switchpoint, Descriptor* desc) {
  Index nvals_t;
  Index nsize_t;
  if (vec_type_ == GrB_SPARSE) {
    CHECK(sparse_.nvals(&nvals_t));
    CHECK(sparse_.size(&nsize_t));
  } else if (vec_type_ == GrB_DENSE) {
    CHECK(dense_.nnz(&nvals_t));
    CHECK(dense_.computeNnz(&nvals_t, identity, desc));
    CHECK(dense_.nvals(&nsize_t));
  } else {
    return GrB_UNINITIALIZED_OBJECT;
  }

  float ratio = static_cast<float>(nvals_t)/nsize_t;
  if (desc->dirinfo())
    std::cout << "Nnz ratio: " << ratio << " Switch point: "
        << switchpoint << std::endl;

  if (vec_type_ == GrB_SPARSE) {
    if (ratio > switchpoint && ratio > ratio_)
      CHECK(sparse2dense(identity, desc));
    else
      ratio_ = ratio;
  } else if (vec_type_ == GrB_DENSE) {
    if (ratio <= switchpoint && ratio < ratio_)
      CHECK(dense2sparse(identity, desc));
    else
      ratio_ = ratio;
  }
  return GrB_SUCCESS;
}

template <typename T>
Info Vector<T>::sparse2dense(T identity, Descriptor* desc) {
  if (vec_type_ == GrB_DENSE)
    return GrB_SUCCESS;
  if (vec_type_ == GrB_UNKNOWN) {
    CHECK(setStorage(GrB_DENSE));
    return GrB_SUCCESS;
  }

  if (desc != NULL && desc->dirinfo())
    std::cout << "Converting from sparse to dense!\n";

  // 1. Initialize memory
  // 2. Call scatter

  CHECK(setStorage(GrB_DENSE));
  const int nt    = 128;
  const int nvals = sparse_.nvals_;

  if (desc == NULL || !desc->opreuse()) {
    dim3 NT, NB;
    NT.x = nt;
    NT.y = 1;
    NT.z = 1;
    NB.x = (nvals+nt-1)/nt;
    NB.y = 1;

    dense_.fill(identity);
    if (desc != NULL && desc->struconly())
      scatter<<<NB, NT>>>(dense_.d_val_, sparse_.d_ind_, (T)1, nvals);
    else
      scatter<<<NB, NT>>>(dense_.d_val_, sparse_.d_ind_, sparse_.d_val_, nvals);
  }

  vec_type_           = GrB_DENSE;
  dense_.need_update_ = true;
  dense_.nnz_         = nvals;

  return GrB_SUCCESS;
}

template <typename T>
Info Vector<T>::dense2sparse(T identity, Descriptor* desc) {
  if (vec_type_ == GrB_SPARSE) return GrB_INVALID_OBJECT;

  if (desc->dirinfo())
    std::cout << "Converting from dense to sparse!\n";

  // 1. Initialize memory
  // 2. Run kernel

  CHECK(setStorage(GrB_DENSE));
  const int nt    = 128;
  const int nvals = dense_.nvals_;

  dim3 NT, NB;
  NT.x = nt;
  NT.y = 1;
  NT.z = 1;
  NB.x = (nvals+nt-1)/nt;
  NB.y = 1;

  desc->resize((2*nvals)*std::max(sizeof(Index), sizeof(T)), "buffer");
  Index* d_flag = reinterpret_cast<Index*>(desc->d_buffer_);
  Index* d_scan = reinterpret_cast<Index*>(desc->d_buffer_)+nvals;

  updateFlagKernel<<<NB, NT>>>(d_flag, identity, dense_.d_val_, nvals);
  mgpu::Scan<mgpu::MgpuScanTypeExc>(d_flag, nvals, (Index)0,
      mgpu::plus<Index>(),  // NOLINT(build/include_what_you_use)
      reinterpret_cast<Index*>(0), &sparse_.nvals_, d_scan,
      *(desc->d_context_));

  if (desc->debug()) {
    printDevice("d_val",  dense_.d_val_, nvals);
    printDevice("d_flag", d_flag, nvals);
    printDevice("d_scan", d_scan, nvals);
    std::cout << "Dense frontier size: " << nvals << std::endl;
    std::cout << "Sparse frontier size: " << sparse_.nvals_ << std::endl;
  }

  LoadBalanceMode mxv_mode = getEnv("GRB_LOAD_BALANCE_MODE",
      GrB_LOAD_BALANCE_MERGE);

  if (desc->struconly() && mxv_mode == GrB_LOAD_BALANCE_MERGE)
    streamCompactDenseKernel<<<NB, NT>>>(sparse_.d_ind_, d_scan, (Index)1,
        d_flag, nvals);
  else
    streamCompactDenseKernel<<<NB, NT>>>(sparse_.d_ind_,
        sparse_.d_val_, d_scan, (T)identity, dense_.d_val_, nvals);

  if (desc->debug()) {
    printDevice("sparse_ind", sparse_.d_ind_, sparse_.nvals_);
    if (!desc->struconly())
      printDevice("sparse_val", sparse_.d_val_, sparse_.nvals_);
  }

  vec_type_ = GrB_SPARSE;
  sparse_.need_update_ = true;

  return GrB_SUCCESS;
}

// Assume both are of the same type to make things easier
template <typename T>
Info Vector<T>::swap(Vector* rhs) {  // NOLINT(build/include_what_you_use)
  if (vec_type_ != rhs->vec_type_ || vec_type_ == GrB_UNKNOWN)  {
    // std::cout << vec_type_ << " != " << rhs->vec_type_ << std::endl;
    // std::cout << "Error: Format not equivalent!\n";
    return GrB_INVALID_OBJECT;
  }

  if (vec_type_ == GrB_SPARSE) CHECK(sparse_.swap(&rhs->sparse_));
  else if (vec_type_ == GrB_DENSE) CHECK(dense_.swap(&rhs->dense_));

  Index temp_nsize = nsize_;
  Index temp_nvals = nvals_;
  float temp_ratio = ratio_;
  nsize_ = rhs->nsize_;
  nvals_ = rhs->nvals_;
  ratio_ = rhs->ratio_;
  rhs->nsize_ = temp_nsize;
  rhs->nvals_ = temp_nvals;
  rhs->ratio_ = temp_ratio;

  return GrB_SUCCESS;
}
}  // namespace backend
}  // namespace graphblas

#endif  // GRAPHBLAS_BACKEND_CUDA_VECTOR_HPP_
