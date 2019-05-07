#ifndef GRAPHBLAS_BACKEND_CUDA_EWISEADD_HPP_
#define GRAPHBLAS_BACKEND_CUDA_EWISEADD_HPP_

#include <iostream>
#include <string>

#include "graphblas/backend/cuda/kernels/kernels.hpp"

namespace graphblas {
namespace backend {
/*
 * \brief 4 vector variants
 */

// Sparse x sparse vector
template <typename W, typename U, typename V, typename M,
          typename BinaryOpT,     typename SemiringT>
Info eWiseAddInner(DenseVector<W>*       w,
                   const Vector<M>*       mask,
                   BinaryOpT              accum,
                   SemiringT              op,
                   const SparseVector<U>* u,
                   const SparseVector<V>* v,
                   Descriptor*            desc) {
  std::cout << "Error: eWiseAdd sparse-sparse not implemented yet!\n";
  return GrB_SUCCESS;
}

// Dense x dense vector
template <typename W, typename U, typename V, typename M,
          typename BinaryOpT,     typename SemiringT>
Info eWiseAddInner(DenseVector<W>*       w,
                   const Vector<M>*      mask,
                   BinaryOpT             accum,
                   SemiringT             op,
                   const DenseVector<U>* u,
                   const DenseVector<V>* v,
                   Descriptor*           desc) {
  // Get descriptor parameters for SCMP, REPL
  Desc_value scmp_mode, repl_mode;
  CHECK(desc->get(GrB_MASK, &scmp_mode));
  CHECK(desc->get(GrB_OUTP, &repl_mode));

  std::string accum_type = typeid(accum).name();
  // TODO(@ctcyang): add accum and replace support
  // -have masked variants as separate kernel
  // -have scmp as template parameter
  // -accum and replace as parts in flow
  bool use_mask  = (mask != NULL);
  bool use_accum = (accum_type.size() > 1);
  bool use_scmp  = (scmp_mode == GrB_SCMP);
  bool use_repl  = (repl_mode == GrB_REPLACE);

  if (desc->debug()) {
    std::cout << "Executing eWiseAdd dense-dense\n";
    printState(use_mask, use_accum, use_scmp, use_repl, 0);
  }

  // Get descriptor parameters for nthreads
  Desc_value nt_mode;
  CHECK(desc->get(GrB_NT, &nt_mode));
  const int nt = static_cast<int>(nt_mode);

  // Get number of elements
  Index u_nvals;
  u->nvals(&u_nvals);

  DenseVector<U>* u_t = const_cast<DenseVector<U>*>(u);
  DenseVector<V>* v_t = const_cast<DenseVector<V>*>(v);

  if (use_mask) {
    std::cout << "Error: Masked eWiseAdd dense-dense not implemented yet!\n";
  } else {
    dim3 NT, NB;
    NT.x = nt;
    NT.y = 1;
    NT.z = 1;
    NB.x = (u_nvals + nt - 1) / nt;
    NB.y = 1;
    NB.z = 1;

    eWiseAddDenseDenseKernel<<<NB, NT>>>(w->d_val_, NULL, extractAdd(op),
        u_t->d_val_, v_t->d_val_, u_nvals);
  }
  w->need_update_ = true;

  return GrB_SUCCESS;
}

// Sparse x dense vector
template <typename W, typename U, typename V, typename M,
          typename BinaryOpT,     typename SemiringT>
Info eWiseAddInner(DenseVector<W>*        w,
                   const Vector<M>*       mask,
                   BinaryOpT              accum,
                   SemiringT              op,
                   const SparseVector<U>* u,
                   const DenseVector<V>*  v,
                   bool                   reverse,
                   Descriptor*            desc) {
  // Get descriptor parameters for SCMP, REPL
  Desc_value scmp_mode, repl_mode;
  CHECK(desc->get(GrB_MASK, &scmp_mode));
  CHECK(desc->get(GrB_OUTP, &repl_mode));

  std::string accum_type = typeid(accum).name();
  // TODO(@ctcyang): add accum and replace support
  // -have masked variants as separate kernel
  // -have scmp as template parameter
  // -accum and replace as parts in flow
  bool use_mask  = (mask != NULL);
  bool use_accum = (accum_type.size() > 1);
  bool use_scmp  = (scmp_mode == GrB_SCMP);
  bool use_repl  = (repl_mode == GrB_REPLACE);

  if (desc->debug()) {
    std::cout << "Executing eWiseAdd sparse-dense\n";
    printState(use_mask, use_accum, use_scmp, use_repl, 0);
  }

  // Get descriptor parameters for nthreads
  Desc_value nt_mode;
  CHECK(desc->get(GrB_NT, &nt_mode));
  const int nt = static_cast<int>(nt_mode);

  // Get number of elements
  Index u_nvals;
  Index v_nvals;
  u->nvals(&u_nvals);
  v->nvals(&v_nvals);

  if (use_mask) {
    std::cout << "Error: Masked eWiseAdd sparse-dense not implemented yet!\n";
  } else {
    if (v != w)
      w->dup(v);

    dim3 NT, NB;
    NT.x = nt;
    NT.y = 1;
    NT.z = 1;
    NB.x = (v_nvals + nt - 1) / nt;
    NB.y = 1;
    NB.z = 1;

    // Need to consider cases where op(a,b) != op (b,a) e.g. LessMonoid
    eWiseAddDenseConstantKernel<<<NB, NT>>>(w->d_val_, extractAdd(op),
        op.identity(), reverse, v_nvals);

    NB.x = (u_nvals + nt - 1) / nt;
    eWiseAddSparseDenseKernel<<<NB, NT>>>(w->d_val_, NULL, extractAdd(op),
        u->d_ind_, u->d_val_, v->d_val_, u_nvals);
  }
  w->need_update_ = true;
  return GrB_SUCCESS;
}

// Sparse vector x Broadcast scalar (no mask)
template <typename W, typename U, typename V, typename M,
          typename BinaryOpT,     typename SemiringT>
Info eWiseAddInner(DenseVector<W>*        w,
                   const Vector<M>*       mask,
                   BinaryOpT              accum,
                   SemiringT              op,
                   const SparseVector<U>* u,
                   V                      val,
                   Descriptor*            desc) {
  // Get descriptor parameters for SCMP, REPL
  Desc_value scmp_mode, repl_mode;
  CHECK(desc->get(GrB_MASK, &scmp_mode));
  CHECK(desc->get(GrB_OUTP, &repl_mode));

  std::string accum_type = typeid(accum).name();
  // TODO(@ctcyang): add accum and replace support
  // -have masked variants as separate kernel
  // -have scmp as template parameter
  // -accum and replace as parts in flow
  bool use_mask  = (mask != NULL);
  bool use_accum = (accum_type.size() > 1);
  bool use_scmp  = (scmp_mode == GrB_SCMP);
  bool use_repl  = (repl_mode == GrB_REPLACE);

  if (desc->debug()) {
    std::cout << "Executing eWiseMult sparse matrix-scalar\n";
    printState(use_mask, use_accum, use_scmp, use_repl, 0);
  }

  // Get descriptor parameters for nthreads
  Desc_value nt_mode;
  CHECK(desc->get(GrB_NT, &nt_mode));
  const int nt = static_cast<int>(nt_mode);

  // Get number of elements
  Index u_nvals;
  u->nvals(&u_nvals);

  if (use_mask) {
    std::cout << "eWiseMult Sparse Matrix Broadcast Scalar with Mask\n";
    std::cout << "Error: Feature not implemented yet!\n";
  } else {
    auto add_op = extractAdd(op);
    w->fill(add_op(op.identity(), val));

    dim3 NT, NB;
    NT.x = nt;
    NT.y = 1;
    NT.z = 1;
    NB.x = (u_nvals + nt - 1) / nt;
    NB.y = 1;
    NB.z = 1;

    eWiseAddSparseDenseKernel<<<NB, NT>>>(w->d_val_, NULL, extractAdd(op),
        u->d_ind_, u->d_val_, w->d_val_, u_nvals);
  }
  w->need_update_ = true;

  return GrB_SUCCESS;
}

// Dense vector x Broadcast scalar (no mask)
template <typename W, typename U, typename V, typename M,
          typename BinaryOpT,     typename SemiringT>
Info eWiseAddInner(DenseVector<W>*       w,
                   const Vector<M>*      mask,
                   BinaryOpT             accum,
                   SemiringT             op,
                   const DenseVector<U>* u,
                   V                     val,
                   Descriptor*           desc) {
  // Get descriptor parameters for SCMP, REPL
  Desc_value scmp_mode, repl_mode;
  CHECK(desc->get(GrB_MASK, &scmp_mode));
  CHECK(desc->get(GrB_OUTP, &repl_mode));

  std::string accum_type = typeid(accum).name();
  // TODO(@ctcyang): add accum and replace support
  // -have masked variants as separate kernel
  // -have scmp as template parameter
  // -accum and replace as parts in flow
  bool use_mask  = (mask != NULL);
  bool use_accum = (accum_type.size() > 1);
  bool use_scmp  = (scmp_mode == GrB_SCMP);
  bool use_repl  = (repl_mode == GrB_REPLACE);

  if (desc->debug()) {
    std::cout << "Executing eWiseMult sparse matrix-scalar\n";
    printState(use_mask, use_accum, use_scmp, use_repl, 0);
  }

  // Get descriptor parameters for nthreads
  Desc_value nt_mode;
  CHECK(desc->get(GrB_NT, &nt_mode));
  const int nt = static_cast<int>(nt_mode);

  // Get number of elements
  Index u_size;
  u->size(&u_size);

  if (use_mask) {
    std::cout << "eWiseMult Sparse Matrix Broadcast Scalar with Mask\n";
    std::cout << "Error: Feature not implemented yet!\n";
  } else {
    if (u != w)
      CHECK(w->dup(u));

    dim3 NT, NB;
    NT.x = nt;
    NT.y = 1;
    NT.z = 1;
    NB.x = (u_size + nt - 1) / nt;
    NB.y = 1;
    NB.z = 1;

    eWiseMultKernel<<<NB, NT>>>(w->d_val_, NULL, extractAdd(op),
        w->d_val_, u_size, val);
  }
  w->need_update_ = true;

  return GrB_SUCCESS;
}
}  // namespace backend
}  // namespace graphblas

#endif  // GRAPHBLAS_BACKEND_CUDA_EWISEADD_HPP_
