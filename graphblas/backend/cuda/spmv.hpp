#ifndef GRAPHBLAS_BACKEND_CUDA_SPMV_HPP_
#define GRAPHBLAS_BACKEND_CUDA_SPMV_HPP_

#include <cuda.h>
#include <cusparse.h>

#include <moderngpu.cuh>
#include <cub/cub.cuh>

#include <iostream>
#include <string>

#include "graphblas/backend/cuda/kernels/kernels.hpp"

namespace graphblas {
namespace backend {

template <typename W, typename a, typename U, typename M,
          typename BinaryOpT,      typename SemiringT>
Info spmv(DenseVector<W>*        w,
          const Vector<M>*       mask,
          BinaryOpT              accum,
          SemiringT              op,
          const SparseMatrix<a>* A,
          const DenseVector<U>*  u,
          Descriptor*            desc) {
  // Get descriptor parameters for SCMP, REPL, TRAN
  Desc_value scmp_mode, repl_mode, inp0_mode, inp1_mode;
  CHECK(desc->get(GrB_MASK, &scmp_mode));
  CHECK(desc->get(GrB_OUTP, &repl_mode));
  CHECK(desc->get(GrB_INP0, &inp0_mode));
  CHECK(desc->get(GrB_INP1, &inp1_mode));

  std::string accum_type = typeid(accum).name();
  // TODO(@ctcyang): add accum and replace support
  // -have masked variants as separate kernel
  // -have scmp as template parameter
  // -accum and replace as parts in flow
  bool use_mask  = (mask != NULL);
  bool use_accum = (accum_type.size() > 1);
  bool use_scmp  = (scmp_mode == GrB_SCMP);
  bool use_repl  = (repl_mode == GrB_REPLACE);
  bool use_tran  = (inp0_mode == GrB_TRAN || inp1_mode == GrB_TRAN);

  if (desc->debug()) {
    std::cout << "Executing Spmv\n";
    printState(use_mask, use_accum, use_scmp, use_repl, use_tran);
  }

  // Transpose (default is CSR):
  const Index* A_csrRowPtr = (use_tran) ? A->d_cscColPtr_ : A->d_csrRowPtr_;
  const Index* A_csrColInd = (use_tran) ? A->d_cscRowInd_ : A->d_csrColInd_;
  const a*     A_csrVal    = (use_tran) ? A->d_cscVal_    : A->d_csrVal_;
  const Index  A_nrows     = (use_tran) ? A->ncols_       : A->nrows_;

  if (desc->debug()) {
    std::cout << "cscColPtr: " << A->d_cscColPtr_ << std::endl;
    std::cout << "cscRowInd: " << A->d_cscRowInd_ << std::endl;
    std::cout << "cscVal:    " << A->d_cscVal_    << std::endl;

    std::cout << "csrRowPtr: " << A->d_csrRowPtr_ << std::endl;
    std::cout << "csrColInd: " << A->d_csrColInd_ << std::endl;
    std::cout << "csrVal:    " << A->d_csrVal_    << std::endl;
  }

  // Get descriptor parameters for nthreads
  Desc_value ta_mode, tb_mode, nt_mode;
  CHECK(desc->get(GrB_TA, &ta_mode));
  CHECK(desc->get(GrB_TB, &tb_mode));
  CHECK(desc->get(GrB_NT, &nt_mode));

  const int ta = static_cast<int>(ta_mode);
  const int tb = static_cast<int>(tb_mode);
  const int nt = static_cast<int>(nt_mode);

  /*!
   * /brief atomicAdd() 3+5  = 8
   *        atomicSub() 3-5  =-2
   *        atomicMin() 3,5  = 3
   *        atomicMax() 3,5  = 5
   *        atomicOr()  3||5 = 1
   *        atomicXor() 3^^5 = 0
   */
  auto add_op = extractAdd(op);
  int functor = add_op(3, 5);

  if (desc->debug()) {
    std::cout << "Fused mask: " << desc->fusedmask() << std::endl;
    std::cout << "Functor:    " << functor << std::endl;
  }

  if (desc->struconly() && functor != 1)
    std::cout << "Warning: Using structure-only mode and not using logical or "
        << "semiring may result in unintended behaviour. Is this intended?\n";

  if (use_mask && desc->fusedmask() && functor == 1) {
    // Mask type
    // 1) Dense mask
    // 2) Sparse mask TODO(@ctcyang)
    // 3) Uninitialized
    Storage mask_vec_type;
    CHECK(mask->getStorage(&mask_vec_type));

    if (mask_vec_type == GrB_DENSE) {
      dim3 NT, NB;
      NT.x = nt;
      NT.y = 1;
      NT.z = 1;
      NB.x = (A_nrows+nt-1)/nt;
      NB.y = 1;
      NB.z = 1;

      int variant = 0;
      variant |= use_scmp          ? 4 : 0;
      variant |= desc->earlyexit() ? 2 : 0;
      variant |= desc->opreuse()   ? 1 : 0;

      switch (variant) {
        case 0:
          spmvDenseMaskedOrKernel<false, false, false><<<NB, NT>>>(
              w->d_val_, mask->dense_.d_val_, NULL, op.identity(),
              extractMul(op), extractAdd(op), A_nrows, A->nvals_,
              A_csrRowPtr, A_csrColInd, A_csrVal, u->d_val_);
          break;
        case 1:
          spmvDenseMaskedOrKernel<false, false, true><<<NB, NT>>>(
              w->d_val_, mask->dense_.d_val_, NULL, op.identity(),
              extractMul(op), extractAdd(op), A_nrows, A->nvals_,
              A_csrRowPtr, A_csrColInd, A_csrVal, u->d_val_);
          break;
        case 2:
          spmvDenseMaskedOrKernel<false, true, false><<<NB, NT>>>(
              w->d_val_, mask->dense_.d_val_, NULL, op.identity(),
              extractMul(op), extractAdd(op), A_nrows, A->nvals_,
              A_csrRowPtr, A_csrColInd, A_csrVal, u->d_val_);
          break;
        case 3:
          spmvDenseMaskedOrKernel<false, true, true><<<NB, NT>>>(
              w->d_val_, mask->dense_.d_val_, NULL, op.identity(),
              extractMul(op), extractAdd(op), A_nrows, A->nvals_,
              A_csrRowPtr, A_csrColInd, A_csrVal, u->d_val_);
          break;
        case 4:
          spmvDenseMaskedOrKernel<true, false, false><<<NB, NT>>>(
              w->d_val_, mask->dense_.d_val_, NULL, op.identity(),
              extractMul(op), extractAdd(op), A_nrows, A->nvals_,
              A_csrRowPtr, A_csrColInd, A_csrVal, u->d_val_);
          break;
        case 5:
          spmvDenseMaskedOrKernel<true, false, true><<<NB, NT>>>(
              w->d_val_, mask->dense_.d_val_, NULL, op.identity(),
              extractMul(op), extractAdd(op), A_nrows, A->nvals_,
              A_csrRowPtr, A_csrColInd, A_csrVal, u->d_val_);
          break;
        case 6:
          spmvDenseMaskedOrKernel<true, true, false><<<NB, NT>>>(
              w->d_val_, mask->dense_.d_val_, NULL, op.identity(),
              extractMul(op), extractAdd(op), A_nrows, A->nvals_,
              A_csrRowPtr, A_csrColInd, A_csrVal, u->d_val_);
          break;
        case 7:
          spmvDenseMaskedOrKernel<true, true, true><<<NB, NT>>>(
              w->d_val_, mask->dense_.d_val_, NULL, op.identity(),
              extractMul(op), extractAdd(op), A_nrows, A->nvals_,
              A_csrRowPtr, A_csrColInd, A_csrVal, u->d_val_);
          break;
        default:
          break;
      }
      if (desc->debug())
        printDevice("w_val", w->d_val_, A_nrows);
    } else if (mask_vec_type == GrB_SPARSE) {
      std::cout << "DeVec Sparse Mask logical_or Spmv\n";
      std::cout << "Error: Feature not implemented yet!\n";
    } else {
      return GrB_UNINITIALIZED_OBJECT;
    }
  } else {
    Index* w_ind;
    W*     w_val;

    if (use_accum) {
      CHECK(desc->resize(A_nrows*sizeof(W), "buffer"));
      w_val = reinterpret_cast<W*>(desc->d_buffer_);
    } else {
      w_val = w->d_val_;
    }
    mgpu::SpmvCsrBinary(A_csrVal, A_csrColInd, A->nvals_, A_csrRowPtr, A_nrows,
        u->d_val_, true, w_val, op.identity(), extractMul(op), extractAdd(op),
        *(desc->d_context_) );
    dim3 NT, NB;
    NT.x = nt;
    NT.y = 1;
    NT.z = 1;
    NB.x = (A_nrows+nt-1)/nt;
    NB.y = 1;
    NB.z = 1;
    w->nvals_ = u->nvals_;
    if (desc->debug()) {
      std::cout << w->nvals_ << " nnz in vector w\n";
      printDevice("w_val", w_val, A_nrows);
    }
    if (use_mask) {
      if (use_scmp)
        assignDenseDenseMaskedKernel<false, true, true><<<NB, NT>>>(
            w_val, w->nvals_, mask->dense_.d_val_, extractAdd(op),
            op.identity(), reinterpret_cast<Index*>(NULL), A_nrows);
      else
        assignDenseDenseMaskedKernel< true, true, true><<<NB, NT>>>(
            w_val, w->nvals_, mask->dense_.d_val_, extractAdd(op),
            op.identity(), reinterpret_cast<Index*>(NULL), A_nrows);
    }
    if (use_accum) {
      if (desc->debug()) {
        std::cout << "Doing eWiseAdd accumulate:\n";
        printDevice("w_val", w->d_val_, A_nrows);
      }
      eWiseAddDenseDenseKernel<<<NB, NT>>>(w->d_val_, NULL, extractAdd(op),
          w->d_val_, w_val, A_nrows);
    }

    if (desc->debug())
      printDevice("w_val", w->d_val_, A_nrows);
    // TODO(@ctcyang): add semiring inputs to CUB
    /*size_t temp_storage_bytes = 0;
    cub::DeviceSpmv::CsrMV(desc->d_temp_, temp_storage_bytes, A->d_csrVal_,
        A->d_csrRowPtr_, A->d_csrColInd_, u->d_val_, w->d_val_,
        A->nrows_, A->ncols_, A->nvals_, 1.f, op->identity());
    desc->resize( temp_storage_bytes, "temp" );
    cub::DeviceSpmv::CsrMV(desc->d_temp_, desc->d_temp_size_, A->d_csrVal_,
        A->d_csrRowPtr_, A->d_csrColInd_, u->d_val_, w->d_val_,
        A->nrows_, A->ncols_, A->nvals_, 1.f, op->identity());*/
  }
  w->need_update_ = true;
  return GrB_SUCCESS;
}
}  // namespace backend
}  // namespace graphblas

#endif  // GRAPHBLAS_BACKEND_CUDA_SPMV_HPP_
