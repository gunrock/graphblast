#ifndef GRAPHBLAS_BACKEND_CUDA_SPMSPV_HPP_
#define GRAPHBLAS_BACKEND_CUDA_SPMSPV_HPP_

#include <iostream>
#include <algorithm>
#include <string>

#include "graphblas/backend/cuda/kernels/kernels.hpp"

namespace graphblas {
namespace backend {

template <typename W, typename a, typename U, typename M,
          typename BinaryOpT, typename SemiringT>
Info spmspvMerge(SparseVector<W>*       w,
                 const Vector<M>*       mask,
                 BinaryOpT              accum,
                 SemiringT              op,
                 const SparseMatrix<a>* A,
                 const SparseVector<U>* u,
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
  // -accum and replace as parts in flow
  // -special case of inverting GrB_SCMP since we are using it to zero out
  // values in GrB_assign instead of passing them through
  bool use_mask  = (mask != NULL);
  bool use_accum = (accum_type.size() > 1);
  bool use_scmp  = (scmp_mode != GrB_SCMP);
  bool use_repl  = (repl_mode == GrB_REPLACE);
  bool use_tran  = (inp0_mode == GrB_TRAN || inp1_mode == GrB_TRAN);
  bool use_allowdupl;

  if (desc->debug()) {
    std::cout << "Executing Spmspv MERGE\n";
    if (desc->struconly())
      std::cout << "In structure only mode\n";
    else
      std::cout << "In key-value mode\n";
    printState(use_mask, use_accum, use_scmp, use_repl, use_tran);
  }

  // Transpose (default is CSC):
  const Index* A_csrRowPtr  = (!use_tran) ? A->d_cscColPtr_ : A->d_csrRowPtr_;
  const Index* A_csrColInd  = (!use_tran) ? A->d_cscRowInd_ : A->d_csrColInd_;
  const a*     A_csrVal     = (!use_tran) ? A->d_cscVal_    : A->d_csrVal_;
  const Index  A_nrows      = (!use_tran) ? A->ncols_       : A->nrows_;
  const Index* Ah_csrRowPtr = (!use_tran) ? A->h_cscColPtr_ : A->h_csrRowPtr_;

  // temp_ind and temp_val need |V| memory for masked case, so just allocate
  // this much memory for now. TODO(@ctcyang): optimize for memory
  int size = static_cast<float>(A->nvals_*desc->memusage()+1);
  if (desc->struconly())
    desc->resize((2*A_nrows+2*size)*std::max(sizeof(Index), sizeof(T)),
        "buffer");
  else
    desc->resize((2*A_nrows+4*size)*std::max(sizeof(Index), sizeof(T)),
        "buffer");

  // Only difference between masked and unmasked versions if whether
  // eWiseMult() is called afterwards or not
  if (use_mask) {
    // temp_ind and temp_val need |V| memory
    Index* temp_ind   = reinterpret_cast<Index*>(desc->d_buffer_);
    a*     temp_val   = reinterpret_cast<a*>(desc->d_buffer_+A_nrows);
    Index  temp_nvals = 0;

    spmspvApspieMerge(temp_ind, temp_val, &temp_nvals, NULL, op, A_nrows,
        A->nvals_, A_csrRowPtr, A_csrColInd, A_csrVal, u->d_ind_, u->d_val_,
        &u->nvals_, desc);

    if (temp_nvals == 0) {
      if (desc->debug())
        std::cout << "No neighbours!\n";
      w->nvals_       = 0;
      w->need_update_ = true;
      return GrB_SUCCESS;
    } else {
      if (desc->debug())
        std::cout << "temp_nvals: " << temp_nvals << std::endl;
    }

    // Get descriptor parameters for nthreads
    Desc_value nt_mode;
    CHECK(desc->get(GrB_NT, &nt_mode));
    const int nt = static_cast<int>(nt_mode);
    dim3 NT, NB;
    NT.x = nt;
    NT.y = 1;
    NT.z = 1;
    NB.x = (temp_nvals+nt-1)/nt;
    NB.y = 1;
    NB.z = 1;

    // Mask type
    // 1) Dense mask
    // 2) Sparse mask (TODO)
    // 3) Uninitialized
    Storage mask_vec_type;
    CHECK(mask->getStorage(&mask_vec_type));
    assert(mask->dense_.nvals_ >= temp_nvals);

    if (desc->struconly()) {
      if (!desc->sort()) {
        if (use_scmp)
          assignDenseDenseMaskedKernel<true, true, true><<<NB, NT>>>(temp_ind,
              temp_nvals, (mask->dense_).d_val_, NULL, (Index)0,
              reinterpret_cast<Index*>(NULL), A_nrows);
        else
          assignDenseDenseMaskedKernel<false, true, true><<<NB, NT>>>(temp_ind,
              temp_nvals, (mask->dense_).d_val_, NULL, (Index)0,
              reinterpret_cast<Index*>(NULL), A_nrows);

        if (desc->debug()) {
          printDevice("mask", (mask->dense_).d_val_, A_nrows);
          printDevice("temp_ind", temp_ind, A_nrows);
        }

        // Turn dense vector into sparse
        desc->resize((4*A_nrows)*std::max(sizeof(Index), sizeof(T)), "buffer");
        Index* d_scan = reinterpret_cast<Index*>(desc->d_buffer_+2*A_nrows);
        Index* d_temp = reinterpret_cast<Index*>(desc->d_buffer_+3*A_nrows);

        mgpu::ScanPrealloc<mgpu::MgpuScanTypeExc>(temp_ind, A_nrows,
            (Index)0, mgpu::plus<Index>(), reinterpret_cast<Index*>(0),
            &w->nvals_, d_scan, d_temp, *(desc->d_context_));

        if (desc->debug()) {
          printDevice("d_scan", d_scan, A_nrows);
          std::cout << "Pre-assign frontier size: " << temp_nvals <<std::endl;
          std::cout << "Frontier size: " << w->nvals_ << std::endl;
        }

        streamCompactDenseKernel<<<NB, NT>>>(w->d_ind_, d_scan, (W)1, temp_ind,
            A_nrows);

        if (desc->debug())
          printDevice("w_ind", w->d_ind_, w->nvals_);
      } else {
        // For visited nodes, assign 0.f to vector
        // For GrB_DENSE mask, need to add parameter for mask_identity to user
        // Scott: this is not necessary. Checking castable to (bool)1 is fine
        if (mask_vec_type == GrB_DENSE) {
          if (use_scmp)
            assignSparseKernel<true, true, true><<<NB, NT>>>(temp_ind,
              temp_nvals, (mask->dense_).d_val_, NULL, (Index)-1,
              reinterpret_cast<Index*>(NULL), A_nrows);
          else
            assignSparseKernel<false, true, true><<<NB, NT>>>(temp_ind,
              temp_nvals, (mask->dense_).d_val_, NULL, (Index)-1,
              reinterpret_cast<Index*>(NULL), A_nrows);
        } else if (mask_vec_type == GrB_SPARSE) {
          std::cout << "Spmspv Sparse Mask\n";
          std::cout << "Error: Feature not implemented yet!\n";
        } else {
          return GrB_UNINITIALIZED_OBJECT;
        }

        if (desc->debug()) {
          printDevice("mask", (mask->dense_).d_val_, A_nrows);
          printDevice("temp_ind", temp_ind, temp_nvals);
        }

        // Prune 0.f's from vector
        desc->resize((4*A_nrows)*std::max(sizeof(Index), sizeof(T)), "buffer");
        Index* d_flag = reinterpret_cast<Index*>(desc->d_buffer_+  A_nrows);
        Index* d_scan = reinterpret_cast<Index*>(desc->d_buffer_+2*A_nrows);
        Index* d_temp = reinterpret_cast<Index*>(desc->d_buffer_+3*A_nrows);

        updateFlagKernel<<<NB, NT>>>(d_flag, -1, temp_ind, temp_nvals);
        mgpu::ScanPrealloc<mgpu::MgpuScanTypeExc>(d_flag, temp_nvals, (Index)0,
            mgpu::plus<Index>(), reinterpret_cast<Index*>(0), &w->nvals_,
            d_scan, d_temp, *(desc->d_context_) );

        if (desc->debug()) {
          printDevice("d_flag", d_flag, temp_nvals);
          printDevice("d_scan", d_scan, temp_nvals);
          std::cout << "Pre-assign frontier size: " << temp_nvals << std::endl;
          std::cout << "Frontier size: " << w->nvals_ << std::endl;
        }

        streamCompactSparseKernel<<<NB, NT>>>(w->d_ind_, d_scan, (Index)1,
            temp_ind, d_flag, temp_nvals);

        if (desc->debug()) {
          printDevice("w_ind", w->d_ind_, w->nvals_);
        }
      }
    } else {
      // For visited nodes, assign 0.f to vector
      // For GrB_DENSE mask, need to add parameter for mask_identity to user
      // Scott: this is not necessary. Checking castable to (bool)1 is enough
      if (mask_vec_type == GrB_DENSE) {
        if (use_scmp)
          assignSparseKernel<true, true, true><<<NB, NT>>>(temp_ind, temp_val,
              temp_nvals, (mask->dense_).d_val_, NULL, (U)0.f,
              reinterpret_cast<Index*>(NULL), A_nrows);
        else
          assignSparseKernel<false, true, true><<<NB, NT>>>(temp_ind, temp_val,
              temp_nvals, (mask->dense_).d_val_, NULL, (U)0.f,
              reinterpret_cast<Index*>(NULL), A_nrows);
      } else if (mask_vec_type == GrB_SPARSE) {
        std::cout << "Spmspv Sparse Mask\n";
        std::cout << "Error: Feature not implemented yet!\n";
      } else {
        return GrB_UNINITIALIZED_OBJECT;
      }

      if (desc->debug()) {
        printDevice("mask", (mask->dense_).d_val_, A_nrows);
        printDevice("temp_ind", temp_ind, temp_nvals);
        printDevice("temp_val", temp_val, temp_nvals);
      }

      // Prune 0.f's from vector
      desc->resize((5*A_nrows)*std::max(sizeof(Index), sizeof(a)), "buffer");
      Index* d_flag = reinterpret_cast<Index*>(desc->d_buffer_+2*A_nrows);
      Index* d_scan = reinterpret_cast<Index*>(desc->d_buffer_+3*A_nrows);
      Index* d_temp = reinterpret_cast<Index*>(desc->d_buffer_+4*A_nrows);

      updateFlagKernel<<<NB, NT>>>(d_flag, (a)0, temp_val, temp_nvals);
      mgpu::ScanPrealloc<mgpu::MgpuScanTypeExc>(d_flag, temp_nvals, (Index)0,
          mgpu::plus<Index>(),  // NOLINT(build/include_what_you_use)
          reinterpret_cast<Index*>(0), &w->nvals_, d_scan,
          d_temp, *(desc->d_context_));

      if (desc->debug()) {
        printDevice("d_flag", d_flag, temp_nvals);
        printDevice("d_scan", d_scan, temp_nvals);
        std::cout << "Pre-assign frontier size: " << temp_nvals << std::endl;
        std::cout << "Frontier size: " << w->nvals_ << std::endl;
      }

      streamCompactSparseKernel<<<NB, NT>>>(w->d_ind_, w->d_val_, d_scan, (W)0,
          temp_ind, temp_val, temp_nvals);

      if (desc->debug()) {
        printDevice("w_ind", w->d_ind_, w->nvals_);
        printDevice("w_val", w->d_val_, w->nvals_);
      }
    }
  } else {
    spmspvApspieMerge(w->d_ind_, w->d_val_, &w->nvals_, NULL, op, A_nrows,
        A->nvals_, A_csrRowPtr, A_csrColInd, A_csrVal, u->d_ind_, u->d_val_,
        &u->nvals_, desc);
  }
  w->need_update_ = true;
  return GrB_SUCCESS;
}

template <typename W, typename a, typename U, typename M,
          typename BinaryOpT, typename SemiringT>
Info spmspvSimple(DenseVector<W>*        w,
                  const Vector<M>*       mask,
                  BinaryOpT              accum,
                  SemiringT              op,
                  const SparseMatrix<a>* A,
                  const SparseVector<U>* u,
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
  // -accum and replace as parts in flow
  bool use_mask  = (mask != NULL);
  bool use_accum = (accum_type.size() > 1);
  bool use_scmp  = (scmp_mode == GrB_SCMP);
  bool use_repl  = (repl_mode == GrB_REPLACE);
  bool use_tran  = (inp0_mode == GrB_TRAN || inp1_mode == GrB_TRAN);
  bool use_allowdupl;

  if (desc->debug()) {
    std::cout << "Executing Spmspv SIMPLE\n";
    if (desc->struconly())
      std::cout << "In structure only mode\n";
    else
      std::cout << "In key-value mode\n";
    printState(use_mask, use_accum, use_scmp, use_repl, use_tran);
  }

  // Transpose (default is CSC):
  const Index* A_csrRowPtr  = (!use_tran) ? A->d_cscColPtr_ : A->d_csrRowPtr_;
  const Index* A_csrColInd  = (!use_tran) ? A->d_cscRowInd_ : A->d_csrColInd_;
  const a*     A_csrVal     = (!use_tran) ? A->d_cscVal_    : A->d_csrVal_;
  const Index  A_nrows      = (!use_tran) ? A->ncols_       : A->nrows_;
  const Index* Ah_csrRowPtr = (!use_tran) ? A->h_cscColPtr_ : A->h_csrRowPtr_;

  // Get number of nonzeroes in vector u
  Index u_nvals;
  CHECK(u->nvals(&u_nvals));

  // Get descriptor parameters for nthreads
  Desc_value nt_mode;
  CHECK(desc->get(GrB_NT, &nt_mode));
  const int nt = static_cast<int>(nt_mode);
  dim3 NT, NB;
  NT.x = nt;
  NT.y = 1;
  NT.z = 1;
  NB.x = (u_nvals+nt-1)/nt;
  NB.y = 1;
  NB.z = 1;

  if (desc->debug()) {
    printDevice("u_ind", u->d_ind_, u_nvals);
    printDevice("u_val", u->d_val_, u_nvals);
  }

  if (!use_accum)
    zeroKernel<<<NB, NT>>>(w->d_val_, op.identity(), A_nrows);

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
  if (desc->debug())
    std::cout << "Functor: " << functor << "\n";
  if (functor == 8) {
    spmspvSimpleAddKernel<<<NB, NT>>>(w->d_val_, NULL, op.identity(),
        extractMul(op), A_csrRowPtr, A_csrColInd, A_csrVal, u->d_ind_,
        u->d_val_, u_nvals);
    if (desc->debug())
      std::cout << "Using atomicAdd!\n";
  } else if (functor == 1 && !desc->atomic()) {
    spmspvSimpleOrKernel<<<NB, NT>>>(w->d_val_, NULL, op.identity(),
        extractMul(op), A_csrRowPtr, A_csrColInd, A_csrVal, u->d_ind_,
        u->d_val_, u_nvals);
    if (desc->debug())
      std::cout << "Using Boolean or!\n";
  } else {
    spmspvSimpleKernel<<<NB, NT>>>(w->d_val_, NULL, op.identity(),
        extractMul(op), add_op, A_csrRowPtr, A_csrColInd, A_csrVal,
        u->d_ind_, u->d_val_, u_nvals);
    if (desc->debug())
      std::cout << "Using atomic operation!\n";
  }

  if (desc->debug())
    printDevice("w_val", w->d_val_, A_nrows);

  // Run mask kernel to filter stuff out
  if (use_mask) {
    Storage mask_vec_type;
    CHECK(mask->getStorage(&mask_vec_type));
    if (mask_vec_type == GrB_SPARSE) {
      std::cout << "Error: Simple kernel sparse mask not implemented yet!\n";
      return GrB_INVALID_OBJECT;
    } else if (mask_vec_type == GrB_DENSE) {
      if (use_scmp)
        setDenseMaskKernel<true><<<NB, NT>>>(w->d_val_, (mask->dense_).d_val_,
            op.identity(), A_nrows);
      else
        setDenseMaskKernel<false><<<NB, NT>>>(w->d_val_, (mask->dense_).d_val_,
            op.identity(), A_nrows);
    } else {
      std::cout << "Error: Mask is selected, but not initialized!\n";
      return GrB_INVALID_OBJECT;
    }
  }

  w->nvals_ = A_nrows;
  w->need_update_ = true;
  return GrB_SUCCESS;
}
}  // namespace backend
}  // namespace graphblas

#endif  // GRAPHBLAS_BACKEND_CUDA_SPMSPV_HPP_
