#ifndef GRAPHBLAS_BACKEND_CUDA_ASSIGN_HPP_
#define GRAPHBLAS_BACKEND_CUDA_ASSIGN_HPP_

#include <iostream>
#include <vector>

#include "graphblas/backend/cuda/kernels/kernels.hpp"

namespace graphblas {
namespace backend {

template <typename W, typename T, typename M,
          typename BinaryOpT>
Info assignDense(DenseVector<W>*      w,
                 Vector<M>*           mask,
                 BinaryOpT            accum,
                 T                    val,
                 const Vector<Index>* indices,
                 Index                nindices,
                 Descriptor*          desc) {
  // Get descriptor parameters for SCMP, REPL, TRAN
  Desc_value scmp_mode, repl_mode;
  CHECK(desc->get(GrB_MASK, &scmp_mode));
  CHECK(desc->get(GrB_OUTP, &repl_mode));

  // TODO(@ctcyang): add accum and replace support
  // -have masked variants as separate kernel
  // -accum and replace as parts in flow
  // -no need to copy indices from cpuToGpu if user selected all indices
  bool use_mask = (mask != NULL);
  bool use_accum= (accum != NULL);
  bool use_all  = (indices == NULL);
  bool use_scmp = (scmp_mode == GrB_SCMP);
  bool use_repl = (repl_mode == GrB_REPLACE);

  if (desc->debug()) {
    std::cout << "Executing assignDense\n";
    printState(use_mask, use_accum, use_scmp, use_repl, false);
  }

  Index* indices_t = NULL;
  if (!use_all && nindices > 0) {
    desc->resize(nindices*sizeof(Index), "buffer");
    indices_t = reinterpret_cast<Index*>(desc->d_buffer_);
    CUDA_CALL(cudaMemcpy(indices_t, indices, nindices*sizeof(Index),
        cudaMemcpyHostToDevice));
  }

  if (use_mask) {
    // Get descriptor parameters for nthreads
    Desc_value nt_mode;
    CHECK(desc->get(GrB_NT, &nt_mode));
    const int nt = static_cast<int>(nt_mode);
    dim3 NT, NB;
    NT.x = nt;
    NT.y = 1;
    NT.z = 1;
    NB.x = (w->nvals_+nt-1)/nt;
    NB.y = 1;
    NB.z = 1;

    // Mask type
    // 1) Dense mask
    // 2) Sparse mask
    // 3) Uninitialized
    Storage mask_vec_type;
    CHECK(mask->getStorage(&mask_vec_type));

    if (mask_vec_type == GrB_DENSE) {
      if (use_scmp)
        assignDenseDenseMaskedKernel<true, true, true><<<NB, NT>>>(w->d_val_,
            w->nvals_, (mask->dense_).d_val_, accum, (W)val, indices_t,
            nindices);
      else
        assignDenseDenseMaskedKernel<false, true, true><<<NB, NT>>>(w->d_val_,
            w->nvals_, (mask->dense_).d_val_, accum, (W)val, indices_t,
            nindices);
    } else if (mask_vec_type == GrB_SPARSE) {
      if (use_scmp)
        assignDenseSparseMaskedKernel<true, true, true><<<NB, NT>>>(
            w->d_val_, w->nvals_, (mask->sparse_).d_ind_,
            (mask->sparse_).nvals_, accum, (W)val, indices_t, nindices);
      else
        assignDenseSparseMaskedKernel<false, true, true><<<NB, NT>>>(
            w->d_val_, w->nvals_, (mask->sparse_).d_ind_,
            (mask->sparse_).nvals_, accum, (W)val, indices_t, nindices);

      if (desc->debug()) {
        printDevice("mask_ind", (mask->sparse_).d_ind_, mask->sparse_.nvals_);
      }
    } else {
      return GrB_UNINITIALIZED_OBJECT;
    }

    if (desc->debug()) {
      printDevice("mask_val", (mask->sparse_).d_val_, mask->sparse_.nvals_);
      printDevice("w_val", w->d_val_, w->nvals_);
    }
  } else {
    std::cout << "Unmasked DeVec Assign Constant\n";
    std::cout << "Error: Feature not implemented yet!\n";
  }
  w->need_update_ = true;
  return GrB_SUCCESS;
}

template <typename W, typename T, typename M,
          typename BinaryOpT>
Info assignSparse(SparseVector<W>*     w,
                  Vector<M>*           mask,
                  BinaryOpT            accum,
                  T                    val,
                  const Vector<Index>* indices,
                  Index                nindices,
                  Descriptor*          desc) {
  // Get descriptor parameters for SCMP, REPL
  Desc_value scmp_mode, repl_mode;
  CHECK(desc->get(GrB_MASK, &scmp_mode));
  CHECK(desc->get(GrB_OUTP, &repl_mode));

  std::string accum_type = typeid(accum).name();
  // TODO(@ctcyang): add accum and replace support
  // -have masked variants as separate kernel
  // -accum and replace as parts in flow
  // -special case of inverting GrB_SCMP since we are using it to zero out
  // values in GrB_assign instead of passing them through
  bool use_mask  = (mask != NULL);
  bool use_accum = (accum_type.size() > 1);
  bool use_scmp  = (scmp_mode == GrB_SCMP);
  bool use_allowdupl;

  if (desc->debug())
    std::cout << "SpVec Assign Constant\n";

  // temp_ind and temp_val need |V| memory for masked case, so just allocate
  // this much memory for now. TODO(@ctcyang): optimize for memory
  desc->resize((2*nindices)*std::max(sizeof(Index), sizeof(T)),
      "buffer");

  // Only difference between masked and unmasked versions if whether
  // eWiseMult() is called afterwards or not

  // =====Part 2: Computing memory usage=====
	// temp_ind and temp_val need |V| memory
	Index* temp_ind   = reinterpret_cast<Index*>(desc->d_buffer_);
	W*     temp_val   = reinterpret_cast<W*>(desc->d_buffer_)+nindices;
	Index  temp_nvals = 0;
  Index  w_nvals;
  w->nvals(&w_nvals);

	// Get descriptor parameters for nthreads
	Desc_value nt_mode;
	CHECK(desc->get(GrB_NT, &nt_mode));
	const int nt = static_cast<int>(nt_mode);
	dim3 NT, NB;
	NT.x = nt;
	NT.y = 1;
	NT.z = 1;
	NB.x = (w_nvals+nt-1)/nt;
	NB.y = 1;
	NB.z = 1;

	// Mask type
	// 1) Dense mask
	// 2) Sparse mask (TODO)
	// 3) Uninitialized
	Storage mask_vec_type;
	CHECK(mask->getStorage(&mask_vec_type));
  assert(mask->dense_.nvals_ >= temp_nvals);

  // =====Part 3: AssignSparse=====
  // If mask is sparse, use temporary workaround of converting it to dense
	if (mask_vec_type == GrB_SPARSE) {
    mask->convert(static_cast<M>(0), 0.3, desc);
    //mask->sparse2dense(static_cast<M>(0), desc);
	  CHECK(mask->getStorage(&mask_vec_type));
  }
	// For visited nodes, assign val (0.f) to vector
	// For GrB_DENSE mask, need to add parameter for mask_identity to user
	// Scott: this is not necessary. Checking castable to (bool)1 is enough
	if (mask_vec_type == GrB_DENSE) {
		if (use_scmp)
			assignSparseKernel<true, true, true><<<NB, NT>>>(w->d_ind_, w->d_val_,
					w_nvals, (mask->dense_).d_val_, NULL, static_cast<W>(val),
					reinterpret_cast<Index*>(NULL), nindices);
		else
			assignSparseKernel<false, true, true><<<NB, NT>>>(w->d_ind_, w->d_val_,
					w_nvals, (mask->dense_).d_val_, NULL, static_cast<W>(val),
					reinterpret_cast<Index*>(NULL), nindices);
	} else if (mask_vec_type == GrB_SPARSE) {
    // TODO(@ctcyang): Adding sparse mask may be faster than skipping it
    // altogether which is what is currently done
    if (desc->debug()) {
		  std::cout << "SpVec Assign Constant Sparse Mask\n";
		  std::cout << "Error: Feature not implemented yet!\n";
    }
	} else {
		return GrB_UNINITIALIZED_OBJECT;
	}

	if (desc->debug()) {
		printDevice("mask", (mask->dense_).d_val_, nindices);
		printDevice("w_ind", w->d_ind_, w_nvals);
		printDevice("w_val", w->d_val_, w_nvals);
	}

	// Prune vals (0.f's) from vector
	desc->resize((5*nindices)*std::max(sizeof(Index), sizeof(W)), "buffer");
	Index* d_flag = reinterpret_cast<Index*>(desc->d_buffer_)+2*nindices;
	Index* d_scan = reinterpret_cast<Index*>(desc->d_buffer_)+3*nindices;
	Index* d_temp = reinterpret_cast<Index*>(desc->d_buffer_)+4*nindices;

	updateFlagKernel<<<NB, NT>>>(d_flag, (W)val, w->d_val_, w_nvals);
	mgpu::ScanPrealloc<mgpu::MgpuScanTypeExc>(d_flag, w_nvals, (Index)0,
			mgpu::plus<Index>(),  // NOLINT(build/include_what_you_use)
			reinterpret_cast<Index*>(0), &temp_nvals, d_scan,
			d_temp, *(desc->d_context_));

	if (desc->debug()) {
		printDevice("d_flag", d_flag, w_nvals);
		printDevice("d_scan", d_scan, w_nvals);
		std::cout << "Pre-assign frontier size: " << w_nvals << std::endl;
		std::cout << "Frontier size: " << temp_nvals << std::endl;
	}

	streamCompactSparseKernel<<<NB, NT>>>(temp_ind, temp_val, d_scan, (W)val,
			w->d_ind_, w->d_val_, w_nvals);

  CUDA_CALL(cudaMemcpy(w->d_ind_, temp_ind, temp_nvals*sizeof(Index),
      cudaMemcpyDeviceToDevice));
  CUDA_CALL(cudaMemcpy(w->d_val_, temp_val, temp_nvals*sizeof(W),
      cudaMemcpyDeviceToDevice));

  w->nvals_ = temp_nvals;
	if (desc->debug()) {
		printDevice("w_ind", w->d_ind_, w->nvals_);
		printDevice("w_val", w->d_val_, w->nvals_);
  }
  w->need_update_ = true;
  return GrB_SUCCESS;
}
}  // namespace backend
}  // namespace graphblas

#endif  // GRAPHBLAS_BACKEND_CUDA_ASSIGN_HPP_
