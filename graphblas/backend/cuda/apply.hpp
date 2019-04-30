#ifndef GRAPHBLAS_BACKEND_CUDA_APPLY_HPP_
#define GRAPHBLAS_BACKEND_CUDA_APPLY_HPP_

#include <iostream>

namespace graphblas {
namespace backend {

template <typename U, typename W, typename M,
          typename BinaryOpT, typename UnaryOpT>
Info applyDense(DenseVector<W>*  w,
                const Vector<M>* mask,
                BinaryOpT        accum,
                UnaryOpT         op,
                DenseVector<U>*  u,
                Descriptor*      desc) {
  // Get descriptor parameters for SCMP, REPL, BACKEND
  Desc_value scmp_mode, repl_mode, backend;
  CHECK(desc->get(GrB_MASK, &scmp_mode));
  CHECK(desc->get(GrB_OUTP, &repl_mode));
  CHECK(desc->get(GrB_BACKEND, &backend));

  // TODO(@ctcyang): add accum and replace support
  bool use_mask  = (mask != NULL);
  bool use_accum = (accum != NULL);
  bool use_scmp  = (scmp_mode == GrB_SCMP);
  bool use_repl  = (repl_mode == GrB_REPLACE);

  if (desc->debug()) {
    std::cout << "Executing applyDense\n";
    printState(use_mask, use_accum, use_scmp, use_repl, false);
  }

  if (backend == GrB_SEQUENTIAL) {
    if (use_mask) {
      std::cout << "Error: DeVec apply masked not implemented yet!\n";
    } else {
      CHECK(u->gpuToCpu());
      for (Index i = 0; i < u->nvals_; ++i)
        w->h_val_[i] = op(u->h_val_[i]);
      CHECK(w->cpuToGpu());
    }
  } else {
    std::cout << "DeVec apply CPU\n";
    std::cout << "Error: Feature not implemented yet!\n";
  }

  return GrB_SUCCESS;
}

template <typename U, typename W, typename M,
          typename BinaryOpT, typename UnaryOpT>
Info applySparse(SparseVector<W>* w,
                 const Vector<M>* mask,
                 BinaryOpT        accum,
                 UnaryOpT         op,
                 SparseVector<U>* u,
                 Descriptor*      desc) {
  std::cout << "SpVec Apply\n";
  std::cout << "Error: Feature not implemented yet!\n";
  return GrB_SUCCESS;
}

template <typename a, typename c, typename m,
          typename BinaryOpT, typename UnaryOpT>
Info applyDense(DenseMatrix<c>*  C,
                const Matrix<m>* mask,
                BinaryOpT        accum,
                UnaryOpT         op,
                DenseMatrix<a>*  A,
                Descriptor*      desc) {
  std::cout << "DeMat Apply\n";
  std::cout << "Error: Feature not implemented yet!\n";
  return GrB_SUCCESS;
}

template <typename a, typename c, typename m,
          typename BinaryOpT, typename UnaryOpT>
Info applySparse(SparseMatrix<c>* C,
                 const Matrix<m>* mask,
                 BinaryOpT        accum,
                 UnaryOpT         op,
                 SparseMatrix<a>* A,
                 Descriptor*      desc) {
  // Get descriptor parameters for SCMP, REPL, BACKEND
  Desc_value scmp_mode, repl_mode, backend;
  CHECK(desc->get(GrB_MASK, &scmp_mode));
  CHECK(desc->get(GrB_OUTP, &repl_mode));
  CHECK(desc->get(GrB_BACKEND, &backend));

  // TODO(@ctcyang): add accum and replace support
  bool use_mask  = (mask != NULL);
  bool use_accum = (accum != NULL);
  bool use_scmp  = (scmp_mode == GrB_SCMP);
  bool use_repl  = (repl_mode == GrB_REPLACE);

  if (desc->debug()) {
    std::cout << "Executing applySparse\n";
    printState(use_mask, use_accum, use_scmp, use_repl, false);
  }

  if (backend == GrB_SEQUENTIAL) {
    if (use_mask) {
      std::cout << "Error: SpMat apply masked not implemented yet!\n";
    } else {
      CHECK(A->gpuToCpu());
      if (!A->symmetric_) {
        std::cout << "Error: SpMat apply for non-symmetric matrices not";
        std::cout << "implemented yet!\n";
      }
      for (Index i = 0; i < A->nvals_; ++i)
        C->h_csrVal_[i] = op(A->h_csrVal_[i]);
      CHECK(C->cpuToGpu());
    }
  } else {
    std::cout << "DeVec apply CPU\n";
    std::cout << "Error: Feature not implemented yet!\n";
  }
  return GrB_SUCCESS;
}
}  // namespace backend
}  // namespace graphblas

#endif  // GRAPHBLAS_BACKEND_CUDA_APPLY_HPP_
