#ifndef GRAPHBLAS_BACKEND_CUDA_OPERATIONS_HPP_
#define GRAPHBLAS_BACKEND_CUDA_OPERATIONS_HPP_

#include <vector>

namespace graphblas {
namespace backend {

template <typename T>
class Vector;

template <typename T>
class Matrix;

template <typename c, typename a, typename b, typename m,
          typename BinaryOpT,     typename SemiringT>
Info mxm(Matrix<c>*       C,
         const Matrix<a>* mask,
         BinaryOpT        accum,
         SemiringT        op,
         const Matrix<a>* A,
         const Matrix<b>* B,
         Descriptor*      desc) {
  Matrix<a>* A_t = const_cast<Matrix<a>*>(A);
  Matrix<b>* B_t = const_cast<Matrix<b>*>(B);

  if (desc->debug()) {
    std::cout << "===Begin mxm===\n";
    CHECK(A_t->print());
    CHECK(B_t->print());
  }

  Storage A_mat_type;
  Storage B_mat_type;
  CHECK(A->getStorage(&A_mat_type));
  CHECK(B->getStorage(&B_mat_type));

  if (A_mat_type == GrB_SPARSE && B_mat_type == GrB_SPARSE) {
    CHECK(C->setStorage(GrB_SPARSE));
    CHECK(cusparse_spgemm2(&C->sparse_, mask, accum, op, &A->sparse_,
        &B->sparse_, desc));
  } else {
    std::cout << "Error: SpMM and GEMM not implemented yet!\n";
    /*CHECK( C->setStorage( GrB_DENSE ) );
    if( A_mat_type==GrB_SPARSE && B_mat_type==GrB_DENSE )
    {
      CHECK( spmm( &C->dense_, mask, accum, op, &A->sparse_, 
          &B->dense_, desc ) );
    }
    else if( A_mat_type==GrB_DENSE && B_mat_type==GrB_SPARSE )
    {
      CHECK( spmm( &C->dense_, mask, accum, op, &A->dense_, 
          &B->sparse_, desc ) );
    }
    else
    {
      CHECK( gemm( &C->dense_, mask, accum, op, &A->dense_, 
          &B->dense_, desc ) );
    }*/
  }

  if (desc->debug()) {
    std::cout << "===End mxm===\n";
    CHECK(C->print());
  }
  return GrB_SUCCESS;
}

template <typename W, typename U, typename a, typename M,
          typename BinaryOpT, typename SemiringT>
Info vxm(Vector<W>*       w,
         const Vector<M>* mask,
         BinaryOpT        accum,
         SemiringT        op,
         const Vector<U>* u,
         const Matrix<a>* A,
         Descriptor*      desc) {
  Vector<U>* u_t = const_cast<Vector<U>*>(u);

  if (desc->debug()) {
    std::cout << "===Begin vxm===\n";
    CHECK(u_t->print());
  }

  // Get storage
  Storage u_vec_type;
  Storage A_mat_type;
  CHECK(u->getStorage(&u_vec_type));
  CHECK(A->getStorage(&A_mat_type));

  // Transpose
  Desc_value inp0_mode;
  CHECK(desc->get(GrB_INP0, &inp0_mode));
  if (inp0_mode != GrB_DEFAULT) return GrB_INVALID_VALUE;

  // Treat vxm as an mxv with transposed matrix
  CHECK(desc->toggle(GrB_INP1));

  LoadBalanceMode lb_mode = getEnv("GRB_LOAD_BALANCE_MODE",
      GrB_LOAD_BALANCE_MERGE);

  // Conversions
  // TODO(@ctcyang): add tol
  SparseMatrixFormat A_format;
  bool A_symmetric;
  CHECK(A->getFormat(&A_format));
  CHECK(A->getSymmetry(&A_symmetric));

  Desc_value vxm_mode, tol;
  CHECK(desc->get(GrB_MXVMODE, &vxm_mode));
  CHECK(desc->get(GrB_TOL,     &tol));
  if (desc->debug()) {
    std::cout << "Load balance mode: " << lb_mode << std::endl;
    std::cout << "Identity: " << op.identity() << std::endl;
    std::cout << "Sparse format: " << A_format << std::endl;
    std::cout << "Symmetric: " << A_symmetric << std::endl;
  }

  // Fallback for lacking CSC storage overrides any mxvmode selections
  if (!A_symmetric && A_format == GrB_SPARSE_MATRIX_CSRONLY) {
    if (u_vec_type == GrB_DENSE)
      CHECK(u_t->dense2sparse(op.identity(), desc));
  } else if (vxm_mode == GrB_PUSHPULL) {
    CHECK(u_t->convert(op.identity(), desc->switchpoint(), desc));
  } else if (vxm_mode == GrB_PUSHONLY && u_vec_type == GrB_DENSE) {
    CHECK(u_t->dense2sparse(op.identity(), desc));
  } else if (vxm_mode == GrB_PULLONLY && u_vec_type == GrB_SPARSE) {
    CHECK(u_t->sparse2dense(op.identity(), desc));
  }

  // Check if vector type was changed due to conversion!
  CHECK(u->getStorage(&u_vec_type));

  if (desc->debug())
    std::cout << "u_vec_type: " << u_vec_type << std::endl;

  // Breakdown into 4 cases:
  // 1) SpMSpV: SpMat x SpVec
  // 2) SpMV:   SpMat x DeVec (preferred to 3)
  // 3) SpMSpV: SpMat x SpVec (fallback if CSC representation not available)
  // 4) GeMV:   DeMat x DeVec
  //
  // Note: differs from mxv, because mxv would say instead:
  // 3) "... if CSC representation not available ..."
  if (A_mat_type == GrB_SPARSE && u_vec_type == GrB_SPARSE) {
    if (lb_mode == GrB_LOAD_BALANCE_SIMPLE ||
        lb_mode == GrB_LOAD_BALANCE_TWC) {
      CHECK(w->setStorage(GrB_DENSE));
      // 1a) Simple SpMSpV no load-balancing codepath
      if (lb_mode == GrB_LOAD_BALANCE_SIMPLE)
        std::cout << "Simple SPMSPV not implemented yet!\n";
        // CHECK( spmspvSimple(&w->dense_, mask, accum, op, &A->sparse_,
        //     &u->sparse_, desc) );
      // 1b) Thread-warp-block (single kernel) codepath
      else if (lb_mode == GrB_LOAD_BALANCE_TWC)
        std::cout << "Error: B40C load-balance algorithm not implemented yet!\n";
      if (desc->debug()) {
        CHECK(w->getStorage(&u_vec_type));
        std::cout << "w_vec_type: " << u_vec_type << std::endl;
      }
      CHECK(w->dense2sparse(op.identity(), desc));
      if (desc->debug()) {
        CHECK(w->getStorage(&u_vec_type));
        std::cout << "w_vec_type: " << u_vec_type << std::endl;
      }
    // 1c) Merge-path (two-phase decomposition) codepath
    } else if (lb_mode == GrB_LOAD_BALANCE_MERGE) {
      CHECK(w->setStorage(GrB_SPARSE));
      CHECK(spmspvMerge(&w->sparse_, mask, accum, op, &A->sparse_,
          &u->sparse_, desc));
    } else {
      std::cout << "Error: Invalid load-balance algorithm!\n";
    }
    desc->lastmxv_ = GrB_PUSHONLY;
  } else {
    // TODO(@ctcyang): Some performance left on table, sparse2dense should
    // only convert rather than setStorage if accum is being used
    CHECK(w->setStorage(GrB_DENSE));
    // CHECK(w->sparse2dense(op.identity(), desc));
    if (A_mat_type == GrB_SPARSE)
      CHECK(spmv(&w->dense_, mask, accum, op, &A->sparse_, &u->dense_,
          desc));
    else
      CHECK(gemv(&w->dense_, mask, accum, op, &A->dense_, &u->dense_,
          desc));
    desc->lastmxv_ = GrB_PULLONLY;
  }

  // Undo change to desc by toggling again
  CHECK(desc->toggle(GrB_INP1));

  if (desc->debug()) {
    std::cout << "===End vxm===\n";
    CHECK(w->print());
  }
  return GrB_SUCCESS;
}

// Only difference between vxm and mxv is an additional check for gemv
// to transpose A
// -this is because w=uA is same as w=A^Tu
// -i.e. GraphBLAS treats 1xn Vector the same as nx1 Vector
template <typename W, typename a, typename U, typename M,
          typename BinaryOpT, typename SemiringT>
Info mxv(Vector<W>*       w,
         const Vector<M>* mask,
         BinaryOpT        accum,
         SemiringT        op,
         const Matrix<a>* A,
         const Vector<U>* u,
         Descriptor*      desc) {
  Vector<U>* u_t = const_cast<Vector<U>*>(u);

  if (desc->debug()) {
    std::cout << "===Begin mxv===\n";
    CHECK(u_t->print());
  }

  // Get storage:
  Storage u_vec_type;
  Storage A_mat_type;
  CHECK(u->getStorage(&u_vec_type));
  CHECK(A->getStorage(&A_mat_type));

  // Transpose:
  Desc_value inp1_mode;
  CHECK(desc->get(GrB_INP1, &inp1_mode));
  if (inp1_mode != GrB_DEFAULT) return GrB_INVALID_VALUE;

  LoadBalanceMode lb_mode = getEnv("GRB_LOAD_BALANCE_MODE",
      GrB_LOAD_BALANCE_MERGE);
  if (desc->debug())
    std::cout << "Load balance mode: " << lb_mode << std::endl;

  // Conversions:
  SparseMatrixFormat A_format;
  bool A_symmetric;
  CHECK(A->getFormat(&A_format));
  CHECK(A->getSymmetry(&A_symmetric));

  Desc_value mxv_mode, tol;
  CHECK(desc->get(GrB_MXVMODE, &mxv_mode));
  CHECK(desc->get(GrB_TOL,     &tol));

  // Fallback for lacking CSC storage overrides any mxvmode selections
  if (!A_symmetric && A_format == GrB_SPARSE_MATRIX_CSRONLY) {
    if (u_vec_type == GrB_SPARSE)
      CHECK(u_t->sparse2dense(op.identity(), desc));
  } else if (mxv_mode == GrB_PUSHPULL) {
    CHECK(u_t->convert(op.identity(), desc->switchpoint(), desc));
  } else if (mxv_mode == GrB_PUSHONLY && u_vec_type == GrB_DENSE) {
    CHECK(u_t->dense2sparse(op.identity(), desc));
  } else if (mxv_mode == GrB_PULLONLY && u_vec_type == GrB_SPARSE) {
    CHECK(u_t->sparse2dense(op.identity(), desc));
  }

  // Check if vector type was changed due to conversion!
  CHECK(u->getStorage(&u_vec_type));

  // 3 cases:
  // 1) SpMSpV: SpMat x SpVec (preferred to 3)
  // 2) SpMV:   SpMat x DeVec
  // 3) SpMV:   SpMat x DeVec (fallback if CSC representation not available)
  // 4) GeMV:   DeMat x DeVec
  if (A_mat_type == GrB_SPARSE && u_vec_type == GrB_SPARSE) {
    if (lb_mode == GrB_LOAD_BALANCE_SIMPLE ||
        lb_mode == GrB_LOAD_BALANCE_TWC) {
      CHECK(w->setStorage(GrB_DENSE));
      // 1a) Simple SpMSpV no load-balancing codepath
      if (lb_mode == GrB_LOAD_BALANCE_SIMPLE)
        std::cout << "Simple SPMSPV not implemented yet!\n";
        // CHECK( spmspvSimple(&w->dense_, mask, accum, op, &A->sparse_,
        //     &u->sparse_, desc) );
      // 1b) Thread-warp-block (single kernel) codepath
      else if (lb_mode == GrB_LOAD_BALANCE_TWC)
        std::cout << "Error: B40C load-balance algorithm not implemented yet!\n";
      if (desc->debug()) {
        CHECK(w->getStorage(&u_vec_type));
        std::cout << "w_vec_type: " << u_vec_type << std::endl;
      }
      CHECK(w->dense2sparse(op.identity(), desc));
      if (desc->debug()) {
        CHECK(w->getStorage(&u_vec_type));
        std::cout << "w_vec_type: " << u_vec_type << std::endl;
      }
    // 1c) Merge-path (two-phase decomposition) codepath
    } else if (lb_mode == GrB_LOAD_BALANCE_MERGE) {
      CHECK(w->setStorage(GrB_SPARSE));
      CHECK(spmspvMerge(&w->sparse_, mask, accum, op, &A->sparse_,
          &u->sparse_, desc));
    } else {
      std::cout << "Error: Invalid load-balance algorithm!\n";
    }
    desc->lastmxv_ = GrB_PUSHONLY;
  } else {
    CHECK(w->sparse2dense(op.identity(), desc));
    if (A_mat_type == GrB_SPARSE) {
      CHECK(spmv(&w->dense_, mask, accum, op, &A->sparse_,
          &u->dense_, desc));
    } else {
      CHECK(gemv(&w->dense_, mask, accum, op, &A->dense_,
          &u->dense_, desc));
    }
    desc->lastmxv_ = GrB_PULLONLY;
  }

  if (desc->debug()) {
    std::cout << "===End mxv===\n";
    CHECK(w->print());
  }
  return GrB_SUCCESS;
}

template <typename W, typename U, typename V, typename M,
          typename BinaryOpT,     typename SemiringT>
Info eWiseMult(Vector<W>*       w,
               const Vector<M>* mask,
               BinaryOpT        accum,
               SemiringT        op,
               const Vector<U>* u,
               const Vector<V>* v,
               Descriptor*      desc) {
  Vector<U>* u_t = const_cast<Vector<U>*>(u);
  Vector<V>* v_t = const_cast<Vector<V>*>(v);

  if (desc->debug()) {
    std::cout << "===Begin eWiseMult===\n";
    CHECK(u_t->print());
    CHECK(v_t->print());
  }

  Storage u_vec_type;
  Storage v_vec_type;
  CHECK(u->getStorage(&u_vec_type));
  CHECK(v->getStorage(&v_vec_type));

  /* 
   * \brief 4 cases:
   * 1) sparse x sparse
   * 2) dense  x dense
   * 3) sparse x dense
   * 4) dense  x sparse
   */
  if (u_vec_type == GrB_SPARSE && v_vec_type == GrB_SPARSE) {
    CHECK(w->setStorage(GrB_SPARSE));
    CHECK(eWiseMultInner(&w->sparse_, mask, accum, op, &u->sparse_,
        &v->sparse_, desc));
  } else if (u_vec_type == GrB_DENSE && v_vec_type == GrB_DENSE) {
    // depending on whether sparse mask is present or not
    if (mask != NULL) {
      Storage mask_type;
      CHECK(mask->getStorage(&mask_type));
      if (mask_type == GrB_DENSE) {
        CHECK(w->setStorage(GrB_DENSE));
        CHECK(eWiseMultInner(&w->dense_, mask, accum, op, &u->dense_,
            &v->dense_, desc));
      } else if (mask_type == GrB_SPARSE) {
        CHECK(w->setStorage(GrB_SPARSE));
        CHECK(eWiseMultInner(&w->sparse_, &mask->sparse_, accum, op,
            &u->dense_, &v->dense_, desc));
      } else {
        return GrB_INVALID_OBJECT;
      }
    } else {
      CHECK(w->setStorage(GrB_DENSE));
      CHECK(eWiseMultInner(&w->dense_, mask, accum, op, &u->dense_,
          &v->dense_, desc));
    }
  } else if (u_vec_type == GrB_SPARSE && v_vec_type == GrB_DENSE) {
    // The boolean here keeps track of whether operators have been reversed.
    // This is important for non-commutative ops i.e. op(a,b) != op(b,a)
    CHECK(w->setStorage(GrB_SPARSE));
    CHECK(eWiseMultInner(&w->sparse_, mask, accum, op, &u->sparse_,
        &v->dense_, false, desc));
  } else if (u_vec_type == GrB_DENSE && v_vec_type == GrB_SPARSE) {
    CHECK(w->setStorage(GrB_SPARSE));
    CHECK(eWiseMultInner(&w->sparse_, mask, accum, op, &v->sparse_,
        &u->dense_, true, desc));
  } else {
    return GrB_INVALID_OBJECT;
  }

  if (desc->debug()) {
    std::cout << "===End eWiseMult===\n";
    CHECK(w->print());
  }
  return GrB_SUCCESS;
}

template <typename c, typename a, typename b, typename m,
          typename BinaryOpT,     typename SemiringT>
Info eWiseMult(Matrix<c>*       C,
               const Matrix<m>* mask,
               BinaryOpT        accum,
               SemiringT        op,
               const Matrix<a>* A,
               const Matrix<b>* B,
               Descriptor*      desc) {
  // Use either op->operator() or op->mul() as the case may be
  std::cout << "Error: eWiseMult matrix variant not implemented yet!\n";
}

/*!
 * Extension Method
 * Element-wise multiply of a matrix and scalar which gets broadcasted
 */
template <typename c, typename a, typename b, typename m,
          typename BinaryOpT,     typename SemiringT>
Info eWiseMult(Matrix<c>*       C,
               const Matrix<m>* mask,
               BinaryOpT        accum,
               SemiringT        op,
               const Matrix<a>* A,
               b                val,
               Descriptor*      desc) {
  if (desc->debug()) {
    std::cout << "===Begin eWiseMult===\n";
    std::cout << "val: " << val << std::endl;
  }

  Storage A_mat_type;
  CHECK(A->getStorage(&A_mat_type));

  if (A_mat_type == GrB_DENSE) {
    std::cout << "eWiseMult Dense Matrix Broadcast Scalar\n";
    std::cout << "Error: Feature not implemented yet!\n";
  } else if (A_mat_type == GrB_SPARSE) {
    // depending on whether mask is present or not
    if (mask != NULL) {
      std::cout << "eWiseMult Sparse Matrix Broadcast Scalar with Mask\n";
      std::cout << "Error: Feature not implemented yet!\n";
    } else {
      CHECK(C->setStorage(GrB_SPARSE));
      CHECK(eWiseMultInner(&C->sparse_, mask, accum, op, &A->sparse_,
          val, desc));
    }
  } else {
    return GrB_INVALID_OBJECT;
  }

  if (desc->debug()) {
    std::cout << "===End eWiseMult===\n";
  }
  return GrB_SUCCESS;
}

/*!
 * Extension Method
 * Element-wise multiply of a matrix and column vector which gets broadcasted.
 *  * If row vector broadcast is needed instead, set Input 1 to be transposed.
 */
template <typename c, typename a, typename b, typename m,
          typename BinaryOpT,     typename SemiringT>
Info eWiseMult(Matrix<c>*       C,
               const Matrix<m>* mask,
               BinaryOpT        accum,
               SemiringT        op,
               const Matrix<a>* A,
               const Vector<b>* B,
               Descriptor*      desc) {
  Vector<b>* B_t = const_cast<Vector<b>*>(B);
  if (desc->debug()) {
    std::cout << "===Begin eWiseMult===\n";
    CHECK(B_t->print());
  }

  // Transpose
  Desc_value inp0_mode, inp1_mode;
  CHECK(desc->get(GrB_INP0, &inp0_mode));
  CHECK(desc->get(GrB_INP1, &inp1_mode));
  if (inp0_mode != GrB_DEFAULT) return GrB_INVALID_VALUE;

  Storage A_mat_type;
  Storage B_vec_type;
  CHECK(A->getStorage(&A_mat_type));
  CHECK(B->getStorage(&B_vec_type));

  // u is column vector which gets broadcasted
  //   C = mask .* (A .* u)
  if (inp1_mode != GrB_TRAN) {
    if (A_mat_type == GrB_DENSE) {
      std::cout << "eWiseMult Dense Matrix Broadcast Vector\n";
      std::cout << "Error: Feature not implemented yet!\n";
    } else if (A_mat_type == GrB_SPARSE) {
      // depending on whether mask is present or not
      if (mask != NULL) {
        std::cout << "eWiseMult Sparse Matrix Broadcast Vector with Mask\n";
        std::cout << "Error: Feature not implemented yet!\n";
      } else {
        CHECK(C->setStorage(GrB_SPARSE));
        if (B_vec_type == GrB_SPARSE) {
          std::cout << "eWiseMult Sparse Matrix Broadcast Sparse Vector\n";
          std::cout << "Error: Feature not implemented yet!\n";
        } else {
          CHECK(eWiseMultColInner(&C->sparse_, mask, accum, op, &A->sparse_,
              &B->dense_, desc));
        }
      }
    } else {
      return GrB_INVALID_OBJECT;
    }
  // u^T is row vector which gets broadcasted
  //   C = mask .* (A .* u^T)
  } else {
    if (A_mat_type == GrB_DENSE) {
      std::cout << "eWiseMult Dense Matrix Broadcast Vector\n";
      std::cout << "Error: Feature not implemented yet!\n";
    } else if (A_mat_type == GrB_SPARSE) {
      // depending on whether mask is present or not
      if (mask != NULL) {
        std::cout << "eWiseMult Sparse Matrix Broadcast Vector with Mask\n";
        std::cout << "Error: Feature not implemented yet!\n";
      } else {
        CHECK(C->setStorage(GrB_SPARSE));
        if (B_vec_type == GrB_SPARSE) {
          std::cout << "eWiseMult Sparse Matrix Broadcast Sparse Vector\n";
          std::cout << "Error: Feature not implemented yet!\n";
        } else {
          CHECK(eWiseMultRowInner(&C->sparse_, mask, accum, op, &A->sparse_,
              &B->dense_, desc));
        }
      }
    } else {
      return GrB_INVALID_OBJECT;
    }
  }

  if (desc->debug()) {
    std::cout << "===End eWiseMult===\n";
  }
  return GrB_SUCCESS;
}

template <typename W, typename U, typename V, typename M,
          typename BinaryOpT,     typename SemiringT>
Info eWiseAdd(Vector<W>*       w,
              const Vector<M>* mask,
              BinaryOpT        accum,
              SemiringT        op,
              const Vector<U>* u,
              const Vector<V>* v,
              Descriptor*      desc) {
  Vector<U>* u_t = const_cast<Vector<U>*>(u);
  Vector<V>* v_t = const_cast<Vector<V>*>(v);

  if (desc->debug()) {
    std::cout << "===Begin eWiseAdd===\n";
    CHECK(u_t->print());
    CHECK(v_t->print());
  }

  Storage u_vec_type;
  Storage v_vec_type;
  CHECK(u->getStorage(&u_vec_type));
  CHECK(v->getStorage(&v_vec_type));

  /* 
   * \brief 4 cases:
   * 1) sparse x sparse
   * 2) dense  x dense
   * 3) sparse x dense
   * 4) dense  x sparse
   */
  if ((u == w && u_vec_type == GrB_SPARSE) ||
      (v == w && v_vec_type == GrB_SPARSE)) {
    if (u == w) {
      u_t->sparse2dense(op.identity(), desc);
      u_vec_type = GrB_DENSE;
    } else if (v == w) {
      v_t->sparse2dense(op.identity(), desc);
      v_vec_type = GrB_DENSE;
    }
  }

  CHECK(w->setStorage(GrB_DENSE));
  if (u_vec_type == GrB_SPARSE && v_vec_type == GrB_SPARSE) {
    CHECK(eWiseAddInner(&w->dense_, mask, accum, op, &u->sparse_,
        &v->sparse_, desc));
  } else if (u_vec_type == GrB_DENSE && v_vec_type == GrB_DENSE) {
    CHECK(eWiseAddInner(&w->dense_, mask, accum, op, &u->dense_,
        &v->dense_, desc));
  } else if (u_vec_type == GrB_SPARSE && v_vec_type == GrB_DENSE) {
    // The boolean here keeps track of whether operators have been reversed.
    // This is important for non-commutative ops i.e. op(a,b) != op(b,a)
    CHECK(eWiseAddInner(&w->dense_, mask, accum, op, &u->sparse_,
        &v->dense_, false, desc));
  } else if (u_vec_type == GrB_DENSE && v_vec_type == GrB_SPARSE) {
    CHECK(eWiseAddInner(&w->dense_, mask, accum, op, &v->sparse_,
        &u->dense_, true, desc));
  } else {
    std::cout << "Error: eWiseAdd backend invalid choice!\n";
    return GrB_INVALID_OBJECT;
  }

  if (desc->debug()) {
    std::cout << "===End eWiseAdd===\n";
    CHECK(w->print());
  }
  return GrB_SUCCESS;
}

template <typename c, typename a, typename b, typename m,
          typename BinaryOpT,     typename SemiringT>
Info eWiseAdd(Matrix<c>*       C,
              const Matrix<m>* mask,
              BinaryOpT        accum,
              SemiringT        op,
              const Matrix<a>* A,
              const Matrix<b>* B,
              Descriptor*      desc) {
  // Use either op->operator() or op->add() as the case may be
  std::cout << "Error: eWiseAdd matrix variant not implemented yet!\n";
}

/*!
 * Extension Method
 * Element-wise addition of a vector and scalar which gets broadcasted
 */
template <typename W, typename U, typename V, typename M,
          typename BinaryOpT,     typename SemiringT>
Info eWiseAdd(Vector<W>*       w,
              const Vector<M>* mask,
              BinaryOpT        accum,
              SemiringT        op,
              const Vector<U>* u,
              V                val,
              Descriptor*      desc) {
  Vector<U>* u_t = const_cast<Vector<U>*>(u);
  if (desc->debug()) {
    std::cout << "===Begin eWiseAdd===\n";
    CHECK(u_t->print());
    std::cout << "val: " << val << std::endl;
  }

  Storage u_vec_type;
  CHECK(u->getStorage(&u_vec_type));

  if (u_vec_type == GrB_DENSE) {
    if (mask != NULL) {
      std::cout << "eWiseAdd Dense Vector-Scalar with Mask\n";
      std::cout << "Error: Feature not implemented yet!\n";
    } else {
      CHECK(w->setStorage(GrB_DENSE));
      CHECK(eWiseAddInner(&w->dense_, mask, accum, op, &u->dense_,
          val, desc));
    }
  } else if (u_vec_type == GrB_SPARSE) {
    if (mask != NULL) {
      std::cout << "eWiseAdd Sparse Vector-Scalar Mask\n";
      std::cout << "Error: Feature not implemented yet!\n";
    } else {
      CHECK(w->setStorage(GrB_DENSE));
      CHECK(eWiseAddInner(&w->dense_, mask, accum, op, &u->sparse_,
          val, desc));
    }
  } else {
    return GrB_INVALID_OBJECT;
  }

  if (desc->debug()) {
    std::cout << "===End eWiseAdd===\n";
    CHECK(w->print());
  }
  return GrB_SUCCESS;
}

template <typename W, typename U, typename M,
          typename BinaryOpT>
Info extract(Vector<W>*                w,
             const Vector<M>*          mask,
             BinaryOpT                 accum,
             const Vector<U>*          u,
             const std::vector<Index>* indices,
             Index                     nindices,
             Descriptor*               desc) {
  std::cout << "Error: extract vector variant not implemented yet!\n";
}

template <typename c, typename a, typename m,
          typename BinaryOpT>
Info extract(Matrix<c>*                C,
             const Matrix<m>*          mask,
             BinaryOpT                 accum,
             const Matrix<a>*          A,
             const std::vector<Index>* row_indices,
             Index                     nrows,
             const std::vector<Index>* col_indices,
             Index                     ncols,
             Descriptor*               desc) {
  std::cout << "Error: extract matrix variant not implemented yet!\n";
}

template <typename W, typename a, typename M,
          typename BinaryOpT>
Info extract(Vector<W>*                w,
             const Vector<M>*          mask,
             BinaryOpT                 accum,
             const Matrix<a>*          A,
             const std::vector<Index>* row_indices,
             Index                     nrows,
             Index                     col_index,
             Descriptor*               desc) {
  std::cout << "Error: extract vector variant not implemented yet!\n";
}

template <typename W, typename U, typename M,
          typename BinaryOpT>
Info assign(Vector<W>*                w,
            const Vector<M>*          mask,
            BinaryOpT                 accum,
            const Vector<U>*          u,
            const std::vector<Index>* indices,
            Index                     nindices,
            Descriptor*               desc) {
  std::cout << "Error: assign vector variant not implemented yet!\n";
  return GrB_SUCCESS;
}

template <typename c, typename a, typename m,
          typename BinaryOpT>
Info assign(Matrix<c>*                C,
            const Matrix<m>*          mask,
            BinaryOpT                 accum,
            const Matrix<a>*          A,
            const std::vector<Index>* row_indices,
            Index                     nrows,
            const std::vector<Index>* col_indices,
            Index                     ncols,
            Descriptor*               desc) {
  std::cout << "Error: assign matrix variant not implemented yet!\n";
}

template <typename c, typename U, typename M,
          typename BinaryOpT>
Info assign(Matrix<c>*                C,
            const Vector<M>*          mask,
            BinaryOpT                 accum,
            const Vector<U>*          u,
            const std::vector<Index>* row_indices,
            Index                     nrows,
            Index                     col_index,
            Descriptor*               desc) {
  std::cout << "Error: assign matrix column variant not implemented yet!\n";
}

template <typename c, typename U, typename M,
          typename BinaryOpT>
Info assign(Matrix<c>*                C,
            const Vector<M>*          mask,
            BinaryOpT                 accum,
            const Vector<U>*          u,
            Index                     row_index,
            const std::vector<Index>* col_indices,
            Index                     ncols,
            Descriptor*               desc) {
  std::cout << "Error: assign matrix row variant not implemented yet!\n";
}

template <typename W, typename T, typename M,
          typename BinaryOpT>
Info assign(Vector<W>*                w,
            Vector<M>*                mask,
            BinaryOpT                 accum,
            T                         val,
            const std::vector<Index>* indices,
            Index                     nindices,
            Descriptor*               desc) {
  if (desc->debug()) {
    std::cout << "===Begin assign===\n";
    std::cout << "Input: " << val << std::endl;
  }

  // Get storage:
  Storage vec_type;
  CHECK(w->getStorage(&vec_type));

  // 2 cases:
  // 1) SpVec
  // 2) DeVec
  if (vec_type == GrB_SPARSE) {
    CHECK(w->setStorage(GrB_SPARSE));
    CHECK(assignSparse(&w->sparse_, mask, accum, val, indices, nindices,
        desc));
  } else if (vec_type == GrB_DENSE) {
    CHECK(w->setStorage(GrB_DENSE));
    CHECK(assignDense(&w->dense_, mask, accum, val, indices, nindices,
        desc));
  }

  if (desc->debug()) {
    std::cout << "===End assign===\n";
    CHECK(w->print());
  }
  return GrB_SUCCESS;
}

template <typename c, typename T, typename m,
          typename BinaryOpT>
Info assign(Matrix<c>*                C,
            const Matrix<m>*          mask,
            BinaryOpT                 accum,
            T                         val,
            const std::vector<Index>* row_indices,
            Index                     nrows,
            const std::vector<Index>* col_indices,
            Index                     ncols,
            Descriptor*               desc) {
  std::cout << "Error: assign matrix variant not implemented yet!\n";
}

template <typename W, typename U, typename M,
          typename BinaryOpT,     typename UnaryOpT>
Info apply(Vector<W>*       w,
           const Vector<M>* mask,
           BinaryOpT        accum,
           UnaryOpT         op,
           const Vector<U>* u,
           Descriptor*      desc) {
  Vector<U>* u_t = const_cast<Vector<U>*>(u);

  if (desc->debug()) {
    std::cout << "===Begin apply===\n";
    CHECK(u_t->print());
  }

  Storage u_vec_type;
  CHECK(u->getStorage(&u_vec_type));

  // sparse variant
  if (u_vec_type == GrB_SPARSE) {
    CHECK(w->setStorage(GrB_SPARSE));
    applySparse(&w->sparse_, mask, accum, op, &u_t->sparse_, desc);
  // dense variant
  } else if (u_vec_type == GrB_DENSE) {
    CHECK(w->setStorage(GrB_DENSE));
    applyDense(&w->dense_, mask, accum, op, &u_t->dense_, desc);
  } else {
    return GrB_UNINITIALIZED_OBJECT;
  }

  if (desc->debug()) {
    std::cout << "===End apply===\n";
    CHECK(w->print());
  }
  return GrB_SUCCESS;
}

template <typename c, typename a, typename m,
          typename BinaryOpT,     typename UnaryOpT>
Info apply(Matrix<c>*       C,
           const Matrix<m>* mask,
           BinaryOpT        accum,
           UnaryOpT         op,
           const Matrix<a>* A,
           Descriptor*      desc) {
  Matrix<a>* A_t = const_cast<Matrix<a>*>(A);

  if (desc->debug()) {
    std::cout << "===Begin apply===\n";
    CHECK(A_t->print());
  }

  Storage A_mat_type;
  CHECK(A->getStorage(&A_mat_type));

  // sparse variant
  if (A_mat_type == GrB_SPARSE) {
    CHECK(C->setStorage(GrB_SPARSE));
    applySparse(&C->sparse_, mask, accum, op, &A_t->sparse_, desc);
  // dense variant
  } else if (A_mat_type == GrB_DENSE) {
    CHECK(C->setStorage(GrB_DENSE));
    applyDense(&C->dense_, mask, accum, op, &A_t->dense_, desc);
  } else {
    return GrB_UNINITIALIZED_OBJECT;
  }

  if (desc->debug()) {
    std::cout << "===End apply===\n";
    CHECK(C->print());
  }
  return GrB_SUCCESS;
}

template <typename W, typename a, typename M,
          typename BinaryOpT,     typename MonoidT>
Info reduce(Vector<W>*       w,
            const Vector<M>* mask,
            BinaryOpT        accum,
            MonoidT          op,
            const Matrix<a>* A,
            Descriptor*      desc) {
  if (desc->debug()) {
    std::cout << "===Begin reduce===\n";
  }

  // Get storage:
  Storage mat_type;
  CHECK(A->getStorage(&mat_type));
  CHECK(w->setStorage(GrB_DENSE));

  if (mask == NULL) {
  // 2 cases:
  // 1) SpMat
  // 2) DeMat
    if (mat_type == GrB_SPARSE)
      CHECK(reduceInner(&w->dense_, mask, accum, op, &A->sparse_, desc));
    else if (mat_type == GrB_DENSE)
      CHECK(reduceInner(&w->dense_, mask, accum, op, &A->dense_, desc));
    else
      return GrB_UNINITIALIZED_OBJECT;
  } else {
    std::cout << "Error: Masked reduce not implemented yet!\n";
  }

  if (desc->debug()) {
    std::cout << "===End reduce===\n";
    CHECK(w->print());
  }
  return GrB_SUCCESS;
}

template <typename T, typename U,
          typename BinaryOpT, typename MonoidT>
Info reduce(T*               val,
            BinaryOpT        accum,
            MonoidT          op,
            const Vector<U>* u,
            Descriptor*      desc) {
  Vector<U>* u_t = const_cast<Vector<U>*>(u);

  if (desc->debug()) {
    std::cout << "===Begin reduce===\n";
    CHECK(u_t->print());
  }

  // Get storage:
  Storage vec_type;
  CHECK(u->getStorage(&vec_type));

  // 2 cases:
  // 1) SpVec
  // 2) DeVec
  if (vec_type == GrB_SPARSE)
    CHECK(reduceInner(val, accum, op, &u->sparse_, desc));
  else if (vec_type == GrB_DENSE)
    CHECK(reduceInner(val, accum, op, &u->dense_, desc));
  else
    return GrB_UNINITIALIZED_OBJECT;

  if (desc->debug()) {
    std::cout << "===End reduce===\n";
    std::cout << "Output: " << *val << std::endl;
  }
  return GrB_SUCCESS;
}

template <typename T, typename a,
          typename BinaryOpT,     typename MonoidT>
Info reduce(T*               val,
            BinaryOpT        accum,
            MonoidT          op,
            const Matrix<a>* A,
            Descriptor*      desc) {
  if (desc->debug()) {
    std::cout << "===Begin reduce===\n";
  }

  // Get storage:
  Storage mat_type;
  CHECK(A->getStorage(&mat_type));

  // 2 cases:
  // 1) SpMat
  // 2) DeMat
  if (mat_type == GrB_SPARSE) {
    CHECK(reduceInner(val, accum, op, &A->sparse_, desc));
  } else if (mat_type == GrB_DENSE) {
    std::cout << "Error: reduce matrix-scalar for dense matrix\n";
    std::cout << "not implemented yet!\n";
    // CHECK(reduceInner(val, accum, op, &A->dense_, desc));
  } else {
    return GrB_UNINITIALIZED_OBJECT;
  }

  if (desc->debug()) {
    std::cout << "===End reduce===\n";
    std::cout << "Output: " << *val << std::endl;
  }
}

template <typename c, typename a, typename m,
          typename BinaryOpT>
Info transpose(Matrix<c>*       C,
               const Matrix<m>* mask,
               BinaryOpT        accum,
               const Matrix<a>* A,
               Descriptor*      desc) {
  if (desc->debug()) {
    std::cout << "===Begin transpose===\n";
  }
  std::cout << "Error: transpose not implemented yet!\n";
}

template <typename T, typename a, typename b,
          typename SemiringT>
Info traceMxmTranspose(T*               val,
                       SemiringT        op,
                       const Matrix<a>* A,
                       const Matrix<b>* B,
                       Descriptor*      desc) {
  if (desc->debug()) {
    std::cout << "===Begin traceMxmTranspose===\n";
  }

  // Get storage:
  Storage A_mat_type;
  Storage B_mat_type;
  CHECK(A->getStorage(&A_mat_type));
  CHECK(B->getStorage(&B_mat_type));

  // 4 cases:
  // 1) SpMat x SpMat
  // 2) DeMat x DeMat
  // 3) SpMat x DeMat
  // 4) DeMat x SpMat
  if (A_mat_type == GrB_SPARSE && B_mat_type == GrB_SPARSE)
    CHECK(traceMxmTransposeInner(val, op, &A->sparse_, &B->sparse_, desc));
  else
    std::cout << "Error: Trace operator not implemented!\n";

  if (desc->debug()) {
    std::cout << "===End traceMxmTranspose===\n";
  }
  return GrB_SUCCESS;
}

template <typename W, typename M, typename U, typename T>
Info scatter(Vector<W>*       w,
             const Vector<M>* mask,
             const Vector<U>* u,
             T                val,
             Descriptor*      desc) {
  Vector<U>* u_t = const_cast<Vector<U>*>(u);

  if (desc->debug()) {
    std::cout << "===Begin scatter===\n";
    CHECK(u_t->print());
  }

  Storage u_vec_type;
  CHECK(u->getStorage(&u_vec_type));
  /*!
   * Cases:
   * 1) sparse -> dense
   * 2) dense  -> dense
   */
  CHECK(w->setStorage(GrB_DENSE));
  if (u_vec_type == GrB_SPARSE)
    scatterSparse(&w->dense_, mask, &u->sparse_, val, desc);
  else if (u_vec_type == GrB_DENSE)
    scatterDense(&w->dense_, mask, &u->dense_, val, desc);
  else
    return GrB_UNINITIALIZED_OBJECT;

  if (desc->debug()) {
    std::cout << "===End scatter===\n";
    CHECK(w->print());
  }
  return GrB_SUCCESS;
}

template <typename W, typename a>
Info graphColor(Vector<W>*       w,
                const Matrix<a>* A,
                Descriptor*      desc) {
  if (desc->debug())
    std::cout << "===Begin cuSPARSE graph color===\n";

  CHECK(w->setStorage(GrB_DENSE));
  
  cusparse_color(&w->dense_, &A->sparse_, desc);

  if (desc->debug()) {
    std::cout << "===End cuSPARSE graph color===\n";
    CHECK(w->print());
  }
  return GrB_SUCCESS;
}

template <typename W, typename U, typename a, typename M,
          typename BinaryOpT, typename SemiringT>
Info applyVxm(Vector<W>*       w,
              const Vector<M>* mask,
              BinaryOpT        accum,
              SemiringT        op,
              const Vector<U>* u,
              const Matrix<a>* A,
              Descriptor*      desc) {
  Vector<U>* u_t = const_cast<Vector<U>*>(u);

  if (desc->debug()) {
    std::cout << "===Begin vxm===\n";
    CHECK(u_t->print());
  }

  // Get storage
  Storage u_vec_type;
  Storage A_mat_type;
  CHECK(u->getStorage(&u_vec_type));
  CHECK(A->getStorage(&A_mat_type));

  // Transpose
  Desc_value inp0_mode;
  CHECK(desc->get(GrB_INP0, &inp0_mode));
  if (inp0_mode != GrB_DEFAULT) return GrB_INVALID_VALUE;

  // Treat vxm as an mxv with transposed matrix
  CHECK(desc->toggle(GrB_INP1));

  LoadBalanceMode lb_mode = getEnv("GRB_LOAD_BALANCE_MODE",
      GrB_LOAD_BALANCE_MERGE);

  // Conversions
  // TODO(@ctcyang): add tol
  SparseMatrixFormat A_format;
  bool A_symmetric;
  CHECK(A->getFormat(&A_format));
  CHECK(A->getSymmetry(&A_symmetric));

  Desc_value vxm_mode, tol;
  CHECK(desc->get(GrB_MXVMODE, &vxm_mode));
  CHECK(desc->get(GrB_TOL,     &tol));
  if (desc->debug()) {
    std::cout << "Load balance mode: " << lb_mode << std::endl;
    std::cout << "Identity: " << op.identity() << std::endl;
    std::cout << "Sparse format: " << A_format << std::endl;
    std::cout << "Symmetric: " << A_symmetric << std::endl;
  }

  // Fallback for lacking CSC storage overrides any mxvmode selections
  if (!A_symmetric && A_format == GrB_SPARSE_MATRIX_CSRONLY) {
    if (u_vec_type == GrB_DENSE)
      CHECK(u_t->dense2sparse(op.identity(), desc));
  } else if (vxm_mode == GrB_PUSHPULL) {
    CHECK(u_t->convert(op.identity(), desc));
  } else if (vxm_mode == GrB_PUSHONLY && u_vec_type == GrB_DENSE) {
    CHECK(u_t->dense2sparse(op.identity(), desc));
  } else if (vxm_mode == GrB_PULLONLY && u_vec_type == GrB_SPARSE) {
    CHECK(u_t->sparse2dense(op.identity(), desc));
  }

  // Check if vector type was changed due to conversion!
  CHECK(u->getStorage(&u_vec_type));

  if (desc->debug())
    std::cout << "u_vec_type: " << u_vec_type << std::endl;

  // Breakdown into 4 cases:
  // 1) SpMSpV: SpMat x SpVec
  // 2) SpMV:   SpMat x DeVec (preferred to 3)
  // 3) SpMSpV: SpMat x SpVec (fallback if CSC representation not available)
  // 4) GeMV:   DeMat x DeVec
  //
  // Note: differs from mxv, because mxv would say instead:
  // 3) "... if CSC representation not available ..."
  if (A_mat_type == GrB_SPARSE && u_vec_type == GrB_SPARSE) {
    if (lb_mode == GrB_LOAD_BALANCE_SIMPLE ||
        lb_mode == GrB_LOAD_BALANCE_TWC) {
      CHECK(w->setStorage(GrB_DENSE));
      // 1a) Simple SpMSpV no load-balancing codepath
      if (lb_mode == GrB_LOAD_BALANCE_SIMPLE)
        std::cout << "Simple SPMSPV not implemented yet!\n";
        // CHECK( spmspvSimple(&w->dense_, mask, accum, op, &A->sparse_,
        //     &u->sparse_, desc) );
      // 1b) Thread-warp-block (single kernel) codepath
      else if (lb_mode == GrB_LOAD_BALANCE_TWC)
        std::cout << "Error: B40C load-balance algorithm not implemented yet!\n";
      if (desc->debug()) {
        CHECK(w->getStorage(&u_vec_type));
        std::cout << "w_vec_type: " << u_vec_type << std::endl;
      }
      CHECK(w->dense2sparse(op.identity(), desc));
      if (desc->debug()) {
        CHECK(w->getStorage(&u_vec_type));
        std::cout << "w_vec_type: " << u_vec_type << std::endl;
      }
    // 1c) Merge-path (two-phase decomposition) codepath
    } else if (lb_mode == GrB_LOAD_BALANCE_MERGE) {
      CHECK(w->setStorage(GrB_SPARSE));
      CHECK(spmspvMerge(&w->sparse_, mask, accum, op, &A->sparse_,
          &u->sparse_, desc));
    } else {
      std::cout << "Error: Invalid load-balance algorithm!\n";
    }
    desc->lastmxv_ = GrB_PUSHONLY;
  } else {
    // TODO(@ctcyang): Some performance left on table, sparse2dense should
    // only convert rather than setStorage if accum is being used
    CHECK(w->setStorage(GrB_DENSE));
    // CHECK(w->sparse2dense(op.identity(), desc));
    if (A_mat_type == GrB_SPARSE)
      CHECK(applySpmv(&w->dense_, mask, accum, op, &A->sparse_, &u->dense_,
          desc));
    else
      CHECK(gemv(&w->dense_, mask, accum, op, &A->dense_, &u->dense_,
          desc));
    desc->lastmxv_ = GrB_PULLONLY;
  }

  // Undo change to desc by toggling again
  CHECK(desc->toggle(GrB_INP1));

  if (desc->debug()) {
    std::cout << "===End vxm===\n";
    CHECK(w->print());
  }
  return GrB_SUCCESS;
}
}  // namespace backend
}  // namespace graphblas

#endif  // GRAPHBLAS_BACKEND_CUDA_OPERATIONS_HPP_
