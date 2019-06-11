#ifndef GRAPHBLAS_OPERATIONS_HPP_
#define GRAPHBLAS_OPERATIONS_HPP_

#include <vector>

#define __GRB_BACKEND_OPERATIONS_HEADER <graphblas/backend/__GRB_BACKEND_ROOT/operations.hpp>
#include __GRB_BACKEND_OPERATIONS_HEADER
#undef __GRB_BACKEND_OPERATIONS_HEADER

namespace graphblas {
// TODO(@ctcyang): make all operations mxm, mxv, etc. follow vxm() and assign()
// constant variant

/*!
 * Matrix-matrix product
 *   C = C + mask .* (A * B)    +: accum
 *                              *: op
 *                             .*: Boolean and
 */
template <typename c, typename m, typename a, typename b,
          typename BinaryOpT,     typename SemiringT>
Info mxm(Matrix<c>*       C,
         const Matrix<m>* mask,
         BinaryOpT        accum,
         SemiringT        op,
         const Matrix<a>* A,
         const Matrix<b>* B,
         Descriptor*      desc) {
  // Null pointer check
  if (C == NULL || A == NULL || B == NULL || desc == NULL)
    return GrB_UNINITIALIZED_OBJECT;

  // Dimension check
  // Case 1: A *B
  CHECK(checkDimRowCol(B, A,    "B.nrows != A.ncols"));
  CHECK(checkDimRowRow(A, C,    "A.nrows != C.nrows"));
  CHECK(checkDimColCol(B, C,    "B.ncols != C.ncols"));
  CHECK(checkDimRowRow(C, mask, "C.nrows != mask.nrows"));
  CHECK(checkDimColCol(C, mask, "C.ncols != mask.ncols"));
  // Case 2: AT*B
  // Case 3: A *BT
  // Case 4: AT*BT

  const backend::Matrix<m>* mask_t = (mask == NULL) ? NULL : &mask->matrix_;
  backend::Descriptor* desc_t = (desc == NULL) ? NULL : &desc->descriptor_;

  return backend::mxm<c, a, b, m>(&C->matrix_, mask_t, accum, op, &A->matrix_,
      &B->matrix_, desc_t);
}

/*!
 * Vector-matrix product
 *   w^T = w^T + mask^T .* (u^T * A)    +: accum
 *                                      *: op
 *                                     .*: Boolean and
 */
template <typename W, typename M, typename U, typename a,
          typename BinaryOpT, typename SemiringT>
Info vxm(Vector<W>*       w,
         const Vector<M>* mask,
         BinaryOpT        accum,
         SemiringT        op,
         const Vector<U>* u,
         const Matrix<a>* A,
         Descriptor*      desc) {
  // Null pointer check
  if (w == NULL || u == NULL || A == NULL || desc == NULL)
    return GrB_UNINITIALIZED_OBJECT;

  // Dimension check
  Index u_nvals = 0;
  CHECK(u->nvals(&u_nvals));
  if (u_nvals == 0)
    return GrB_UNINITIALIZED_OBJECT;

  // Case 1: u*A
  CHECK(checkDimRowSize(A,  u,    "A.nrows != u.size"));
  CHECK(checkDimColSize(A,  w,    "A.ncols != w.size"));
  CHECK(checkDimSizeSize(w, mask, "w.size  != mask.size"));

  // Case 2: u*AT
  const backend::Vector<M>* mask_t = (mask == NULL) ? NULL : &mask->vector_;
  backend::Descriptor*      desc_t = (desc == NULL) ? NULL : &desc->descriptor_;

  return backend::vxm<W, U, a, M>(&w->vector_, mask_t, accum, op, &u->vector_,
      &A->matrix_, desc_t);
}

/*!
 * Matrix-vector product
 *   w = w + mask .* (A * u)    +: accum
 *                              *: op
 *                             .*: Boolean and
 */
template <typename W, typename M, typename a, typename U,
          typename BinaryOpT, typename SemiringT>
Info mxv(Vector<W>*       w,
         const Vector<M>* mask,
         BinaryOpT        accum,
         SemiringT        op,
         const Matrix<a>* A,
         const Vector<U>* u,
         Descriptor*      desc) {
  // Null pointer check
  // TODO(@ctcyang): need way to check op is not empty now that op is no longer
  // pointer
  if (w == NULL || u == NULL || A == NULL || desc == NULL)
    return GrB_UNINITIALIZED_OBJECT;

  // Dimension check
  Index u_nvals = 0;
  CHECK(u->nvals(&u_nvals));
  if (u_nvals == 0)
    return GrB_UNINITIALIZED_OBJECT;

  // Case 1: A *u
  CHECK(checkDimColSize(A,  u,    "A.ncols != u.size"));
  CHECK(checkDimRowSize(A,  w,    "A.nrows != w.size"));
  CHECK(checkDimSizeSize(w, mask, "w.size  != mask.size"));

  // Case 2: AT*u
  const backend::Vector<M>* mask_t = (mask == NULL) ? NULL : &mask->vector_;
  backend::Descriptor*      desc_t = (desc == NULL) ? NULL : &desc->descriptor_;

  return backend::mxv<W, U, a, M>(&w->vector_, mask_t, accum, op, &A->matrix_,
      &u->vector_, desc_t);
}

/*!
 * Element-wise multiply of two vectors
 *   w = w + mask .* (u .* v)    +: accum
 *                 2     1    1).*: op (semiring multiply)
 *                            2).*: Boolean and
 */
template <typename W, typename M, typename U, typename V,
          typename BinaryOpT,     typename SemiringT>
Info eWiseMult(Vector<W>*       w,
               const Vector<M>* mask,
               BinaryOpT        accum,
               SemiringT        op,
               const Vector<U>* u,
               const Vector<V>* v,
               Descriptor*      desc) {
  // Null pointer check
  if (w == NULL || u == NULL || v == NULL || desc == NULL)
    return GrB_UNINITIALIZED_OBJECT;

  // Dimension check
  CHECK(checkDimSizeSize(u, v,    "u.size != v.size"));
  CHECK(checkDimSizeSize(u, w,    "u.size != mask.size"));
  CHECK(checkDimSizeSize(u, mask, "v.size != mask.size"));

  const backend::Vector<M>* mask_t = (mask == NULL) ? NULL : &mask->vector_;
  backend::Descriptor*      desc_t = (desc == NULL) ? NULL : &desc->descriptor_;

  return backend::eWiseMult(&w->vector_, mask_t, accum, op, &u->vector_,
      &v->vector_, desc_t);
}

/*!
 * Element-wise multiply of two matrices
 *   C = C + mask .* (A .* B)    +: accum
 *                 2     1    1).*: op (semiring multiply)
 *                            2).*: Boolean and
 */
template <typename c, typename m, typename a, typename b,
          typename BinaryOpT,     typename SemiringT>
Info eWiseMult(Matrix<c>*       C,
               const Matrix<m>* mask,
               BinaryOpT        accum,
               SemiringT        op,
               const Matrix<a>* A,
               const Matrix<b>* B,
               Descriptor*      desc) {
  // Null pointer check
  if (C == NULL || A == NULL || B == NULL || desc == NULL)
    return GrB_UNINITIALIZED_OBJECT;

  // Dimension check
  CHECK(checkDimRowRow(B, A,    "B.nrows != A.nrows"));
  CHECK(checkDimColCol(B, A,    "B.ncols != A.ncols"));
  CHECK(checkDimRowRow(A, C,    "A.nrows != C.nrows"));
  CHECK(checkDimColCol(A, C,    "A.ncols != C.ncols"));
  CHECK(checkDimRowRow(C, mask, "C.nrows != mask.nrows"));
  CHECK(checkDimColCol(C, mask, "C.ncols != mask.ncols"));

  const backend::Matrix<m>* mask_t = (mask == NULL) ? NULL : &mask->matrix_;
  backend::Descriptor*      desc_t = (desc == NULL) ? NULL : &desc->descriptor_;

  return backend::eWiseMult(&C->matrix_, mask_t, accum, op, &A->matrix_,
      &B->matrix_, desc_t);
}

/*!
 * Extension Method
 * Element-wise multiply of a matrix and scalar which gets broadcasted
 *   C = C + mask .* (A .* val)  +: accum
 *                 2     1    1).*: op (semiring multiply)
 *                            2).*: Boolean and
 *
 * TODO(@ctcyang): add scalar .* matrix variant for non-commutative semiring
 * multiply
 */
template <typename c, typename m, typename a, typename b,
          typename BinaryOpT,     typename SemiringT>
Info eWiseMult(Matrix<c>*       C,
               const Matrix<m>* mask,
               BinaryOpT        accum,
               SemiringT        op,
               const Matrix<a>* A,
               b                val,
               Descriptor*      desc) {
  // Null pointer check
  if (C == NULL || A == NULL || desc == NULL)
    return GrB_UNINITIALIZED_OBJECT;

  // Dimension check
  CHECK(checkDimRowRow(A, C,    "A.nrows != C.nrows"));
  CHECK(checkDimColCol(A, C,    "A.ncols != C.ncols"));
  CHECK(checkDimRowRow(A, mask, "A.nrows != mask.nrows"));
  CHECK(checkDimColCol(A, mask, "A.ncols != mask.ncols"));

  const backend::Matrix<m>* mask_t = (mask == NULL) ? NULL : &mask->matrix_;
  backend::Descriptor*      desc_t = (desc == NULL) ? NULL : &desc->descriptor_;

  return backend::eWiseMult(&C->matrix_, mask_t, accum, op, &A->matrix_,
      val, desc_t);
}

/*!
 * Extension Method
 * Element-wise multiply of a matrix and column vector which gets broadcasted.
 * If row vector broadcast is needed instead, set Input 1 to be transposed.
 *   C = C + mask .* (A .* u)    +: accum
 *                 2     1    1).*: op (semiring multiply)
 *                            2).*: Boolean and
 */
template <typename c, typename m, typename a, typename b,
          typename BinaryOpT,     typename SemiringT>
Info eWiseMult(Matrix<c>*       C,
               const Matrix<m>* mask,
               BinaryOpT        accum,
               SemiringT        op,
               const Matrix<a>* A,
               const Vector<b>* B,
               Descriptor*      desc) {
  // Null pointer check
  if (C == NULL || A == NULL || B == NULL || desc == NULL)
    return GrB_UNINITIALIZED_OBJECT;

  // Dimension check
  //
  // This check is dependent on whether GrB_INP1 is set to transpose or not, so
  // we should delegate this check to backend
  // CHECK(checkDimRowSize(A, B, "A.nrows != B.size"));  // if B transposed
  // CHECK(checkDimRowSize(A, B, "A.ncols != B.size"));  // if B not transposed
  CHECK(checkDimRowRow( A, C,    "A.nrows != C.nrows"));
  CHECK(checkDimColCol( A, C,    "A.ncols != C.ncols"));
  CHECK(checkDimRowRow( A, mask, "A.nrows != mask.nrows"));
  CHECK(checkDimColCol( A, mask, "A.ncols != mask.ncols"));

  const backend::Matrix<m>* mask_t = (mask == NULL) ? NULL : &mask->matrix_;
  backend::Descriptor*      desc_t = (desc == NULL) ? NULL : &desc->descriptor_;

  return backend::eWiseMult(&C->matrix_, mask_t, accum, op, &A->matrix_,
      &B->vector_, desc_t);
}

/*!
 * Element-wise addition of two vectors
 *   w = w + mask .* (u + v)    +: accum
 *                             .+: op (semiring add)
 *                             .*: Boolean and
 */
template <typename W, typename M, typename U, typename V,
          typename BinaryOpT,     typename SemiringT>
Info eWiseAdd(Vector<W>*       w,
              const Vector<M>* mask,
              BinaryOpT        accum,
              SemiringT        op,
              const Vector<U>* u,
              const Vector<V>* v,
              Descriptor*      desc) {
  // Null pointer check
  if (w == NULL || u == NULL || v == NULL || desc == NULL)
    return GrB_UNINITIALIZED_OBJECT;

  // Dimension check
  CHECK(checkDimSizeSize(u, v,    "u.size != v.size"));
  CHECK(checkDimSizeSize(u, mask, "u.size != mask.size"));
  CHECK(checkDimSizeSize(v, mask, "v.size != mask.size"));
  CHECK(checkDimSizeSize(w, mask, "w.size != mask.size"));

  const backend::Vector<M>* mask_t = (mask == NULL) ? NULL : &mask->vector_;
  backend::Descriptor*      desc_t = (desc == NULL) ? NULL : &desc->descriptor_;

  return backend::eWiseAdd(&w->vector_, mask_t, accum, op, &u->vector_,
      &v->vector_, desc_t);
}

/*!
 * Element-wise addition of two matrices
 *   C = C + mask .* (A .+ B)    +: accum
 *                              .+: op (semiring add)
 *                              .*: Boolean and
 */
template <typename c, typename m, typename a, typename b,
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
 *   w = w + mask .* (u + val)  +: accum
 *                             .+: op (semiring add)
 *                             .*: Boolean and
 *
 * TODO(@ctcyang): add scalar .+ vector variant for non-commutative semiring
 * multiply
 */
template <typename W, typename M, typename U, typename V,
          typename BinaryOpT,     typename SemiringT>
Info eWiseAdd(Vector<W>*       w,
              const Vector<M>* mask,
              BinaryOpT        accum,
              SemiringT        op,
              const Vector<U>* u,
              V                val,
              Descriptor*      desc) {
  // Null pointer check
  if (w == NULL || u == NULL || desc == NULL)
    return GrB_UNINITIALIZED_OBJECT;

  // Dimension check
  CHECK(checkDimSizeSize(u, w,    "u.size != mask.size"));
  CHECK(checkDimSizeSize(u, mask, "v.size != mask.size"));

  const backend::Vector<M>* mask_t = (mask == NULL) ? NULL : &mask->vector_;
  backend::Descriptor*      desc_t = (desc == NULL) ? NULL : &desc->descriptor_;

  return backend::eWiseAdd(&w->vector_, mask_t, accum, op, &u->vector_,
      val, desc_t);
}

/*!
 * Subvector extraction from vector
 *   w = w + mask .* (u[indices])   +: accum
 *                                 .*: Boolean and
 */
template <typename W, typename M, typename U,
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

/*!
 * Submatrix extraction from matrix
 *   C = C + mask .* (A[row_indices, col_indices])   +: accum
 *                                                  .*: Boolean and
 */
template <typename c, typename m, typename a,
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

/*!
 * Column vector extraction from matrix
 *   w = w + mask .* (A[row_indices, col_index])   +: accum
 *                                                .*: Boolean and
 */
template <typename W, typename M, typename a,
          typename BinaryOpT>
Info extract(Vector<W>*                w,
             const Vector<M>*          mask,
             BinaryOpT                 accum,
             const Matrix<a>*          A,
             const std::vector<Index>* row_indices,
             Index                     nrows,
             Index                     col_index,
             Descriptor*               desc) {
  std::cout << "Error: extract matrix variant not implemented yet!\n";
}

/*!
 * Assign vector to vector subset
 *   w[indices] = w[indices] + mask .* u   +: accum
 *                                        .*: Boolean and
 */
template <typename W, typename M, typename U,
          typename BinaryOpT>
Info assign(Vector<W>*                w,
            const Vector<U>*          mask,
            BinaryOpT                 accum,
            const Vector<U>*          u,
            const std::vector<Index>* indices,
            Index                     nindices,
            Descriptor*               desc) {
  std::cout << "Error: assign vector variant not implemented yet!\n";
}

/*!
 * Assign matrix to matrix subset
 *   C[row_indices, col_indices] = 
 *       C[row_indices, col_indices] + mask .* A   +: accum
 *                                                .*: Boolean and
 */
template <typename c, typename m, typename a,
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

/*!
 * Assign vector to matrix column subset
 *   C[row_indices, col_index] = 
 *       C[row_indices, col_index] + mask .* u   +: accum
 *                                              .*: Boolean and
 */
template <typename c, typename M, typename U,
          typename BinaryOpT>
Info assign(Matrix<c>*                C,
            const Vector<M>*          mask,
            BinaryOpT                 accum,
            const Vector<U>*          u,
            const std::vector<Index>* row_indices,
            Index                     nrows,
            Index                     col_index,
            Descriptor*               desc) {
  std::cout << "Error: assign matrix variant not implemented yet!\n";
}

/*!
 * Assign vector to matrix row subset
 *   C[row_index, col_indices] = 
 *       C[row_index, col_indices] + mask .* u   +: accum
 *                                              .*: Boolean and
 */
template <typename c, typename M, typename U,
          typename BinaryOpT>
Info assign(Matrix<c>*                C,
            const Vector<M>*          mask,
            BinaryOpT                 accum,
            const Vector<U>*          u,
            Index                     row_index,
            const std::vector<Index>* col_indices,
            Index                     ncols,
            Descriptor*               desc) {
  std::cout << "Error: assign matrix variant not implemented yet!\n";
}

/*!
 * Assign constant to vector subset
 *   w[indices] = w[indices] + val   +: accum
 *                                  .*: Boolean and
 */
template <typename W, typename M, typename T,
          typename BinaryOpT>
Info assign(Vector<W>*                w,
            Vector<M>*                mask,
            BinaryOpT                 accum,
            T                         val,
            const std::vector<Index>* indices,
            Index                     nindices,
            Descriptor*               desc) {
  // Null pointer check
  if (w == NULL || desc == NULL)
    return GrB_UNINITIALIZED_OBJECT;

  // Dimension check
  // -only have one case (no transpose option)
  CHECK(checkDimSizeSize(w, mask, "w.size  != mask.size"));

  auto                 mask_t = (mask == NULL) ? NULL : &mask->vector_;
  backend::Descriptor* desc_t = (desc == NULL) ? NULL : &desc->descriptor_;

  return backend::assign(&w->vector_, mask_t, accum, val, indices, nindices,
      desc_t);
}

/*!
 * Assign constant to matrix subset
 *   C[row_indices, col_indices] = 
 *     C[row_indices, col_indices] + val   +: accum
 *                                        .*: Boolean and
 */
template <typename c, typename m, typename T,
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

/*!
 * Apply unary operation to vector
 *   w = w + mask .* op(u)    +: accum
 *                           .*: Boolean and
 */
template <typename W, typename M, typename U,
          typename BinaryOpT,     typename UnaryOpT>
Info apply(Vector<W>*       w,
           const Vector<M>* mask,
           BinaryOpT        accum,
           UnaryOpT         op,
           const Vector<U>* u,
           Descriptor*      desc) {
  // Null pointer check
  if (w == NULL || u == NULL)
    return GrB_UNINITIALIZED_OBJECT;

  // Dimension check
  CHECK(checkDimSizeSize(u, w,    "u.size != w.size"));
  CHECK(checkDimSizeSize(u, mask, "u.size != mask.size"));
  CHECK(checkDimSizeSize(w, mask, "w.size != mask.size"));

  const backend::Vector<M>* mask_t = (mask == NULL) ? NULL : &mask->vector_;
  backend::Descriptor*      desc_t = (desc == NULL) ? NULL : &desc->descriptor_;

  return backend::apply(&w->vector_, mask_t, accum, op, &u->vector_, desc_t);
}

/*!
 * Apply unary operation to matrix
 *   w = w + mask .* op(u)    +: accum
 *                           .*: Boolean and
 */
template <typename c, typename m, typename a,
          typename BinaryOpT,     typename UnaryOpT>
Info apply(Matrix<c>*       C,
           const Matrix<m>* mask,
           BinaryOpT        accum,
           UnaryOpT         op,
           const Matrix<a>* A,
           Descriptor*      desc) {
  // Null pointer check
  if (A == NULL || C == NULL)
    return GrB_UNINITIALIZED_OBJECT;

  // Dimension check
  CHECK(checkDimRowRow(A, C,    "A.nrows != C.nrows"));
  CHECK(checkDimColCol(A, C,    "A.ncols != C.ncols"));
  CHECK(checkDimRowRow(A, mask, "A.nrows != mask.nrows"));
  CHECK(checkDimColCol(A, mask, "A.ncols != mask.ncols"));
  CHECK(checkDimRowRow(C, mask, "C.nrows != mask.nrows"));
  CHECK(checkDimColCol(C, mask, "C.ncols != mask.ncols"));

  const backend::Matrix<m>* mask_t = (mask == NULL) ? NULL : &mask->matrix_;
  backend::Descriptor*      desc_t = (desc == NULL) ? NULL : &desc->descriptor_;

  return backend::apply(&C->matrix_, mask_t, accum, op, &A->matrix_, desc_t);
}

/*!
 * Reduction along matrix rows to form vector
 *   w(i) = w(i) + mask(i) .* \sum_j A(i,j) for all j    +: accum
 *                                                     sum: op
 *                                                      .*: Boolean and
 */
template <typename W, typename M, typename a,
          typename BinaryOpT,     typename MonoidT>
Info reduce(Vector<W>*       w,
            const Vector<M>* mask,
            BinaryOpT        accum,
            MonoidT          op,
            const Matrix<a>* A,
            Descriptor*      desc) {
  if (w == NULL || A == NULL)
    return GrB_UNINITIALIZED_OBJECT;

  const backend::Vector<M>* mask_t = (mask == NULL) ? NULL : &mask->vector_;
  backend::Descriptor*      desc_t = (desc == NULL) ? NULL : &desc->descriptor_;

  return backend::reduce(&w->vector_, mask_t, accum, op, &A->matrix_, desc_t);
}

/*!
 * Reduction of vector to form scalar
 *   val = val + \sum_i u(i) for all i    +: accum
 *                                      sum: op
 */
template <typename T, typename U,
          typename BinaryOpT, typename MonoidT>
Info reduce(T*               val,
            BinaryOpT        accum,
            MonoidT          op,
            const Vector<U>* u,
            Descriptor*      desc) {
  if (val == NULL || u == NULL)
    return GrB_UNINITIALIZED_OBJECT;

  backend::Descriptor* desc_t = (desc == NULL) ? NULL : &desc->descriptor_;

  return backend::reduce(val, accum, op, &u->vector_, desc_t);
}

/*!
 * Reduction of matrix to form scalar
 *   val = val + \sum_{i,j} A(i,j) for all i,j    +: accum
 *                                              sum: op
 */
template <typename T, typename a,
          typename BinaryOpT,     typename MonoidT>
Info reduce(T*               val,
            BinaryOpT        accum,
            MonoidT          op,
            const Matrix<a>* A,
            Descriptor*      desc) {
  if (val == NULL || A == NULL)
    return GrB_UNINITIALIZED_OBJECT;

  backend::Descriptor* desc_t = (desc == NULL) ? NULL : &desc->descriptor_;

  return backend::reduce(val, accum, op, &A->matrix_, desc_t);
}

/*!
 * Matrix transposition
 *   C = C + mask .* (A^T)    +: accum
 *                           .*: Boolean and
 */
template <typename c, typename m, typename a,
          typename BinaryOpT>
Info transpose(Matrix<c>*       C,
               const Matrix<m>* mask,
               BinaryOpT        accum,
               const Matrix<a>* A,
               Descriptor*      desc) {
  std::cout << "Error: transpose not implemented yet!\n";
}

/*!
 * Trace of matrix-matrix product
 *   val = Tr(A * B^T)    *: op
 *
 * Note: A and B are assumed to be square matrices
 */
template <typename T, typename a, typename b,
          typename SemiringT>
Info traceMxmTranspose(T*               val,
                       SemiringT        op,
                       const Matrix<a>* A,
                       const Matrix<b>* B,
                       Descriptor*      desc) {
  if (val == NULL || A == NULL || B == NULL)
    return GrB_UNINITIALIZED_OBJECT;

  backend::Descriptor* desc_t = (desc == NULL) ? NULL : &desc->descriptor_;

  return backend::traceMxmTranspose(val, op, &A->matrix_, &B->matrix_, desc_t);
}

/*!
 * Multiply matrix by scalar
 *   B = A * val    *: op
 */
template <typename b, typename a, typename T,
          typename BinaryOpT>
Info scale(Matrix<b>*       B,
           BinaryOpT        op,
           const Matrix<a>* A,
           T                val,
           Descriptor*      desc) {
  std::cout << "Error: scale not implemented yet!\n";
}

/*!
 * Multiply vector by scalar
 *   w = u * val    *: op
 */
template <typename W, typename U, typename T,
          typename BinaryOpT>
Info scale(Vector<W>*       w,
           BinaryOpT        op,
           const Vector<U>* u,
           T                val,
           Descriptor*      desc) {
  std::cout << "Error: scale not implemented yet!\n";
}

/*!
 * Scatter constant to indices (vector u) to another vector w
 *   w[u] = val
 */
template <typename W, typename M, typename U, typename T>
Info scatter(Vector<W>*       w,
             const Vector<M>* mask,
             const Vector<U>* u,
             T                val,
             Descriptor*      desc) {
  if (u == NULL || w == NULL)
    return GrB_UNINITIALIZED_OBJECT;

  const backend::Vector<M>* mask_t = (mask == NULL) ? NULL : &mask->vector_;
  backend::Descriptor*      desc_t = (desc == NULL) ? NULL : &desc->descriptor_;

  return backend::scatter(&w->vector_, mask_t, &u->vector_, val, desc_t);
}

template <typename W, typename a>
Info graphColor(Vector<W>*       w,
                const Matrix<a>* A,
                Descriptor*      desc) {
  if (A == NULL || w == NULL)
    return GrB_UNINITIALIZED_OBJECT;

  backend::Descriptor* desc_t = (desc == NULL) ? NULL : &desc->descriptor_;

  return backend::graphColor(&w->vector_, &A->matrix_, desc_t);
}

/*!
 * Fused apply & vector-matrix product
 *   w^T = w^T + mask^T .* (u^T * A)    +: accum
 *                                      *: op
 *                                     .*: Boolean and
 */
template <typename W, typename M, typename U, typename a,
          typename BinaryOpT, typename SemiringT>
Info applyVxm(Vector<W>*       w,
              const Vector<M>* mask,
              BinaryOpT        accum,
              SemiringT        op,
              const Vector<U>* u,
              const Matrix<a>* A,
              Descriptor*      desc) {
  // Null pointer check
  if (w == NULL || u == NULL || A == NULL || desc == NULL)
    return GrB_UNINITIALIZED_OBJECT;

  // Dimension check
  Index u_nvals = 0;
  CHECK(u->nvals(&u_nvals));
  if (u_nvals == 0)
    return GrB_UNINITIALIZED_OBJECT;

  // Case 1: u*A
  CHECK(checkDimRowSize(A,  u,    "A.nrows != u.size"));
  CHECK(checkDimColSize(A,  w,    "A.ncols != w.size"));
  CHECK(checkDimSizeSize(w, mask, "w.size  != mask.size"));

  // Case 2: u*AT
  const backend::Vector<M>* mask_t = (mask == NULL) ? NULL : &mask->vector_;
  backend::Descriptor*      desc_t = (desc == NULL) ? NULL : &desc->descriptor_;

  return backend::applyVxm<W, U, a, M>(&w->vector_, mask_t, accum, op,
      &u->vector_, &A->matrix_, desc_t);
}

}  // namespace graphblas

#endif  // GRAPHBLAS_OPERATIONS_HPP_
