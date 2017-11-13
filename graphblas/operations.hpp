#ifndef GRB_OPERATIONS_HPP
#define GRB_OPERATIONS_HPP

#define __GRB_BACKEND_OPERATIONS_HEADER <graphblas/backend/__GRB_BACKEND_ROOT/operations.hpp>
#include __GRB_BACKEND_OPERATIONS_HEADER
#undef __GRB_BACKEND_OPERATIONS_HEADER

namespace graphblas
{
  template <typename c, typename a, typename b, typename m, 
            typename BinaryOpT,     typename SemiringT>
  Info mxm( Matrix<c>*        C,
            const Matrix<m>*  mask,
            const BinaryOpT*  accum,
            const SemiringT*  op,
            const Matrix<a>*  A,
            const Matrix<b>*  B,
            const Descriptor* desc )
  {
    // Null pointer check
    if( C==NULL || op==NULL || A==NULL || B==NULL )
      return GrB_UNINITIALIZED_OBJECT;

    // Dimension check
    CHECK( checkDimRowCol( B, A,    "B.nrows != A.ncols"    ) );
    CHECK( checkDimRowRow( A, C,    "A.nrows != C.nrows"    ) );
    CHECK( checkDimColCol( B, C,    "B.ncols != C.ncols"    ) );
    CHECK( checkDimRowRow( C, mask, "C.nrows != mask.nrows" ) );
    CHECK( checkDimColCol( C, mask, "C.ncols != mask.ncols" ) );

    int variant = 0;
    variant |= (mask==NULL)  ? 0 : 4;
    variant |= (accum==NULL) ? 0 : 2;
    variant |= (desc==NULL)  ? 0 : 1;

    switch( variant )
    {
      case 0:
        return backend::mxm<variant>( C.matrix_, NULL, NULL, op.op_, A.matrix_, 
            B.matrix_, NULL );
      case 1:
        return backend::mxm<variant>( C.matrix_, NULL, NULL, op.op_, A.matrix_,
            B.matrix_, desc.descriptor_ );
      case 2:
        return backend::mxm<variant>( C.matrix_, NULL, accum.op_, op.op_, 
            A.matrix_, B.matrix_, NULL );
      case 3:
        return backend::mxm<variant>( C.matrix_, NULL, accum.op_, op.op_, 
            A.matrix_, B.matrix_, desc.descriptor_ );
      case 4:
        return backend::mxm<variant>( C.matrix_, mask.matrix_, NULL, op.op_, 
            A.matrix_, B.matrix_, NULL );
      case 5:
        return backend::mxm<variant>( C.matrix_, mask.matrix_, NULL, op.op_, 
            A.matrix_, B.matrix_, desc.descriptor_ );
      case 6:
        return backend::mxm<variant>( C.matrix_, mask.matrix_, accum.op_, 
            op.op_, A.matrix_, B.matrix_, NULL );
      case 7:
        return backend::mxm<variant>( C.matrix_, mask.matrix_, accum.op_, 
            op.op_, A.matrix_, B.matrix_, desc.descriptor_ );
    }
  }

  template <typename W, typename U, typename a, typename M, 
            typename BinaryOpT,     typename SemiringT>
  Info vxm( Vector<W>*        w,
            const Vector<M>*  mask,
            const BinaryOpT*  accum,
            const SemiringT*  op,
            const Vector<U>*  u,
            const Matrix<a>*  A,
            const Descriptor* desc )
  {

  }

  template <typename W, typename a, typename U, typename M, 
            typename BinaryOpT,     typename SemiringT>
  Info mxv( Vector<W>*        w,
            const Vector<M>*  mask,
            const BinaryOpT*  accum,
            const SemiringT*  op,
            const Matrix<a>*  A,
            const Vector<U>*  u,
            const Descriptor* desc )
  {

  }

  template <typename W, typename U, typename V, typename M,
            typename BinaryOpT,     typename SemiringT>
  Info eWiseMult( Vector<W>*              w,
                  const Vector<M>*  mask,
                  const BinaryOpT*  accum,
                  const SemiringT*  op,
                  const Vector<U>*  u,
                  const Vector<V>*  v,
                  const Descriptor* desc )
  {
    // Use either op->operator() or op->mul() as the case may be
  }

  template <typename c, typename a, typename b, typename m,
            typename BinaryOpT,     typename SemiringT>
  Info eWiseMult( Matrix<c>*              C,
                  const Matrix<m>*  mask,
                  const BinaryOpT*  accum,
                  const SemiringT*  op,
                  const Matrix<a>*  A,
                  const Matrix<b>*  B,
                  const Descriptor* desc )
  {
    // Use either op->operator() or op->mul() as the case may be
  }

  template <typename W, typename U, typename V, typename M,
            typename BinaryOpT,     typename SemiringT>
  Info eWiseAdd( Vector<W>*        w,
                 const Vector<M>*  mask,
                 const BinaryOpT*  accum,
                 const SemiringT*  op,
                 const Vector<U>*  u,
                 const Vector<V>*  v,
                 const Descriptor* desc )
  {
    // Use either op->operator() or op->add() as the case may be
  }

  template <typename c, typename a, typename b, typename m,
            typename BinaryOpT,     typename SemiringT>
  Info eWiseAdd( Matrix<c>*        C,
                 const Matrix<m>*  mask,
                 const BinaryOpT*  accum,
                 const SemiringT*  op,
                 const Matrix<a>*  A,
                 const Matrix<b>*  B,
                 const Descriptor* desc )
  {
    // Use either op->operator() or op->add() as the case may be
  }

  template <typename W, typename U, typename M,
            typename BinaryOpT>
  Info extract( Vector<W>*                w,
                const Vector<M>*          mask,
                const BinaryOpT*          accum,
                const Vector<U>*          u,
                const std::vector<Index>* indices,
                Index                     nindices,
                const Descriptor*         desc )
  {

  }

  template <typename c, typename a, typename m,
            typename BinaryOpT>
  Info extract( Matrix<c>*                C,
                const Matrix<m>*          mask,
                const BinaryOpT*          accum,
                const Matrix<a>*          A,
                const std::vector<Index>* row_indices,
                Index                     nrows,
                const std::vector<Index>* col_indices,
                Index                     ncols,
                const Descriptor*         desc )
  {

  }

  template <typename W, typename a, typename M,
            typename BinaryOpT>
  Info extract( Vector<W>*                w,
                const Vector<M>*          mask,
                const BinaryOpT*          accum,
                const Matrix<a>*          A,
                const std::vector<Index>* row_indices,
                Index                     nrows,
                Index                     col_index,
                const Descriptor*         desc )
  {

  }

  template <typename W, typename U, typename M,
            typename BinaryOpT>
  Info assign( Vector<W>*                w,
               const Vector<M>*          mask,
               const BinaryOpT*          accum,
               const Vector<U>*          u,
               const std::vector<Index>* indices,
               Index                     nindices,
               const Descriptor*         desc )
  {

  }

  template <typename c, typename a, typename m,
            typename BinaryOpT>
  Info assign( Matrix<c>*                C,
               const Matrix<m>*          mask,
               const BinaryOpT*          accum,
               const Matrix<a>*          A,
               const std::vector<Index>* row_indices,
               Index                     nrows,
               const std::vector<Index>* col_indices,
               Index                     ncols,
               const Descriptor*         desc )
  {

  }

  template <typename c, typename U, typename M,
            typename BinaryOpT>
  Info assign( Matrix<c>*                C,
               const Vector<M>*          mask,
               const BinaryOpT*          accum,
               const Vector<U>*          u,
               const std::vector<Index>* row_indices,
               Index                     nrows,
               Index                     col_index,
               const Descriptor*         desc )
  {

  }

  template <typename c, typename U, typename M,
            typename BinaryOpT>
  Info assign( Matrix<c>*                C,
               const Vector<M>*          mask,
               const BinaryOpT*          accum,
               const Vector<U>*          u,
               Index                     row_index,
               const std::vector<Index>* col_indices,
               Index                     ncols,
               const Descriptor*         desc )
  {

  }

  template <typename W, typename T, typename M,
            typename BinaryOpT>
  Info assign( Vector<W>*                w,
               const Vector<M>*          mask,
               const BinaryOpT*          accum,
               T                         val,
               const std::vector<Index>* indices,
               Index                     nindices,
               const Descriptor*         desc )
  {

  }

  template <typename c, typename T, typename m,
            typename BinaryOpT>
  Info assign( Matrix<c>*                C,
               const Matrix<m>*          mask,
               const BinaryOpT*          accum,
               T                         val,
               const std::vector<Index>* row_indices,
               Index                     nrows,
               const std::vector<Index>* col_indices,
               Index                     ncols,
               const Descriptor*         desc )
  {

  }

  template <typename W, typename U, typename M,
            typename BinaryOpT,     typename UnaryOpT>
  Info apply( Vector<W>*        w,
              const Vector<M>*  mask,
              const BinaryOpT*  accum,
              const UnaryOpT*   op,
              const Vector<U>*  u,
              const Descriptor* desc )
  {

  }

  template <typename c, typename a, typename m,
            typename BinaryOpT,     typename UnaryOpT>
  Info apply( Matrix<c>*        C,
              const Matrix<m>*  mask,
              const BinaryOpT*  accum,
              const UnaryOpT*   op,
              const Matrix<a>*  A,
              const Descriptor* desc )
  {

  }

  template <typename W, typename a, typename M,
            typename BinaryOpT,     typename MonoidT>
  Info reduce( Vector<W>*        w,
               const Vector<M>*  mask,
               const BinaryOpT*  accum,
               const MonoidT*    op,
               const Matrix<a>   A,
               const Descriptor* desc )
  {
    // Use op->operator()
  }

  template <typename T, typename U,
            typename BinaryOpT,     typename MonoidT>
  Info reduce( T*                val,
               const BinaryOpT*  accum,
               const MonoidT*    op,
               const Vector<U>*  u,
               const Descriptor* desc )
  {

  }

  template <typename T, typename a,
            typename BinaryOpT,     typename MonoidT>
  Info reduce( T*                val,
               const BinaryOpT*  accum,
               const MonoidT*    op,
               const Matrix<a>*  A,
               const Descriptor* desc )
  {

  }

  template <typename c, typename a, typename m,
            typename BinaryOpT>
  Info transpose( Matrix<c>*        C,
                  const Matrix<m>*  mask,
                  const BinaryOpT*  accum,
                  const Matrix<a>*  A,
                  const Descriptor* desc )
  {

  }

}  // graphblas

#endif  // GRB_OPERATIONS_HPP
