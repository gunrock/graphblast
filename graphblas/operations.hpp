#ifndef GRB_OPERATIONS_HPP
#define GRB_OPERATIONS_HPP

#define __GRB_BACKEND_OPERATIONS_HEADER <graphblas/backend/__GRB_BACKEND_ROOT/operations.hpp>
#include __GRB_BACKEND_OPERATIONS_HEADER
#undef __GRB_BACKEND_OPERATIONS_HEADER

#include "graphblas/dimension.hpp"
#include "graphblas/Matrix.hpp"
#include "graphblas/Descriptor.hpp"

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
    // Case 1: A *B
    CHECK( checkDimRowCol( B, A,    "B.nrows != A.ncols"    ) );
    CHECK( checkDimRowRow( A, C,    "A.nrows != C.nrows"    ) );
    CHECK( checkDimColCol( B, C,    "B.ncols != C.ncols"    ) );
    CHECK( checkDimRowRow( C, mask, "C.nrows != mask.nrows" ) );
    CHECK( checkDimColCol( C, mask, "C.ncols != mask.ncols" ) );
    // Case 2: AT*B
    // Case 3: A *BT
    // Case 4: AT*BT

    int variant_t = 0;
    variant_t |= (mask==NULL)  ? 0 : 4;
    variant_t |= (accum==NULL) ? 0 : 2;
    variant_t |= (desc==NULL)  ? 0 : 1;
    const int variant = static_cast<int>(variant_t);

    switch( variant )
    {
      case 0:
        return backend::mxm<0>( C->matrix_, NULL, NULL, op->op_, A->matrix_, 
            B->matrix_, NULL );
      case 1:
        return backend::mxm<1>( C->matrix_, NULL, NULL, op->op_, A->matrix_, 
            B->matrix_, desc->descriptor_ );
      case 2:
        return backend::mxm<2>( C->matrix_, NULL, accum->op_, op->op_, 
            A->matrix_, B->matrix_, NULL );
      case 3:
        return backend::mxm<3>( C->matrix_, NULL, accum->op_, op->op_, 
            A->matrix_, B->matrix_, desc->descriptor_ );
      case 4:
        return backend::mxm<4>( C->matrix_, mask->matrix_, NULL, op->op_, 
            A->matrix_, B->matrix_, NULL );
      case 5:
        return backend::mxm<5>( C->matrix_, mask->matrix_, NULL, op->op_, 
            A->matrix_, B->matrix_, desc->descriptor_ );
      case 6:
        return backend::mxm<6>( C->matrix_, mask->matrix_, accum->op_, 
            op->op_, A->matrix_, B->matrix_, NULL );
      case 7:
        return backend::mxm<7>( C->matrix_, mask->matrix_, accum->op_, op->op_, 
            A->matrix_, B->matrix_, desc->descriptor_ );
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
    // Null pointer check
    if( w==NULL || op==NULL || u==NULL || A==NULL )
      return GrB_UNINITIALIZED_OBJECT;

    // Dimension check
    // Case 1: u*A
    CHECK( checkDimRowSize(  A, u,    "A.nrows != u.size"    ) );
    CHECK( checkDimColSize(  A, w,    "A.ncols != w.size"    ) );
    CHECK( checkDimSizeSize( w, mask, "w.size  != mask.size" ) );

    // Case 2: u*AT

    int variant_t = 0;
    variant_t |= (mask==NULL)  ? 0 : 4;
    variant_t |= (accum==NULL) ? 0 : 2;
    variant_t |= (desc==NULL)  ? 0 : 1;
    const int variant = static_cast<int>(variant_t);

    switch( variant )
    {
      case 0:
        return backend::vxm<0>( w->vector_, NULL, NULL, op->op_, u->vector_, 
            A->matrix_, NULL );
      case 1:
        return backend::vxm<1>( w->vector_, NULL, NULL, op->op_, u->vector_, 
            A->matrix_, desc->descriptor_ );
      case 2:
        return backend::vxm<2>( w->vector_, NULL, accum->op_, op->op_, 
            u->vector_, A->matrix_, NULL );
      case 3:
        return backend::vxm<3>( w->vector_, NULL, accum->op_, op->op_, 
            u->vector_, A->matrix_, desc->descriptor_ );
      case 4:
        return backend::vxm<4>( w->vector_, mask->matrix_, NULL, op->op_, 
            u->vector_, A->matrix_, NULL );
      case 5:
        return backend::vxm<5>( w->vector_, mask->matrix_, NULL, op->op_, 
            u->vector_, A->matrix_, desc->descriptor_ );
      case 6:
        return backend::vxm<6>( w->vector_, mask->matrix_, accum->op_, op->op_,
            u->vector_, A->matrix_, NULL );
      case 7:
        return backend::vxm<7>( w->vector_, mask->matrix_, accum->op_, op->op_, 
            u->vector, A->matrix_, desc->descriptor_ );
    }
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
    // Null pointer check
    if( w==NULL || op==NULL || u==NULL || A==NULL )
      return GrB_UNINITIALIZED_OBJECT;

    // Dimension check
    // Case 1: A *u
    CHECK( checkDimColSize(  A, u,    "A.ncols != u.size"    ) );
    CHECK( checkDimRowSize(  A, w,    "A.nrows != w.size"    ) );
    CHECK( checkDimSizeSize( w, mask, "w.size  != mask.size" ) );

    // Case 2: AT*u

    int variant_t = 0;
    variant_t |= (mask==NULL)  ? 0 : 4;
    variant_t |= (accum==NULL) ? 0 : 2;
    variant_t |= (desc==NULL)  ? 0 : 1;
    const int variant = static_cast<int>(variant_t);

    switch( variant )
    {
      case 0:
        return backend::mxv<0>( w->vector_, NULL, NULL, op->op_, A->matrix_,
            u->vector_, NULL );
      case 1:
        return backend::mxv<1>( w->vector_, NULL, NULL, op->op_, A->matrix_,
            u->vector_, desc->descriptor_ );
      case 2:
        return backend::mxv<2>( w->vector_, NULL, accum->op_, op->op_, 
            A->matrix_, u->vector_, NULL );
      case 3:
        return backend::mxv<3>( w->vector_, NULL, accum->op_, op->op_, 
            A->matrix_, u->vector_, desc->descriptor_ );
      case 4:
        return backend::mxv<4>( w->vector_, mask->matrix_, NULL, op->op_, 
            A->matrix_, u->vector_, NULL );
      case 5:
        return backend::mxv<5>( w->vector_, mask->matrix_, NULL, op->op_, 
            A->matrix_, u->vector_, desc->descriptor_ );
      case 6:
        return backend::mxv<6>( w->vector_, mask->matrix_, accum->op_, op->op_,
            A->matrix_, u->vector_, NULL );
      case 7:
        return backend::mxv<7>( w->vector_, mask->matrix_, accum->op_, op->op_, 
            A->matrix_, u->vector_, desc->descriptor_ );
    }
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
    // Null pointer check
    if( w==NULL || op==NULL || u==NULL || v==NULL )
      return GrB_UNINITIALIZED_OBJECT;

    // Dimension check
    CHECK( checkDimSizeSize( u, v,    "u.size != v.size"    ) );
    CHECK( checkDimSizeSize( u, mask, "u.size != mask.size" ) );
    CHECK( checkDimSizeSize( v, mask, "v.size != mask.size" ) );
    CHECK( checkDimSizeSize( w, mask, "w.size != mask.size" ) );

    int variant_t = 0;
    variant_t |= (mask==NULL)  ? 0 : 4;
    variant_t |= (accum==NULL) ? 0 : 2;
    variant_t |= (desc==NULL)  ? 0 : 1;
    const int variant = static_cast<int>(variant_t);

    switch( variant )
    {
      case 0:
        return backend::mxv<0>( w->vector_, NULL, NULL, op->op_, u->vector_,
            v->vector_, NULL );
      case 1:
        return backend::mxv<1>( w->vector_, NULL, NULL, op->op_, u->vector_,
            v->vector_, desc->descriptor_ );
      case 2:
        return backend::mxv<2>( w->vector_, NULL, accum->op_, op->op_, 
            u->vector_, v->vector_, NULL );
      case 3:
        return backend::mxv<3>( w->vector_, NULL, accum->op_, op->op_, 
            u->vector_, v->vector_, desc->descriptor_ );
      case 4:
        return backend::mxv<4>( w->vector_, mask->matrix_, NULL, op->op_, 
            u->vector_, v->vector_, NULL );
      case 5:
        return backend::mxv<5>( w->vector_, mask->matrix_, NULL, op->op_, 
            u->vector_, v->vector_, desc->descriptor_ );
      case 6:
        return backend::mxv<6>( w->vector_, mask->matrix_, accum->op_, op->op_,
            u->vector_, v->vector_, NULL );
      case 7:
        return backend::mxv<7>( w->vector_, mask->matrix_, accum->op_, op->op_, 
            u->vector_, v->vector_, desc->descriptor_ );
    }
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
    // Null pointer check
    if( w==NULL )
      return GrB_UNINITIALIZED_OBJECT;

    // Dimension check
    // -only have one case (no transpose option)
    CHECK( checkDimSizeSize( w, mask, "w.size  != mask.size" ) );

    int variant_t = 0;
    variant_t |= (indices==NULL) ? 0 : 8;
    variant_t |= (mask==NULL)    ? 0 : 4;
    variant_t |= (accum==NULL)   ? 0 : 2;
    variant_t |= (desc==NULL)    ? 0 : 1;
    const int variant = static_cast<int>(variant_t);

    switch( variant )
    {
      case 0:
        return backend::assign<0>( w->vector_, GrB_NULL, GrB_NULL, val, 
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
