#ifndef GRB_OPERATIONS_HPP
#define GRB_OPERATIONS_HPP

#define __GRB_BACKEND_OPERATIONS_HEADER <graphblas/backend/__GRB_BACKEND_ROOT/operations.hpp>
#include __GRB_BACKEND_OPERATIONS_HEADER
#undef __GRB_BACKEND_OPERATIONS_HEADER

#include "graphblas/types.hpp"
#include "graphblas/dimension.hpp"
#include "graphblas/Matrix.hpp"
#include "graphblas/Descriptor.hpp"

namespace graphblas
{
  // TODO: make all operations mxm, mxv, etc. follow vxm() and assign() 
  // constant variant 

  template <typename c, typename a, typename b, typename m, 
            typename BinaryOpT,     typename SemiringT>
  Info mxm( Matrix<c>*       C,
            const Matrix<m>* mask,
            const BinaryOpT* accum,
            const SemiringT* op,
            const Matrix<a>* A,
            const Matrix<b>* B,
            Descriptor*      desc )
  {
    // Null pointer check
    if( C==NULL || op==NULL || A==NULL || B==NULL || desc==NULL )
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

    backend::Matrix<m>*  mask_t = (mask==NULL ) ? NULL : &mask->matrix_;
    auto                accum_t = (accum==NULL) ? NULL : &accum->op_;
    backend::Descriptor* desc_t = (desc==NULL ) ? NULL : &desc->descriptor_;

    return backend::mxm( &C->matrix_, mask_t, accum_t, &op->op_, &A->matrix_, 
        &B->matrix_, desc_t );
  }

  template <typename W, typename U, typename a>
  Info vxm( Vector<W>*             w,
            const Vector<U>*       mask,
            const BinaryOp<a,a,a>* accum,
            const Semiring<a,a,a>* op,
            const Vector<U>*       u,
            const Matrix<a>*       A,
            Descriptor*            desc )
  {
    // Null pointer check
    if( w==NULL || op==NULL || u==NULL || A==NULL || desc==NULL )
      return GrB_UNINITIALIZED_OBJECT;

    // Dimension check
    Index u_nvals = 0;
    CHECK( u->nvals(&u_nvals) );
    if( u_nvals==0 )
    {
      if( desc->descriptor_.debug() )
        std::cout << "u.nvals == 0\n";
      CHECK( w->dup(u) );
      return GrB_SUCCESS;
    }
    //CHECK( checkDimVecNvals(  u,    "u.nvals == 0"    ) );

    // Case 1: u*A
    CHECK( checkDimRowSize(  A, u,    "A.nrows != u.size"    ) );
    CHECK( checkDimColSize(  A, w,    "A.ncols != w.size"    ) );
    if( mask!=NULL )
    CHECK( checkDimSizeSize( w, mask, "w.size  != mask.size" ) );

    // Case 2: u*AT

    const backend::Vector<U>*        mask_t = (mask==NULL ) ? NULL 
        : &mask->vector_;
    const backend::BinaryOp<a,a,a>* accum_t = (accum==NULL) ? NULL 
        : &accum->op_;
    backend::Descriptor*             desc_t = (desc==NULL ) ? NULL 
        : &desc->descriptor_;

    //std::cout << "graphblas: " << (mask_t!=NULL) << std::endl;
    return backend::vxm<W,U,a>( &w->vector_, mask_t, accum_t, &op->op_, 
        &u->vector_, &A->matrix_, desc_t );
  }

  template <typename W, typename a, typename U>
  Info mxv( Vector<W>*       w,
            const Vector<U>* mask,
            const BinaryOp<a,a,a>* accum,
            const Semiring<a,a,a>* op,
            const Matrix<a>* A,
            const Vector<U>* u,
            Descriptor*      desc )
  {
    // Null pointer check
    if( w==NULL || op==NULL || u==NULL || A==NULL || desc==NULL )
      return GrB_UNINITIALIZED_OBJECT;

    // Dimension check
    Index u_nvals = 0;
    CHECK( u->nvals(&u_nvals) );
    if( u_nvals==0 )
    {
      if( desc->descriptor_.debug() )
        std::cout << "u.nvals == 0\n";
      CHECK( w->dup(u) );
      return GrB_SUCCESS;
    }
    //CHECK( checkDimVecNvals( u, "u.nvals == 0" ) );

    // Case 1: A *u
    CHECK( checkDimColSize(  A, u,    "A.ncols != u.size"    ) );
    CHECK( checkDimRowSize(  A, w,    "A.nrows != w.size"    ) );
    CHECK( checkDimSizeSize( w, mask, "w.size  != mask.size" ) );

    // Case 2: AT*u

    const backend::Vector<U>*        mask_t = (mask==NULL ) ? NULL 
        : &mask->vector_;
    const backend::BinaryOp<a,a,a>* accum_t = (accum==NULL) ? NULL 
        : &accum->op_;
    backend::Descriptor*             desc_t = (desc==NULL ) ? NULL 
        : &desc->descriptor_;

    return backend::mxv( &w->vector_, mask_t, accum_t, &op->op_, &A->matrix_,
        &u->vector_, desc_t );
  }

  template <typename W, typename U, typename V, typename M,
            typename BinaryOpT,     typename SemiringT>
  Info eWiseMult( Vector<W>*       w,
                  const Vector<M>* mask,
                  const BinaryOpT* accum,
                  const SemiringT* op,
                  const Vector<U>* u,
                  const Vector<V>* v,
                  Descriptor*      desc )
  {
    // Use either op->operator() or op->mul() as the case may be
    // Null pointer check
    if( w==NULL || op==NULL || u==NULL || v==NULL || desc==NULL )
      return GrB_UNINITIALIZED_OBJECT;

    // Dimension check
    CHECK( checkDimSizeSize( u, v,    "u.size != v.size"    ) );
    CHECK( checkDimSizeSize( u, mask, "u.size != mask.size" ) );
    CHECK( checkDimSizeSize( v, mask, "v.size != mask.size" ) );
    CHECK( checkDimSizeSize( w, mask, "w.size != mask.size" ) );

    backend::Vector<M>*  mask_t = (mask==NULL ) ? NULL : &mask->vector_;
    auto                accum_t = (accum==NULL) ? NULL : &accum->op_;
    backend::Descriptor* desc_t = (desc==NULL ) ? NULL : &desc->descriptor_;

    return backend::eWiseMult( &w->vector_, mask_t, accum_t, &op->op_, 
        &u->vector_, &v->vector_, desc_t );
  }

  template <typename c, typename a, typename b, typename m,
            typename BinaryOpT,     typename SemiringT>
  Info eWiseMult( Matrix<c>*       C,
                  const Matrix<m>* mask,
                  const BinaryOpT* accum,
                  const SemiringT* op,
                  const Matrix<a>* A,
                  const Matrix<b>* B,
                  Descriptor*      desc )
  {
    // Use either op->operator() or op->mul() as the case may be
  }

  template <typename W, typename U, typename V, typename M,
            typename BinaryOpT,     typename SemiringT>
  Info eWiseAdd( Vector<W>*       w,
                 const Vector<M>* mask,
                 const BinaryOpT* accum,
                 const SemiringT* op,
                 const Vector<U>* u,
                 const Vector<V>* v,
                 Descriptor*      desc )
  {
    // Use either op->operator() or op->add() as the case may be
  }

  template <typename c, typename a, typename b, typename m,
            typename BinaryOpT,     typename SemiringT>
  Info eWiseAdd( Matrix<c>*       C,
                 const Matrix<m>* mask,
                 const BinaryOpT* accum,
                 const SemiringT* op,
                 const Matrix<a>* A,
                 const Matrix<b>* B,
                 Descriptor*      desc )
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
                Descriptor*               desc )
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
                Descriptor*               desc )
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
                Descriptor*               desc )
  {

  }

  template <typename W, typename U, typename M,
            typename BinaryOpT>
  Info assign( Vector<W>*                w,
               const Vector<U>*          mask,
               const BinaryOpT*          accum,
               const Vector<U>*          u,
               const std::vector<Index>* indices,
               Index                     nindices,
               Descriptor*               desc )
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
               Descriptor*               desc )
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
               Descriptor*               desc )
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
               Descriptor*               desc )
  {

  }

  template <typename W, typename T>
  Info assign( Vector<W>*                w,
               const Vector<W>*          mask,
               const BinaryOp<W,W,W>*    accum,
               T                         val,
               const std::vector<Index>* indices,
               Index                     nindices,
               Descriptor*               desc )
  {
    // Null pointer check
    if( w==NULL || desc==NULL )
      return GrB_UNINITIALIZED_OBJECT;

    // Dimension check
    // -only have one case (no transpose option)
    CHECK( checkDimSizeSize( w, mask, "w.size  != mask.size" ) );

    auto                 mask_t = (mask==NULL ) ? NULL : &mask->vector_;
    auto                accum_t = (accum==NULL) ? NULL : &accum->op_;
    backend::Descriptor* desc_t = (desc==NULL ) ? NULL : &desc->descriptor_;

    return backend::assign( &w->vector_, mask_t, accum_t, val, indices, 
        nindices, desc_t );
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
               Descriptor*               desc )
  {

  }

  template <typename W, typename U, typename M,
            typename BinaryOpT,     typename UnaryOpT>
  Info apply( Vector<W>*       w,
              const Vector<M>* mask,
              const BinaryOpT* accum,
              const UnaryOpT*  op,
              const Vector<U>* u,
              Descriptor*      desc )
  {

  }

  template <typename c, typename a, typename m,
            typename BinaryOpT,     typename UnaryOpT>
  Info apply( Matrix<c>*       C,
              const Matrix<m>* mask,
              const BinaryOpT* accum,
              const UnaryOpT*  op,
              const Matrix<a>* A,
              Descriptor*      desc )
  {

  }

  template <typename W, typename a, typename M,
            typename BinaryOpT,     typename MonoidT>
  Info reduce( Vector<W>*       w,
               const Vector<M>* mask,
               const BinaryOpT* accum,
               const MonoidT*   op,
               const Matrix<a>  A,
               Descriptor*      desc )
  {
    // Use op->operator()
  }

  template <typename T, typename U>
  Info reduce( T*                     val,
               const BinaryOp<U,U,U>* accum,
               const Monoid<U>*       op,
               const Vector<U>*       u,
               Descriptor*            desc )
  {
    // Null pointer check
    if( val==NULL || u==NULL )
      return GrB_UNINITIALIZED_OBJECT;

    auto                accum_t = (accum==NULL) ? NULL : &accum->op_;
    backend::Descriptor* desc_t = (desc==NULL ) ? NULL : &desc->descriptor_;

    return backend::reduce( val, accum_t, &op->op_, &u->vector_, desc_t );
  }

  template <typename T, typename a,
            typename BinaryOpT,     typename MonoidT>
  Info reduce( T*               val,
               const BinaryOpT* accum,
               const MonoidT*   op,
               const Matrix<a>* A,
               Descriptor*      desc )
  {

  }

  template <typename c, typename a, typename m,
            typename BinaryOpT>
  Info transpose( Matrix<c>*       C,
                  const Matrix<m>* mask,
                  const BinaryOpT* accum,
                  const Matrix<a>* A,
                  Descriptor*      desc )
  {

  }

} // graphblas

#endif  // GRB_OPERATIONS_HPP
