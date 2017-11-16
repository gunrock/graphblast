#ifndef GRB_BACKEND_APSPIE_OPERATIONS_HPP
#define GRB_BACKEND_APSPIE_OPERATIONS_HPP

#include "graphblas/backend/apspie/spgemm.hpp"
#include "graphblas/backend/apspie/spmm.hpp"
#include "graphblas/backend/apspie/gemm.hpp"
#include "graphblas/backend/apspie/spmspv.hpp"
#include "graphblas/backend/apspie/spmv.hpp"
#include "graphblas/backend/apspie/gemv.hpp"

namespace graphblas
{
namespace backend
{
  template <int variant, typename c, typename a, typename b, typename m,
            typename BinaryOpT,      typename SemiringT>
  Info mxm( Matrix<c>*              C,
            const Matrix<m>*  mask,
            const BinaryOpT*  accum,
            const SemiringT*  op,
            const Matrix<a>*  A,
            const Matrix<b>*  B,
            const Descriptor* desc )
  {
    Storage A_mat_type;
    Storage B_mat_type;
    CHECK( A->getStorage( &A_mat_type ) );
    CHECK( B->getStorage( &B_mat_type ) );

    Matrix<m>* maskMatrix = (mask==NULL) ? NULL : mask->getMatrix();

    if( A_mat_type==GrB_SPARSE && B_mat_type==GrB_SPARSE )
    {
      CHECK( C->setStorage( GrB_SPARSE ) );
      CHECK( spgemm<variant>( C->getMatrix(), maskMatrix, accum, op,
          A->getMatrix(), B->getMatrix(), desc ) );
    }
    else
    {
      CHECK( C->setStorage( GrB_DENSE ) );
      if( A_mat_type==GrB_SPARSE && B_mat_type==GrB_DENSE )
      {
        CHECK( spmm<variant>( C->getMatrix(), maskMatrix, accum, op, 
            A->getMatrix(), B->getMatrix(), desc ) );
      }
      else if( A_mat_type==GrB_DENSE && B_mat_type==GrB_SPARSE )
      {
        CHECK( spmm<variant>( C->getMatrix(), maskMatrix, accum, op, 
            A->getMatrix(), B->getMatrix(), desc ) );
      }
      else
      {
        CHECK( gemm<variant>( C->getMatrix(), maskMatrix, accum, op, 
            A->getMatrix(), B->getMatrix(), desc ) );
      }
    }
    return GrB_SUCCESS;
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
    // Get storage
    Storage u_vec_type;
    Storage A_mat_type;
    CHECK( u->getStorage( &u_vec_type ) );
    CHECK( A->getStorage( &A_mat_type ) );

    // Get ratio of filled elements
    Index u_nvals;
    Index u_nsize;
    CHECK( u->nvals( &u_nvals ) );
    CHECK( u->size(  &u_nsize ) );

    Vector<m>* maskVector = (mask==NULL) ? NULL : mask->getVector();

    // 3 cases:
    // 1) SpMSpV: SpMat x SpVec (fewer elements than GrB_THRESHOLD)
    // 2a)SpMV:   SpMat x DeVec
    //  b)SpMV:   SpMat x SpVec (more elements than GrB_THRESHOLD)
    //    -convert SpVec->DeVec
    // 3) GeMV:   DeMat x DeVec
    if( A_mat_type==GrB_SPARSE && 
      ( u_vec_type==GrB_SPARSE && u_nvals/u_nsize<GrB_THRESHOLD ) )
    {
      CHECK( w->setStorage( GrB_SPARSE ) );
      CHECK( spmspv<variant>( w->getVector(), maskMatrix, accum, op,
          A->getMatrix(), u->getVector(), desc ) );
    }
    else
    {
      CHECK( w->setStorage( GrB_DENSE ) );
      if( A_mat_type==GrB_SPARSE )
      {
        if( u_vec_type==GrB_SPARSE )
          CHECK( u->sparse2dense() );
        CHECK( spmv<variant>( w->getVector(), maskMatrix, accum, op, 
            A->getMatrix(), u->getVector(), desc ) );
      }
      else
      {
        Descriptor desc_t;
        desc_t.set( GrB_INP1, GrB_TRAN );
        if( desc==NULL ) 
          desc = desc_t;
        else 
          CHECK( desc.toggle( GrB_INP1 ) );

        CHECK( gemv<variant>( w->getVector(), maskMatrix, accum, op, 
          A->getMatrix(), u->getVector(), desc ) );
      }
    }
    return GrB_SUCCESS;
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
    // Get storage
    Storage u_vec_type;
    Storage A_mat_type;
    CHECK( u->getStorage( &u_vec_type ) );
    CHECK( A->getStorage( &A_mat_type ) );

    // Get ratio of filled elements
    Index u_nvals;
    Index u_nsize;
    CHECK( u->nvals( &u_nvals ) );
    CHECK( u->size(  &u_nsize ) );

    Vector<m>* maskVector = (mask==NULL) ? NULL : mask->getVector();

    if( A_mat_type==GrB_SPARSE && 
      ( u_vec_type==GrB_SPARSE && u_nvals/u_nsize<GrB_THRESHOLD ) )
    {
      CHECK( w->setStorage( GrB_SPARSE ) );
      CHECK( spmspv<variant>( w->getVector(), maskMatrix, accum, op,
          A->getMatrix(), u->getVector(), desc ) );
    }
    else
    {
      CHECK( w->setStorage( GrB_DENSE ) );
      if( A_mat_type==GrB_DENSE )
      {
        CHECK( spmv<variant>( w->getVector(), maskMatrix, accum, op, 
            A->getMatrix(), u->getVector(), desc ) );
      }
      else
      {
        CHECK( gemv<variant>( w->getVector(), maskMatrix, accum, op, 
          A->getMatrix(), u->getVector(), desc ) );
      }
    }
    return GrB_SUCCESS;
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

}  // backend
}  // graphblas

#endif  // GRB_BACKEND_APSPIE_OPERATIONS_HPP
