#ifndef GRB_BACKEND_APSPIE_OPERATIONS_HPP
#define GRB_BACKEND_APSPIE_OPERATIONS_HPP

#include "graphblas/backend/apspie/spgemm.hpp"
#include "graphblas/backend/apspie/gemm.hpp"
#include "graphblas/backend/apspie/spmm.hpp"

namespace graphblas
{
namespace backend
{
  template <int variant, typename c, typename a, typename b, typename m,
            typename BinaryOpT,     typename SemiringT>
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
    A->getStorage( &A_mat_type );
    B->getStorage( &B_mat_type );

    Matrix<m>* maskMatrix = (mask==NULL) ? NULL : mask->getMatrix();

    if( A_mat_type==GrB_SPARSE && B_mat_type==GrB_SPARSE )
    {
      C->setStorage( GrB_SPARSE );
      spgemm<variant>( C->getMatrix(), maskMatrix, accum, op, A->getMatrix(), 
          B->getMatrix(), desc );
    }
    else
    {
      C->setStorage( GrB_DENSE );
      if( A_mat_type==GrB_DENSE && B_mat_type==GrB_DENSE )
      {
        std::cout << "Error: Feature not implemented yet!\n";
        gemm<variant>( C->getMatrix(), maskMatrix, accum, op, A->getMatrix(),
            B->getMatrix(), desc );
      }
      else if( A_mat_type==GrB_SPARSE && B_mat_type==GrB_DENSE )
      {
        spmm<variant>( C->getMatrix(), maskMatrix, accum, op, A->getMatrix(),
            B->getMatrix(), desc );
      }
      else
      {
        spmm<variant>( C->getMatrix(), maskMatrix, accum, op, A->getMatrix(),
            B->getMatrix(), desc );
      }
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

}  // backend
}  // graphblas

#endif  // GRB_BACKEND_APSPIE_OPERATIONS_HPP
