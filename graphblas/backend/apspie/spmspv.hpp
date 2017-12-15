#ifndef GRB_BACKEND_APSPIE_SPMSPV_HPP
#define GRB_BACKEND_APSPIE_SPMSPV_HPP

#include <iostream>

#include <cuda.h>
#include <cusparse.h>

#include <moderngpu.cuh>

#include "graphblas/backend/apspie/Descriptor.hpp"
#include "graphblas/backend/apspie/SparseMatrix.hpp"
#include "graphblas/backend/apspie/DenseMatrix.hpp"
#include "graphblas/backend/apspie/kernels/spmspv.hpp"

namespace graphblas
{
namespace backend
{

  template <typename W, typename a, typename U, typename M,
            typename BinaryOpT,      typename SemiringT>
  Info spmspv( SparseVector<W>*       w,
               const DenseVector<M>*  mask,
               const BinaryOpT*       accum,
               const SemiringT*       op,
               const SparseMatrix<a>* A,
               const SparseVector<U>* u,
               Descriptor*            desc )
  {
    std::cout << "Error: Feature not implemented yet!\n";
    return GrB_SUCCESS;
  }

  template <typename W, typename a, typename U, typename M,
            typename BinaryOpT,      typename SemiringT>
  Info spmspv( SparseVector<W>*       w,
               const SparseVector<M>* mask,
               const BinaryOpT*       accum,
               const SemiringT*       op,
               const SparseMatrix<a>* A,
               const SparseVector<U>* u,
               Descriptor*            desc )
  {
    // Get descriptor parameters for SCMP, REPL, TRAN
    Desc_value scmp_mode, repl_mode, inp0_mode, inp1_mode;
    CHECK( desc->get(GrB_MASK, &scmp_mode) );
    CHECK( desc->get(GrB_OUTP, &repl_mode) );
    CHECK( desc->get(GrB_INP0, &inp0_mode) );
    CHECK( desc->get(GrB_INP1, &inp1_mode) );

    // TODO: add accum and replace support
    // -have masked variants as separate kernel
    // -have scmp as template parameter
    // -accum and replace as parts in flow
    bool use_mask = (mask==NULL )            ? false : true;
    bool use_accum= (accum==NULL)            ? false : true;
    bool use_scmp = (scmp_mode==GrB_SCMP)    ? true : false;
    bool use_repl = (repl_mode==GrB_REPLACE) ? true : false;
    bool use_tran = (inp0_mode==GrB_TRAN || inp1_mode==GrB_TRAN) ?
        true : false;

    // Transpose (default is CSC):
    const Index* A_csrRowPtr = (!use_tran) ? A->d_cscColPtr_ : A->d_csrRowPtr_;
    const Index* A_csrColInd = (!use_tran) ? A->d_cscRowInd_ : A->d_csrColInd_;
    const T*     A_csrVal    = (!use_tran) ? A->d_cscVal_    : A->d_csrVal_;
    const Index  A_nrows     = (!use_tran) ? A->ncols_       : A->nrows_;

    // Get descriptor parameters for nthreads
    Desc_value ta_mode, tb_mode, nt_mode;
    CHECK( desc->get(GrB_TA, &ta_mode) );
    CHECK( desc->get(GrB_TB, &tb_mode) );
    CHECK( desc->get(GrB_NT, &nt_mode) );

    const int ta = static_cast<int>(ta_mode);
    const int tb = static_cast<int>(tb_mode);
    const int nt = static_cast<int>(nt_mode);

    dim3 NT, NB;
    NT.x = nt;
    NT.y = 1;
    NT.z = 1;
    NB.x = (ta*A_nrows+nt-1)/nt;
    NB.y = 1;
    NB.z = 1;

    // Only difference between masked and unmasked versions if whether
    // filterKernel() is called afterwards or not
    if( use_mask )
    {
      spmspvKernel<false,false,false><<<NB,NT>>>(
          w->d_val_, NULL, op->identity(),
          mgpu::multiplies<a>(), mgpu::plus<a>(), A_nrows, A->nvals_,
          A_csrRowPtr, A_csrColInd, A_csrVal, u->d_val_ );
      //filterKernel<<<NB,NT>>>(w->d_val,mask->d_val_);
      //streamCompactKernel<<<NB,NT>(w->d_val);
    }
    else
    {
      spmspvKernel<false,false,false><<<NB,NT>>>(
          w->d_val_, NULL, op->identity(),
          mgpu::multiplies<a>(), mgpu::plus<a>(), A_nrows, A->nvals_,
          A_csrRowPtr, A_csrColInd, A_csrVal, u->d_val_ );
    }
    w->need_update_ = true;
    return GrB_SUCCESS;
  }

}  // backend
}  // graphblas

#endif  // GRB_BACKEND_APSPIE_SPMSPV_HPP
