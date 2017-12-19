#ifndef GRB_BACKEND_APSPIE_SPMSPV_HPP
#define GRB_BACKEND_APSPIE_SPMSPV_HPP

#include <iostream>

#include "graphblas/backend/apspie/Descriptor.hpp"
#include "graphblas/backend/apspie/SparseMatrix.hpp"
#include "graphblas/backend/apspie/DenseMatrix.hpp"
#include "graphblas/backend/apspie/operations.hpp"
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
    const Index* Ah_csrRowPtr= (!use_tran) ? A->h_cscColPtr_ : A->h_csrRowPtr_;

    // Get descriptor parameter on which one to use
    Desc_value spmspv_mode;
    CHECK( desc->get(GrB_SPMSPVMODE, &spmspv_mode) );

    // Only difference between masked and unmasked versions if whether
    // eWiseMult() is called afterwards or not
    if( use_mask )
    {
      // temp_ind and temp_val need |V| memory
      desc->resize(2*A_nrows);
      Index* temp_ind = (Index*) desc->d_buffer_;
      T*     temp_val = (T*)     desc->d_buffer_+A_nrows;

      if( spmspv_mode==GrB_APSPIE )
        spmspvApspie<false,false,false>(
            temp_ind, temp_val, NULL, op->identity(),
            op->mul_, op->add_, A_nrows, A->nvals_,
            A_csrRowPtr, A_csrColInd, A_csrVal, u->d_ind_, u->d_val_, desc );
      else if( spmspv_mode==GrB_APSPIELB )
        spmspvApspieLB<false,false,false>(
            temp_ind, temp_val, NULL, op->identity(),
            op->mul_, op->add_, A_nrows, A->nvals_,
            A_csrRowPtr, A_csrColInd, A_csrVal, u->d_val_, desc );
      else if( spmspv_mode==GrB_GUNROCKLB )
        spmspvGunrockLB<false,false,false>(
            temp_ind, temp_val, NULL, op->identity(),
            op->mul_, op->add_, A_nrows, A->nvals_,
            A_csrRowPtr, A_csrColInd, A_csrVal, u->d_ind_, u->d_val_, desc );
      else if( spmspv_mode==GrB_GUNROCKTWC )
        spmspvGunrockTWC<false,false,false>(
            temp_ind, temp_val, NULL, op->identity(),
            op->mul_, op->add_, A_nrows, A->nvals_,
            A_csrRowPtr, A_csrColInd, A_csrVal, u->d_ind_, u->d_val_, desc );
      //eWiseMultKernel<<<NB,NT>>>( 
      //    w->d_ind_, w->d_val_, NULL, NULL, op, temp_ind, temp_val, 
      //    mask->d_ind_, mask->d_val_ );
      //filterKernel<<<NB,NT>>>(w->d_val,mask->d_val_);
      //streamCompactKernel<<<NB,NT>(w->d_val);
    }
    else
    {
      if( spmspv_mode==GrB_APSPIE )
        spmspvApspie<false,false,false>(
            w->d_ind_, w->d_val_, NULL, op->identity(),
            op->mul_, op->add_, A_nrows, A->nvals_,
            A_csrRowPtr, A_csrColInd, A_csrVal, u->d_ind_, u->d_val_, desc );
      else if( spmspv_mode==GrB_APSPIELB )
        spmspvApspieLB<false,false,false>(
            w->d_ind_, w->d_val_, NULL, op->identity(),
            op->mul_, op->add_, A_nrows, A->nvals_,
            A_csrRowPtr, A_csrColInd, A_csrVal, u->d_ind_, u->d_val_, desc );
      else if( spmspv_mode==GrB_GUNROCKLB )
        spmspvGunrockLB<false,false,false>(
            w->d_ind_, w->d_val_, NULL, op->identity(),
            op->mul_, op->add_, A_nrows, A->nvals_,
            A_csrRowPtr, A_csrColInd, A_csrVal, u->d_ind_, u->d_val_, desc );
      else if( spmspv_mode==GrB_GUNROCKTWC )
        spmspvGunrockTWC<false,false,false>(
            w->d_ind_, w->d_val_, NULL, op->identity(),
            op->mul_, op->add_, A_nrows, A->nvals_,
            A_csrRowPtr, A_csrColInd, A_csrVal, u->d_ind_, u->d_val_, desc );
    }
    w->need_update_ = true;
    return GrB_SUCCESS;
  }

}  // backend
}  // graphblas

#endif  // GRB_BACKEND_APSPIE_SPMSPV_HPP
