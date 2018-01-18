#ifndef GRB_BACKEND_APSPIE_SPMSPV_HPP
#define GRB_BACKEND_APSPIE_SPMSPV_HPP

#include <iostream>

#include "graphblas/backend/apspie/Descriptor.hpp"
#include "graphblas/backend/apspie/SparseMatrix.hpp"
#include "graphblas/backend/apspie/DenseMatrix.hpp"
#include "graphblas/backend/apspie/operations.hpp"
#include "graphblas/backend/apspie/spmspvInner.hpp"
#include "graphblas/backend/apspie/kernels/assignSparse.hpp"
#include "graphblas/backend/apspie/kernels/assignDense.hpp"
#include "graphblas/backend/apspie/kernels/util.hpp"

namespace graphblas
{
namespace backend
{

  template <typename W, typename a, typename U, typename M,
            typename SemiringT>
  Info spmspv( SparseVector<W>*       w,
               const Vector<M>*       mask,
               const BinaryOp<a,a,a>*       accum,
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
    // -accum and replace as parts in flow
    // -special case of inverting GrB_SCMP since we are using it to zero out
    // values instead of passing them through
    bool use_mask = (mask!=NULL);
    bool use_accum= (accum!=NULL);            //TODO
    bool use_scmp = (scmp_mode!=GrB_SCMP);    //Special case
    bool use_repl = (repl_mode==GrB_REPLACE); //TODO
    bool use_tran = (inp0_mode==GrB_TRAN || inp1_mode==GrB_TRAN);
    bool use_allowdupl; //TODO opt4
    bool use_struconly; //TODO opt5

    if( desc->debug())
    {
      std::cout << "Executing Spmspv\n";
      if( desc->struconly() )
        std::cout << "In structure only mode\n";
      else
        std::cout << "In key-value mode\n";
      printState( use_mask, use_accum, use_scmp, use_repl, use_tran );
    }

    // Transpose (default is CSC):
    const Index* A_csrRowPtr = (!use_tran) ? A->d_cscColPtr_ : A->d_csrRowPtr_;
    const Index* A_csrColInd = (!use_tran) ? A->d_cscRowInd_ : A->d_csrColInd_;
    const T*     A_csrVal    = (!use_tran) ? A->d_cscVal_    : A->d_csrVal_;
    const Index  A_nrows     = (!use_tran) ? A->ncols_       : A->nrows_;
    const Index* Ah_csrRowPtr= (!use_tran) ? A->h_cscColPtr_ : A->h_csrRowPtr_;

    // Get descriptor parameter on which one to use
    Desc_value spmspv_mode;
    CHECK( desc->get(GrB_SPMSPVMODE, &spmspv_mode) );

    // temp_ind and temp_val need |V| memory for masked case, so just allocate 
    // this much memory for now. TODO: optimize for memory
    int size          = (float)A->nvals_*GrB_THRESHOLD+1;
    desc->resize((2*A_nrows+4*size)*max(sizeof(Index),sizeof(T)), "buffer");

    // Only difference between masked and unmasked versions if whether
    // eWiseMult() is called afterwards or not
    if( use_mask )
    {
      // temp_ind and temp_val need |V| memory
      Index* temp_ind   = (Index*) desc->d_buffer_;
      T*     temp_val   = (T*)     desc->d_buffer_+A_nrows;
      Index  temp_nvals = 0;
    
      if( spmspv_mode==GrB_APSPIE )
        spmspvApspie(
            temp_ind, temp_val, &temp_nvals, NULL, op->identity(),
            op->mul_, op->add_, A_nrows, A->nvals_,
            A_csrRowPtr, A_csrColInd, A_csrVal, 
            u->d_ind_, u->d_val_, &u->nvals_, desc );
      else if( spmspv_mode==GrB_APSPIELB )
        spmspvApspieLB(
            temp_ind, temp_val, &temp_nvals, NULL, op->identity(),
            //op->mul_, op->add_, A_nrows, A->nvals_,
            mgpu::multiplies<a>(), mgpu::plus<a>(), A_nrows, A->nvals_,
            A_csrRowPtr, A_csrColInd, A_csrVal, 
            u->d_ind_, u->d_val_, &u->nvals_, desc );
      else if( spmspv_mode==GrB_GUNROCKLB )
        spmspvGunrockLB(
            temp_ind, temp_val, &temp_nvals, NULL, op->identity(),
            op->mul_, op->add_, A_nrows, A->nvals_,
            A_csrRowPtr, A_csrColInd, A_csrVal, 
            u->d_ind_, u->d_val_, &u->nvals_, desc );
      else if( spmspv_mode==GrB_GUNROCKTWC )
        spmspvGunrockTWC(
            temp_ind, temp_val, &temp_nvals, NULL, op->identity(),
            op->mul_, op->add_, A_nrows, A->nvals_,
            A_csrRowPtr, A_csrColInd, A_csrVal, 
            u->d_ind_, u->d_val_, &u->nvals_, desc );
      CUDA( cudaDeviceSynchronize() );

      if( temp_nvals==0 )
      {
        std::cout << "No neighbours!\n";
        w->nvals_       = 0;
        w->need_update_ = true;
        return GrB_SUCCESS;
      }

      // Get descriptor parameters for nthreads
      Desc_value nt_mode;
      CHECK( desc->get(GrB_NT, &nt_mode) );
      const int nt = static_cast<int>(nt_mode);
      dim3 NT, NB;
			NT.x = nt;
			NT.y = 1;
			NT.z = 1;
			NB.x = (temp_nvals+nt-1)/nt;
			NB.y = 1;
			NB.z = 1;

      // Mask type
      // 1) Dense mask
      // 2) Sparse mask (TODO)
      // 3) Uninitialized
      Storage mask_vec_type;
      CHECK( mask->getStorage(&mask_vec_type) );
      assert( mask->dense_.nvals_ >= temp_nvals );

      if( desc->struconly() )
      {
        if( use_scmp )
          assignDenseDenseMaskedKernel<true,true,true><<<NB,NT>>>(temp_ind, 
              temp_nvals, (mask->dense_).d_val_, (M)-1.f, 
              (BinaryOp<Index,Index,Index>*)NULL, (Index)0, (Index*)NULL, 
              A_nrows);
        else
          assignDenseDenseMaskedKernel<false,true,true><<<NB,NT>>>(temp_ind, 
              temp_nvals, (mask->dense_).d_val_, (M)-1.f, 
              (BinaryOp<Index,Index,Index>*)NULL, (Index)0, (Index*)NULL, 
              A_nrows);

        if( desc->debug() )
        {
          CUDA( cudaDeviceSynchronize() );
          printDevice("mask", (mask->dense_).d_val_, A_nrows);
          printDevice("temp_ind", temp_ind, A_nrows);
        }

        // Turn dense vector into sparse
        desc->resize((3*A_nrows)*max(sizeof(Index),sizeof(T)), "buffer");
        Index* d_scan = (Index*) desc->d_buffer_+2*A_nrows;

        mgpu::Scan<mgpu::MgpuScanTypeExc>( temp_ind, A_nrows, (Index)0, 
            mgpu::plus<Index>(), (Index*)0, &w->nvals_, d_scan, 
            *(desc->d_context_) );

        if( desc->debug() )
        {
          printDevice("d_scan", d_scan, A_nrows);
        }

        streamCompactKernel<<<NB,NT>>>(w->d_ind_, temp_ind, d_scan, (W)0, 
            temp_ind, A_nrows);

        if( desc->debug() )
        {
          printDevice("w_ind", w->d_ind_, w->nvals_);
        }
      }
      else
      {
        // For visited nodes, assign 0.f to vector
        // For GrB_DENSE mask, need to add parameter for mask_identity to user
        if( mask_vec_type==GrB_DENSE )
        {
          if( use_scmp )
            assignSparseKernel<true, true, true><<<NB,NT>>>(temp_ind, temp_val, 
              temp_nvals, (mask->dense_).d_val_, (M)-1.f, 
              (BinaryOp<U,U,U>*)NULL, (U)0.f, (Index*)NULL, A_nrows);
          else
            assignSparseKernel<false,true, true><<<NB,NT>>>(temp_ind, temp_val, 
              temp_nvals, (mask->dense_).d_val_, (M)-1.f,
              (BinaryOp<U,U,U>*)NULL, (U)0.f, (Index*)NULL, A_nrows);
        }
        else if( mask_vec_type==GrB_SPARSE )
        {
          std::cout << "Spmspv Sparse Mask\n";
          std::cout << "Error: Feature not implemented yet!\n";
        }
        else
        {
          return GrB_UNINITIALIZED_OBJECT;
        }

        if( desc->debug() )
        {
          CUDA( cudaDeviceSynchronize() );
          printDevice("mask", (mask->dense_).d_val_, A_nrows);
          printDevice("temp_ind", temp_ind, temp_nvals);
          printDevice("temp_val", temp_val, temp_nvals);
        }

        // Prune 0.f's from vector
        desc->resize((4*A_nrows)*max(sizeof(Index),sizeof(T)), "buffer");
        Index* d_flag = (Index*) desc->d_buffer_+2*A_nrows;
        Index* d_scan = (Index*) desc->d_buffer_+3*A_nrows;

        updateFlagKernel<<<NB,NT>>>( d_flag, 0.f, temp_val, temp_nvals );
        mgpu::Scan<mgpu::MgpuScanTypeExc>( d_flag, temp_nvals, (Index)0, 
            mgpu::plus<Index>(), d_scan+temp_nvals, &w->nvals_, d_scan, 
            *(desc->d_context_) );

        if( desc->debug() )
        {
          printDevice("d_flag", d_flag, temp_nvals);
          printDevice("d_scan", d_scan, temp_nvals);
        }

        streamCompactKernel<<<NB,NT>>>(w->d_ind_, w->d_val_, d_scan, (W)0,
            temp_ind, temp_val, temp_nvals);

        if( desc->debug() )
        {
          printDevice("w_ind", w->d_ind_, w->nvals_);
          printDevice("w_val", w->d_val_, w->nvals_);
        }
      }
    }
    else
    {
      if( spmspv_mode==GrB_APSPIE )
        spmspvApspie(
            w->d_ind_, w->d_val_, &w->nvals_, NULL, op->identity(),
            op->mul_, op->add_, A_nrows, A->nvals_,
            A_csrRowPtr, A_csrColInd, A_csrVal, 
            u->d_ind_, u->d_val_, &u->nvals_, desc );
      else if( spmspv_mode==GrB_APSPIELB )
        spmspvApspieLB(
            w->d_ind_, w->d_val_, &w->nvals_, NULL, op->identity(),
            mgpu::multiplies<a>(), mgpu::plus<a>(), A_nrows, A->nvals_,
            A_csrRowPtr, A_csrColInd, A_csrVal, 
            u->d_ind_, u->d_val_, &u->nvals_, desc );
      else if( spmspv_mode==GrB_GUNROCKLB )
        spmspvGunrockLB(
            w->d_ind_, w->d_val_, &w->nvals_, NULL, op->identity(),
            op->mul_, op->add_, A_nrows, A->nvals_,
            A_csrRowPtr, A_csrColInd, A_csrVal, 
            u->d_ind_, u->d_val_, &u->nvals_, desc );
      else if( spmspv_mode==GrB_GUNROCKTWC )
        spmspvGunrockTWC(
            w->d_ind_, w->d_val_, &w->nvals_, NULL, op->identity(),
            op->mul_, op->add_, A_nrows, A->nvals_,
            A_csrRowPtr, A_csrColInd, A_csrVal, 
            u->d_ind_, u->d_val_, &u->nvals_, desc );
    }
    w->need_update_ = true;
    return GrB_SUCCESS;
  }

}  // backend
}  // graphblas

#endif  // GRB_BACKEND_APSPIE_SPMSPV_HPP
