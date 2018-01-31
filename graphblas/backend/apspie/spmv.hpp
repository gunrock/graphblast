#ifndef GRB_BACKEND_APSPIE_SPMV_HPP
#define GRB_BACKEND_APSPIE_SPMV_HPP

#include <iostream>

#include <cuda.h>
#include <cusparse.h>

#include <moderngpu.cuh>
#include <cub.cuh>

#include "graphblas/backend/apspie/Descriptor.hpp"
#include "graphblas/backend/apspie/SparseMatrix.hpp"
#include "graphblas/backend/apspie/DenseMatrix.hpp"
#include "graphblas/backend/apspie/kernels/spmv.hpp"

namespace graphblas
{
namespace backend
{

  template <typename W, typename a, typename U, typename M,
            typename BinaryOpT,      typename SemiringT>
  Info spmv( DenseVector<W>*        w,
             const Vector<M>*       mask,
             BinaryOpT              accum,
             SemiringT              op,
             const SparseMatrix<a>* A,
             const DenseVector<U>*  u,
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

    if( desc->debug() )
    {
      std::cout << "Executing Spmv\n";
      printState( use_mask, use_accum, use_scmp, use_repl, use_tran );
    }

    // Transpose (default is CSR):
    const Index* A_csrRowPtr = (use_tran) ? A->d_cscColPtr_ : A->d_csrRowPtr_;
    const Index* A_csrColInd = (use_tran) ? A->d_cscRowInd_ : A->d_csrColInd_;
    const T*     A_csrVal    = (use_tran) ? A->d_cscVal_    : A->d_csrVal_;
    const Index  A_nrows     = (use_tran) ? A->ncols_       : A->nrows_;

    if( desc->debug() )
    {
      std::cout << "cscColPtr: " << A->d_cscColPtr_ << std::endl;
      std::cout << "cscRowInd: " << A->d_cscRowInd_ << std::endl;
      std::cout << "cscVal:    " << A->d_cscVal_    << std::endl;

      std::cout << "csrRowPtr: " << A->d_csrRowPtr_ << std::endl;
      std::cout << "csrColInd: " << A->d_csrColInd_ << std::endl;
      std::cout << "csrVal:    " << A->d_csrVal_    << std::endl;
    }

    // Get descriptor parameters for nthreads
    Desc_value ta_mode, tb_mode, nt_mode;
    CHECK( desc->get(GrB_TA, &ta_mode) );
    CHECK( desc->get(GrB_TB, &tb_mode) );
    CHECK( desc->get(GrB_NT, &nt_mode) );

    const int ta = static_cast<int>(ta_mode);
    const int tb = static_cast<int>(tb_mode);
    const int nt = static_cast<int>(nt_mode);

    if( use_mask )
    {
      // TODO: add if condition here for if( add_ == GrB_LOR )
      if(true)
      {
				// Mask type
				// 1) Dense mask
				// 2) Sparse mask (TODO)
				// 3) Uninitialized
				Storage mask_vec_type;
				CHECK( mask->getStorage(&mask_vec_type) );

				if( mask_vec_type==GrB_DENSE )
				{
					dim3 NT, NB;
					NT.x = nt;
					NT.y = 1;
					NT.z = 1;
					NB.x = (A_nrows+nt-1)/nt;
					NB.y = 1;
					NB.z = 1;

          int variant = 0;
          variant |= (use_scmp         ) ? 4 : 0;
          variant |= (desc->earlyexit()) ? 2 : 0;
          variant |= (desc->opreuse()  ) ? 1 : 0;

          if( desc->earlyexitbench() )
          {
            int* d_stats;
            CUDA( cudaMalloc(&d_stats, A_nrows*sizeof(int)) );
            CUDA( cudaMemset(d_stats, 0, A_nrows*sizeof(int)) );

            switch( variant )
            {
              case 0:
                spmvDenseMaskedOrKernelBench<false,false,false><<<NB,NT>>>( 
                    w->d_val_, d_stats, mask->dense_.d_val_, (M)-1.f, NULL, 0.f,
                    op.mul_op(), op.add_op(), A_nrows, A->nvals_, 
                    A_csrRowPtr, A_csrColInd, A_csrVal, u->d_val_ );
                break;
              case 1:
                spmvDenseMaskedOrKernelBench<false,false,true><<<NB,NT>>>(
                    w->d_val_, d_stats, mask->dense_.d_val_, (M)-1.f, NULL, 0.f,
                    op.mul_op(), op.add_op(), A_nrows, A->nvals_,
                    A_csrRowPtr, A_csrColInd, A_csrVal, u->d_val_ );
                break;
              case 2:
                spmvDenseMaskedOrKernelBench<false, true,false><<<NB,NT>>>(
                    w->d_val_, d_stats, mask->dense_.d_val_, (M)-1.f, NULL, 0.f,
                    op.mul_op(), op.add_op(), A_nrows, A->nvals_,
                    A_csrRowPtr, A_csrColInd, A_csrVal, u->d_val_ );
                break;
              case 3:
                spmvDenseMaskedOrKernelBench<false, true, true><<<NB,NT>>>(
                    w->d_val_, d_stats, mask->dense_.d_val_, (M)-1.f, NULL, 0.f,
                    op.mul_op(), op.add_op(), A_nrows, A->nvals_,
                    A_csrRowPtr, A_csrColInd, A_csrVal, u->d_val_ );
                break;
              case 4:
                spmvDenseMaskedOrKernelBench< true,false,false><<<NB,NT>>>( 
                    w->d_val_, d_stats, mask->dense_.d_val_, (M)-1.f, NULL, 0.f,
                    op.mul_op(), op.add_op(), A_nrows, A->nvals_, 
                    A_csrRowPtr, A_csrColInd, A_csrVal, u->d_val_ );
                break;
              case 5:
                spmvDenseMaskedOrKernelBench< true,false, true><<<NB,NT>>>(
                    w->d_val_, d_stats, mask->dense_.d_val_, (M)-1.f, NULL, 0.f,
                    op.mul_op(), op.add_op(), A_nrows, A->nvals_,
                    A_csrRowPtr, A_csrColInd, A_csrVal, u->d_val_ );
                break;
              case 6:
                spmvDenseMaskedOrKernelBench<true, true,false><<<NB,NT>>>(
                    w->d_val_, d_stats, mask->dense_.d_val_, (M)-1.f, NULL, 0.f,
                    op.mul_op(), op.add_op(), A_nrows, A->nvals_,
                    A_csrRowPtr, A_csrColInd, A_csrVal, u->d_val_ );
                break;
              case 7:
                spmvDenseMaskedOrKernelBench<true, true, true><<<NB,NT>>>(
                    w->d_val_, d_stats, mask->dense_.d_val_, (M)-1.f, NULL, 0.f,
                    op.mul_op(), op.add_op(), A_nrows, A->nvals_,
                    A_csrRowPtr, A_csrColInd, A_csrVal, u->d_val_ );
                break;
              default:
                break;
            }

            int* h_stats = (int*)malloc(A_nrows*sizeof(int));
            CUDA( cudaMemcpy(h_stats, d_stats, A_nrows*sizeof(int),
                cudaMemcpyDeviceToHost) );

            // Total:
            int my_total;
            mgpu::Reduce( u->d_val_, A_nrows, (int)0, mgpu::plus<Index>(),
                (Index*)0, &my_total, *(desc->d_context_) );

            // Min:
            int my_min;
            mgpu::Reduce( d_stats, A_nrows, INT_MAX, mgpu::minimum<int>(), 
                (int*)0, &my_min, *(desc->d_context_) );

            // Max:
            int my_max;
            mgpu::Reduce( d_stats, A_nrows, INT_MIN, mgpu::maximum<int>(), 
                (int*)0, &my_max, *(desc->d_context_) );
            
            // Sum:
            int my_sum;
            mgpu::Reduce( d_stats, A_nrows, (int)0, mgpu::plus<int>(), 
                (int*)0, &my_sum, *(desc->d_context_) );
            
            double my_mean = (double)my_sum/my_total;

            // Stddev:
            double my_var  = 0.;
            for( int i=0; i<A_nrows; i++ )
            {
              if( desc->debug() )
                std::cout << i << " " << h_stats[i] << std::endl;
              double delta = (double)h_stats[i]-my_mean;
              my_var      += (delta*delta);
            }

            printf("%d, %lf, %d, %d\n", my_sum, my_var, my_min, my_max);
            /*std::cout << "Total of " << my_total << " nnz\n";
            std::cout << "Min: "     << my_min   << std::endl;
            std::cout << "Max: "     << my_max   << std::endl;
            std::cout << "Sum: "     << my_sum   << std::endl;
            std::cout << "Var: "     << my_var   << std::endl;*/
            CUDA( cudaFree(d_stats) );
            free( h_stats );
          }
          else
          {
            switch( variant )
            {
              case 0:
                spmvDenseMaskedOrKernel<false,false,false><<<NB,NT>>>( 
                    w->d_val_, mask->dense_.d_val_, NULL, op.identity(),
                    op.mul_op(), op.add_op(), A_nrows, A->nvals_, 
                    A_csrRowPtr, A_csrColInd, A_csrVal, u->d_val_ );
                break;
              case 1:
                spmvDenseMaskedOrKernel<false,false,true><<<NB,NT>>>(
                    w->d_val_, mask->dense_.d_val_, NULL, op.identity(),
                    op.mul_op(), op.add_op(), A_nrows, A->nvals_,
                    A_csrRowPtr, A_csrColInd, A_csrVal, u->d_val_ );
                break;
              case 2:
                spmvDenseMaskedOrKernel<false, true,false><<<NB,NT>>>(
                    w->d_val_, mask->dense_.d_val_, NULL, op.identity(),
                    op.mul_op(), op.add_op(), A_nrows, A->nvals_,
                    A_csrRowPtr, A_csrColInd, A_csrVal, u->d_val_ );
                break;
              case 3:
                spmvDenseMaskedOrKernel<false, true, true><<<NB,NT>>>(
                    w->d_val_, mask->dense_.d_val_, NULL, op.identity(),
                    op.mul_op(), op.add_op(), A_nrows, A->nvals_,
                    A_csrRowPtr, A_csrColInd, A_csrVal, u->d_val_ );
                break;
              case 4:
                spmvDenseMaskedOrKernel< true,false,false><<<NB,NT>>>( 
                    w->d_val_, mask->dense_.d_val_, NULL, op.identity(),
                    op.mul_op(), op.add_op(), A_nrows, A->nvals_, 
                    A_csrRowPtr, A_csrColInd, A_csrVal, u->d_val_ );
                break;
              case 5:
                spmvDenseMaskedOrKernel< true,false, true><<<NB,NT>>>(
                    w->d_val_, mask->dense_.d_val_, NULL, op.identity(),
                    op.mul_op(), op.add_op(), A_nrows, A->nvals_,
                    A_csrRowPtr, A_csrColInd, A_csrVal, u->d_val_ );
                break;
              case 6:
                spmvDenseMaskedOrKernel<true, true,false><<<NB,NT>>>(
                    w->d_val_, mask->dense_.d_val_, NULL, op.identity(),
                    op.mul_op(), op.add_op(), A_nrows, A->nvals_,
                    A_csrRowPtr, A_csrColInd, A_csrVal, u->d_val_ );
                break;
              case 7:
                spmvDenseMaskedOrKernel<true, true, true><<<NB,NT>>>(
                    w->d_val_, mask->dense_.d_val_, NULL, op.identity(),
                    op.mul_op(), op.add_op(), A_nrows, A->nvals_,
                    A_csrRowPtr, A_csrColInd, A_csrVal, u->d_val_ );
                break;
              default:
                break;
            }
            if( desc->debug() )
              printDevice("w_val", w->d_val_, A_nrows);
          }
				}
				else if( mask_vec_type==GrB_SPARSE )
				{
          std::cout << "DeVec Sparse Mask Spmv\n";
					std::cout << "Error: Feature not implemented yet!\n";
				}
				else
				{
					return GrB_UNINITIALIZED_OBJECT;
				}
      }
      // TODO: add else condition here for generic mask semiring
      else
      {
        std::cout << "Indirect Spmv\n";
			  std::cout << "Error: Feature not implemented yet!\n";
        /*mgpu::SpmvCsrIndirectBinary( A_csrVal, A_csrColInd, A->nvals_,
            A_csrRowPtr, mask->dense_.d_ind_, A_nrows, u->d_val_, true, 
            w->d_val_, 0.f, mgpu::multiplies<a>(), mgpu::plus<a>(),
            *(desc->d_context_) );*/
      }
    }
    else
    {
      mgpu::SpmvCsrBinary( A_csrVal, A_csrColInd, A->nvals_, A_csrRowPtr, 
          A_nrows, u->d_val_, true, w->d_val_, op.identity(), op.mul_op(), 
          op.add_op(), *(desc->d_context_) );

      // TODO: add semiring inputs to CUB
      /*size_t temp_storage_bytes = 0;
			cub::DeviceSpmv::CsrMV(desc->d_temp_, temp_storage_bytes, A->d_csrVal_,
					A->d_csrRowPtr_, A->d_csrColInd_, u->d_val_, w->d_val_,
					A->nrows_, A->ncols_, A->nvals_, 1.f, op->identity());
      desc->resize( temp_storage_bytes, "temp" );
			cub::DeviceSpmv::CsrMV(desc->d_temp_, desc->d_temp_size_, A->d_csrVal_,
					A->d_csrRowPtr_, A->d_csrColInd_, u->d_val_, w->d_val_,
					A->nrows_, A->ncols_, A->nvals_, 1.f, op->identity());*/
    }
    w->need_update_ = true;
    return GrB_SUCCESS;
  }

}  // backend
}  // graphblas

#endif  // GRB_BACKEND_APSPIE_SPMV_HPP
