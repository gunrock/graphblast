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
             const DenseVector<M>*  mask,
             const BinaryOpT*       accum,
             const SemiringT*       op,
             const SparseMatrix<a>* A,
             const DenseVector<U>*  u,
             Descriptor*            desc )
  {
    std::cout << "Error: Feature not implemented yet!\n";
    return GrB_SUCCESS;
  }

  template <typename MulOp, typename AddOp, typename T>
  __global__ void kernel( MulOp mul_op, AddOp add_op, T* val )
  {
    *val = add_op(*val,*val);
    *val = mul_op(*val,*val);
  }

  template <typename MulOp, typename AddOp, typename T>
  void cpu_kernel( MulOp mul_op, AddOp add_op, T* val )
  {
    *val = add_op(*val,*val);
    *val = mul_op(*val,*val);
  }

typedef float (*op_func_t) (float, float);
__device__ float add_func(float x, float y) { return x + y; }
__device__ float mul_func(float x, float y) { return x * y; }
__device__ op_func_t p_add_func = add_func;
__device__ op_func_t p_mul_func = mul_func;

  template <typename MulOp, typename AddOp, typename T>
  __global__ void gpu_kernel( MulOp mul_op, AddOp add_op, T* val )
  {
    *val = (*add_op)(*val,*val);
    *val = (*mul_op)(*val,*val);
  }

  template <typename W, typename a, typename U, typename M,
            typename BinaryOpT,      typename SemiringT>
  Info spmv( DenseVector<W>*        w,
             const SparseVector<M>* mask,
             const BinaryOpT*       accum,
             const SemiringT*       op,
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

    // Transpose:
    const Index* A_csrRowPtr = (use_tran) ? A->d_cscColPtr_ : A->d_csrRowPtr_;
    const Index* A_csrColInd = (use_tran) ? A->d_cscRowInd_ : A->d_csrColInd_;
    const T*     A_csrVal    = (use_tran) ? A->d_cscVal_    : A->d_csrVal_;
    const Index  A_nrows     = (use_tran) ? A->ncols_       : A->nrows_;

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
      dim3 NT, NB;
      NT.x = nt;
      NT.y = 1;
      NT.z = 1;
      NB.x = (ta*A_nrows+nt-1)/nt;
      NB.y = 1;
      NB.z = 1;
      spmvMaskedOrKernel<true,false,false><<<NB,NT>>>( 
          w->d_val_, mask->d_val_, NULL, 
          op->mul_, op->add_, A_nrows, A->nvals_, 
          A_csrRowPtr, A_csrColInd, A_csrVal, u->d_val_ );
    }
    else if( !use_mask )
    {
      a* d_val;
      a  h_val = 1.f;
      CUDA( cudaMalloc( &d_val, sizeof(a) ) );
      CUDA( cudaMemcpy( d_val, &h_val, sizeof(a), cudaMemcpyHostToDevice ) );

      op_func_t h_add_func;
      op_func_t h_mul_func;
      cudaMemcpyFromSymbol( &h_add_func, p_add_func, sizeof( op_func_t ) );
      cudaMemcpyFromSymbol( &h_mul_func, p_mul_func, sizeof( op_func_t ) );
      op_func_t d_add_func = h_add_func;
      op_func_t d_mul_func = h_mul_func;

      //cpu_kernel( graphblas::multiplies<a>(), graphblas::plus<a>(), &h_val);
      //cpu_kernel( op->mul_, op->add_, &h_val );
      //kernel<<<1,1>>>( op->mul_, op->add_, d_val );
      gpu_kernel<<<1,1>>>( d_mul_func, d_add_func, d_val );
      //kernel<<<1,1>>>( graphblas::multiplies<a>(), graphblas::plus<a>(), d_val );
      CUDA( cudaMemcpy( &h_val, d_val, sizeof(a), cudaMemcpyDeviceToHost ) );
      std::cout << h_val << std::endl;
      std::cout << &op << " " << &u << " " << &w << std::endl;
      std::cout << &(op->mul_) << " " << &(op->add_) << std::endl;
      std::cout << &(u->d_val_) << " " << &(w->d_val_) << std::endl;
      std::cout << &h_add_func << " " << &p_add_func << std::endl;
      /*mgpu::SpmvCsrBinary( A_csrVal, A_csrColInd, A->nvals_, 
          A_csrRowPtr, A_nrows, u->d_val_, true, w->d_val_, 
          op->identity(), mgpu::multiplies<a>(), mgpu::plus<a>(), 
          *(desc->d_context_) );*/
      /*mgpu::SpmvCsrBinary( A_csrVal, A_csrColInd, A->nvals_, 
          A_csrRowPtr, A_nrows, u->d_val_, true, w->d_val_, 
          op->identity(), op->mul_, op->add_, *(desc->d_context_) );*/

      /*size_t temp_storage_bytes = 0;
			cub::DeviceSpmv::CsrMV(desc->d_buffer_, temp_storage_bytes, A->d_csrVal_,
					A->d_csrRowPtr_, A->d_csrColInd_, u->d_val_, w->d_val_,
					A->nrows_, A->ncols_, A->nvals_, 1.f, 0.f);
      desc->resize( temp_storage_bytes );
			cub::DeviceSpmv::CsrMV(desc->d_buffer_, desc->d_size_, A->d_csrVal_,
					A->d_csrRowPtr_, A->d_csrColInd_, u->d_val_, w->d_val_,
					A->nrows_, A->ncols_, A->nvals_, 1.f, 0.f);*/
    }
    w->need_update_ = true;
    return GrB_SUCCESS;
  }

}  // backend
}  // graphblas

#endif  // GRB_BACKEND_APSPIE_SPMV_HPP
