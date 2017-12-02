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
    bool useMask = (mask==NULL ) ? false : true;
    bool useAccum= (accum==NULL) ? false : true;
    //bool useScmp = (
    //bool useRepl = (
    //bool useTran = (desc->
    if( useMask )
    {
      
    }
    else if( !useMask )
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
      /*mgpu::SpmvCsrBinary( A->d_csrVal_, A->d_csrColInd_, A->nvals_, 
          A->d_csrRowPtr_, A->nrows_, u->d_val_, true, w->d_val_, 
          op->identity(), mgpu::multiplies<a>(), mgpu::plus<a>(), 
          *(desc->d_context_) );*/
      /*mgpu::SpmvCsrBinary( A->d_csrVal_, A->d_csrColInd_, A->nvals_, 
          A->d_csrRowPtr_, A->nrows_, u->d_val_, true, w->d_val_, 
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

#endif  // GRB_BACKEND_APSPIE_SPMM_HPP
