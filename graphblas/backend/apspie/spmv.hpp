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
    if( !useMask )
    {
      mgpu::GrbSpmvCsrBinary( A->d_csrVal_, A->d_csrColInd_, A->nvals_, 
          A->d_csrRowPtr_, A->nrows_, u->d_val_, true, w->d_val_, 
          op->identity(), op->mul_, op->add_, *(desc->d_context_) );

      /*size_t temp_storage_bytes = 0;
			cub::DeviceSpmv::CsrMV(desc->d_buffer_, temp_storage_bytes, A->d_csrVal_,
					A->d_csrRowPtr_, A->d_csrColInd_, u->d_val_, w->d_val_,
					A->nrows_, A->ncols_, A->nvals_, 1.f, 0.f);
      desc->resize( temp_storage_bytes );
			cub::DeviceSpmv::CsrMV(desc->d_buffer_, desc->d_size_, A->d_csrVal_,
					A->d_csrRowPtr_, A->d_csrColInd_, u->d_val_, w->d_val_,
					A->nrows_, A->ncols_, A->nvals_, 1.f, 0.f);*/
    }
    return GrB_SUCCESS;
  }

}  // backend
}  // graphblas

#endif  // GRB_BACKEND_APSPIE_SPMM_HPP
