#ifndef GRB_BACKEND_APSPIE_TRACE_HPP
#define GRB_BACKEND_APSPIE_TRACE_HPP

#include <iostream>

#include "graphblas/backend/apspie/kernels/kernels.hpp"

namespace graphblas
{
namespace backend
{
  // Sparse-sparse variant
  template <typename T, typename a, typename b,
            typename SemiringT>
  Info traceMxmTransposeInner( T*                     val,
                               SemiringT              op,
                               const SparseMatrix<a>* A,
                               const SparseMatrix<b>* B,
                               Descriptor*            desc )
  {
    // Assume A and B are square
    const Index A_nrows = A->nrows_;

    // Get desired number of threads per block
    Desc_value nt_mode;
    CHECK( desc->get(GrB_NT, &nt_mode) );
    const int nt = static_cast<int>(nt_mode);

    int num_warps = nt/32;
    dim3 NT, NB;
    NT.x = nt;
    NT.y = 1;
    NT.z = 1;
    NB.x = (A_nrows + num_warps - 1) / num_warps;
    NB.y = 1;
    NB.z = 1;

    // Initialize val to 0
    CHECK( desc->resize(sizeof(T), "buffer") );
    T* d_val = (T*) desc->d_buffer_;
    *val = 0.f;
    CUDA_CALL( cudaMemcpy(d_val, val, sizeof(T), cudaMemcpyHostToDevice) );

    traceKernel<<<NB, NT>>>(d_val, op.identity(), extractMul(op), 
        extractAdd(op),  A_nrows, 
        A->d_csrRowPtr_, A->d_csrColInd_, A->d_csrVal_, 
        B->d_csrRowPtr_, B->d_csrColInd_, B->d_csrVal_);

    CUDA_CALL( cudaMemcpy(val, d_val, sizeof(T), cudaMemcpyDeviceToHost) );

    return GrB_SUCCESS;
  }

}  // backend
}  // graphblas

#endif  // GRB_BACKEND_APSPIE_TRACE_HPP
