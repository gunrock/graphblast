#ifndef GRB_BACKEND_CUDA_REDUCE_HPP
#define GRB_BACKEND_CUDA_REDUCE_HPP

#include <iostream>
#include <vector>

#include <cub/cub.cuh>

#include "graphblas/types.hpp"

#include "graphblas/backend/apspie/Vector.hpp"
#include "graphblas/backend/apspie/Matrix.hpp"
#include "graphblas/backend/apspie/SparseMatrix.hpp"
#include "graphblas/backend/apspie/DenseMatrix.hpp"

namespace graphblas
{
namespace backend
{
  template <typename T>
  class Matrix;

  template <typename T>
  class Vector;

  template <typename a, typename b>
  Info cubReduce( Vector<b>&             B,
                  const SparseMatrix<a>& A,
                  void*                  d_buffer,
                  size_t                 buffer_size );

  // TODO:
  // Move d_buffer into descriptor
  template <typename a, typename b>
  Info reduce( Vector<b>&       B,
               const Matrix<a>& A,
               void*            d_buffer,
               size_t           buffer_size )
  {
    Storage A_storage;
    A.getStorage( A_storage );

    Info err;
    if( A_storage == GrB_SPARSE )
    {
      err = cubReduce( B, A.sparse_, d_buffer, buffer_size );
    }
    return err;
  }

  template <typename a, typename b>
  Info cubReduce( Vector<b>&             B,
                  const SparseMatrix<a>& A,
                  void*                  d_buffer,
                  size_t                 buffer_size )
  {
    Index A_nrows;
    Index B_nvals;

    A_nrows = A.nrows_;
    B_nvals = B.nvals_;

    // Dimension compatibility check
    if( (A_nrows != B_nvals) )
    {
      std::cout << "Dim mismatch reduce" << std::endl;
      std::cout << A_nrows << " " << B_nvals << std::endl;
      return GrB_DIMENSION_MISMATCH;
    }

    /*void *d_temp_storage      = NULL;
    size_t temp_storage_bytes = 0;

    cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes,
        A.d_csrVal_, B.d_val_, A_nrows, A.d_csrRowPtr_, A.d_csrRowPtr_+1 );
    CUDA( cudaMalloc(&d_temp_storage, temp_storage_bytes) );*/
    cub::DeviceSegmentedReduce::Sum(d_buffer, buffer_size,
        A.d_csrVal_, B.d_val_, A.nrows_, A.d_csrRowPtr_, A.d_csrRowPtr_+1 );
    //CUDA( cudaFree(d_temp_storage) );   
 
    B.need_update_ = true;
    return GrB_SUCCESS;
  }

}  // backend
}  // graphblas

#endif  // GRB_BACKEND_CUDA_TRANSPOSE_HPP
