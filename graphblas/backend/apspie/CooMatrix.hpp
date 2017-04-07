#ifndef GRB_BACKEND_APSPIE_COOMATRIX_HPP
#define GRB_BACKEND_APSPIE_COOMATRIX_HPP

#include <vector>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "graphblas/backend/apspie/apspie.hpp"

namespace graphblas
{
namespace backend
{
  template <typename T>
  class CooMatrix
  {
    public:
    CooMatrix() : nrows_(0), ncols_(0) {}
	  CooMatrix( const Index nrows, const Index ncols );

	  // C API Methods
	  Info build( const std::vector<Index>& row_indices,
		        		const std::vector<Index>& col_indices,
				        const std::vector<T>& values,
				        const Index nvals,
				        const CooMatrix& mask,
				        const BinaryOp& dup );

	  Info build( const std::vector<Index>& row_indices,
		        		const std::vector<Index>& col_indices,
				        const std::vector<T>& values,
				        const Index nvals );

    Info print();

    private:
    Index nrows_;
    Index ncols_;
    Index nvals_;

    Index* h_csrColInd;
    Index* h_csrRowPtr;
    T*     h_csrVal;

    Index* d_csrColInd;
		Index* d_csrRowPtr;
		T*     d_csrVal;
  };

  template <typename T>	
  CooMatrix<T>::CooMatrix( const Index nrows, const Index ncols )
       : nrows_(nrows), ncols_(ncols) 
  {
    // Host alloc
    h_csrRowPtr = (Index*)malloc((nrows+1)*sizeof(Index));

    // RowInd and Val will be allocated in build rather than here
    // since nvals may be unknown
    h_csrColInd = NULL;
    h_csrVal = NULL;
    d_csrColInd = NULL;
    d_csrVal = NULL;

    // Device alloc
    CUDA_SAFE_CALL(cudaMalloc(&d_csrRowPtr, (nrows+1)*sizeof(Index)));
  }

  template <typename T>
  Info CooMatrix<T>::build( const std::vector<Index>& row_indices,
                            const std::vector<Index>& col_indices,
                            const std::vector<T>& values,
                            const Index nvals,
                            const CooMatrix& mask,
                            const BinaryOp& dup) {}

  template <typename T>
  Info CooMatrix<T>::build( const std::vector<Index>& row_indices,
                            const std::vector<Index>& col_indices,
                            const std::vector<T>& values,
                            const Index nvals )
	{
    nvals_ = nvals;

    // Host malloc
    h_csrColInd = (Index*)malloc(nvals*sizeof(Index));
    h_csrVal    = (T*)    malloc(nvals*sizeof(T));

    // Device malloc
    CUDA_SAFE_CALL(cudaMalloc(&d_csrColInd, nvals*sizeof(Index)));
    CUDA_SAFE_CALL(cudaMalloc(&d_csrVal,    nvals*sizeof(T))); 

    // Convert to CSR/CSC
    Index temp, row, dest, cumsum=0;

    // Set all rowPtr to 0
    for( Index i=0; i<=nrows_; i++ )
      h_csrRowPtr[i] = 0;
    // Go through all elements to see how many fall in each row
    for( Index i=0; i<nvals; i++ )
      h_csrRowPtr[ col_indices[i] ]++;
    // Cumulative sum to obtain rowPtr
    for( Index i=0; i<nrows_; i++ ) {
      temp = h_csrRowPtr[i];
      h_csrRowPtr[i] = cumsum;
      cumsum += temp;
    }
    h_csrRowPtr[nrows_] = nvals;


    // Store colInd and val
    for( Index i=0; i<nvals; i++ ) {
      row = row_indices[i];
      dest= h_csrRowPtr[row];
      h_csrColInd[dest] = col_indices[i];
      h_csrVal[dest]    = values[i];
      h_csrRowPtr[row]++;
    }
    cumsum = 0;

    // Undo damage done to rowPtr
    for( Index i=0; i<=nrows_; i++ ) {
      temp = h_csrRowPtr[i];
      h_csrRowPtr[i] = cumsum;
      cumsum = temp;
    }

    // Device memcpy
    CUDA_SAFE_CALL(cudaMemcpy(d_csrVal,    h_csrVal,    nvals*sizeof(T),
        cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_csrColInd, h_csrColInd, nvals*sizeof(Index),
        cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_csrRowPtr, h_csrRowPtr, (nrows_+1)*sizeof(Index),
        cudaMemcpyHostToDevice));
	}

  template <typename T>
  Info CooMatrix<T>::print()
	{
    printArray( "csrColInd", h_csrColInd );
		printArray( "csrRowPtr", h_csrRowPtr );
		printArray( "csrVal",    h_csrVal );
	}

} // backend
} // graphblas

#endif  // GRB_BACKEND_APSPIE_COOMATRIX_HPP
