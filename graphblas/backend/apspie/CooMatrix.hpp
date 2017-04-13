#ifndef GRB_BACKEND_APSPIE_COOMATRIX_HPP
#define GRB_BACKEND_APSPIE_COOMATRIX_HPP

#include <vector>
#include <iostream>
#include <typeinfo>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "graphblas/backend/apspie/apspie.hpp"

namespace graphblas
{
namespace backend
{
  template <typename T, typename S>
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

    Info build( const std::vector<T>& values );

    Info print();

    private:
    Index nrows_;
    Index ncols_;
    Index nvals_;

		// CSR format
    Index* h_csrColInd;
    Index* h_csrRowPtr;
    T*     h_csrVal;
    Index* d_csrColInd;
		Index* d_csrRowPtr;
		T*     d_csrVal;

    // CSC format
		// TODO: add CSC support. 
		// -this will be useful and necessary for direction-optimized SpMV
		/*Index* h_cscRowInd;
		Index* h_cscColPtr;
    T*     h_cscVal;
		Index* d_cscRowInd;
		Index* d_cscColPtr;
		T*     d_csrVal;*/

		// Dense format
		T* h_denseVal;
		T* d_denseVal;
  };

  template <typename T, typename S>	
  CooMatrix<T,S>::CooMatrix( const Index nrows, const Index ncols )
       : nrows_(nrows), ncols_(ncols) 
  {
    if( typeid(S)==typeid(Sparse) ) {
	    // Host alloc
      h_csrRowPtr = (Index*)malloc((nrows+1)*sizeof(Index));

      // Device alloc
      CUDA_SAFE_CALL(cudaMalloc(&d_csrRowPtr, (nrows+1)*sizeof(Index)));

      // RowInd and Val will be allocated in build rather than here
      // since nvals may be unknown
      h_csrColInd = NULL;
      h_csrVal = NULL;
      d_csrColInd = NULL;
      d_csrVal = NULL;

		  // Ignore dense matrices
		  h_denseVal = NULL;
		  d_denseVal = NULL;
		} else {
      // Host alloc
		  h_denseVal = (T*)malloc(nrows_*ncols_*sizeof(T));

      // Device alloc
      CUDA_SAFE_CALL(cudaMalloc(&d_denseVal, nrows_*ncols_*sizeof(T)));

      // Ignore sparse matrices
      h_csrRowPtr = NULL;
      h_csrColInd = NULL;
      h_csrVal = NULL;
      d_csrRowPtr = NULL;
      d_csrColInd = NULL;
      d_csrVal = NULL;
  }}

  template <typename T, typename S>
  Info CooMatrix<T,S>::build( const std::vector<Index>& row_indices,
                            const std::vector<Index>& col_indices,
                            const std::vector<T>& values,
                            const Index nvals,
                            const CooMatrix& mask,
                            const BinaryOp& dup) {}

  template <typename T, typename S>
  Info CooMatrix<T,S>::build( const std::vector<Index>& row_indices,
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
      h_csrRowPtr[ row_indices[i] ]++;
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

	template <typename T, typename S>
  Info CooMatrix<T,S>::build( const std::vector<T>& values )
	{
    // Host alloc
		h_denseVal = (T*)malloc(nrows_*ncols_*sizeof(T));

    // Device alloc
    CUDA_SAFE_CALL(cudaMalloc(&d_denseVal, nrows_*ncols_*sizeof(T)));
	}

  template <typename T, typename S>
  Info CooMatrix<T,S>::print()
	{
    printArray( "csrColInd", h_csrColInd );
		printArray( "csrRowPtr", h_csrRowPtr );
		printArray( "csrVal",    h_csrVal );
	}

} // backend
} // graphblas

#endif  // GRB_BACKEND_APSPIE_COOMATRIX_HPP
