#ifndef GRB_BACKEND_APSPIE_SPARSEMATRIX_HPP
#define GRB_BACKEND_APSPIE_SPARSEMATRIX_HPP

#include <vector>
#include <iostream>
#include <typeinfo>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "graphblas/backend/apspie/Matrix.hpp"
#include "graphblas/backend/apspie/apspie.hpp"
#include "graphblas/backend/apspie/util.hpp"

namespace graphblas
{
namespace backend
{
  template <typename T>
  class SparseMatrix
  {
    public:
    SparseMatrix() : nrows_(0), ncols_(0) {}
    SparseMatrix( const Index nrows, const Index ncols )
			  : nrows_(nrows), ncols_(ncols) {}

	  // C API Methods
	  Info build( const std::vector<Index>& row_indices,
		        		const std::vector<Index>& col_indices,
				        const std::vector<T>& values,
				        const Index nvals,
				        const SparseMatrix& mask,
				        const BinaryOp& dup );

	  Info build( const std::vector<Index>& row_indices,
		        		const std::vector<Index>& col_indices,
				        const std::vector<T>& values,
				        const Index nvals );

    Info print() const; // Const, because host memory unmodified

    Info nnew( const Index nrows, const Index ncols ); // possibly unnecessary in C++
    Info dup( const SparseMatrix& C ) {}
    Info clear() {} 
		Info nrows( Index& nrows ) const;
		Info ncols( Index& ncols ) const;
		Info nvals( Index& nvals ) const;

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

		// Keep track of whether host values are up-to-date with device values
		bool need_update;
  };

  template <typename T>
  Info SparseMatrix<T>::build( const std::vector<Index>& row_indices,
                               const std::vector<Index>& col_indices,
                               const std::vector<T>& values,
                               const Index nvals,
                               const SparseMatrix& mask,
                               const BinaryOp& dup) {}

  template <typename T>
  Info SparseMatrix<T>::build( const std::vector<Index>& row_indices,
                               const std::vector<Index>& col_indices,
                               const std::vector<T>& values,
                               const Index nvals )
	{
    nvals_ = nvals;
		need_update = false;

    // Host malloc
    h_csrRowPtr = (Index*)malloc((nrows_+1)*sizeof(Index));
    h_csrColInd = (Index*)malloc(nvals_*sizeof(Index));
    h_csrVal    = (T*)    malloc(nvals_*sizeof(T));

    // Device malloc
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_csrRowPtr, (nrows_+1)*sizeof(Index)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_csrColInd, nvals_*sizeof(Index)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_csrVal,    nvals_*sizeof(T))); 

    // Convert to CSR/CSC
    Index temp, row, dest, cumsum=0;

    // Set all rowPtr to 0
    for( Index i=0; i<=nrows_; i++ )
      h_csrRowPtr[i] = 0;
    // Go through all elements to see how many fall in each row
    for( Index i=0; i<nvals_; i++ )
      h_csrRowPtr[ row_indices[i] ]++;
    // Cumulative sum to obtain rowPtr
    for( Index i=0; i<nrows_; i++ ) {
      temp = h_csrRowPtr[i];
      h_csrRowPtr[i] = cumsum;
      cumsum += temp;
    }
    h_csrRowPtr[nrows_] = nvals;

    // Store colInd and val
    for( Index i=0; i<nvals_; i++ ) {
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
    CUDA_SAFE_CALL(cudaMemcpy(d_csrVal,    h_csrVal,    nvals_*sizeof(T),
        cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_csrColInd, h_csrColInd, nvals_*sizeof(Index),
        cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_csrRowPtr, h_csrRowPtr, 
				(nrows_+1)*sizeof(Index), cudaMemcpyHostToDevice));

		return GrB_SUCCESS;
	}

  template <typename T>
  Info SparseMatrix<T>::print() const
	{
    // Device memcpy
		if( need_update ) {
      CUDA_SAFE_CALL(cudaMemcpy(h_csrVal,    d_csrVal,    
			  	nvals_*sizeof(T), cudaMemcpyDeviceToHost));
      CUDA_SAFE_CALL(cudaMemcpy(h_csrColInd, d_csrColInd, 
					nvals_*sizeof(Index), cudaMemcpyDeviceToHost));
      CUDA_SAFE_CALL(cudaMemcpy(h_csrRowPtr, d_csrRowPtr, 
			    (nrows_+1)*sizeof(Index), cudaMemcpyDeviceToHost));
		}
    printArray( "csrColInd", h_csrColInd );
		printArray( "csrRowPtr", h_csrRowPtr );
		printArray( "csrVal",    h_csrVal );
		return GrB_SUCCESS;
	}

	template <typename T>
	Info SparseMatrix<T>::nnew( const Index nrows, const Index ncols )
	{
		nrows_ = nrows;
		ncols_ = ncols;
		return GrB_SUCCESS;
	}

	template <typename T>
	Info SparseMatrix<T>::nrows( Index& nrows ) const
	{
		nrows = nrows_;
		return GrB_SUCCESS;
	}

	template <typename T>
	Info SparseMatrix<T>::ncols( Index& ncols ) const
	{
		ncols = ncols_;
		return GrB_SUCCESS;
	}

	template <typename T>
	Info SparseMatrix<T>::nvals( Index& nvals ) const
	{
		nvals = nvals_;
		return GrB_SUCCESS;
	}
} // backend
} // graphblas

#endif  // GRB_BACKEND_APSPIE_COOMATRIX_HPP
