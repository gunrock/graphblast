#ifndef GRB_BACKEND_APSPIE_DENSEMATRIX_HPP
#define GRB_BACKEND_APSPIE_DENSEMATRIX_HPP

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
  class DenseMatrix
  {
    public:
    DenseMatrix() : nrows_(0), ncols_(0) {}
    DenseMatrix( const Index nrows, const Index ncols ) 
				: nrows_(nrows), ncols_(ncols) {}

	  // C API Methods
    Info build( const std::vector<T>& values );
    
    Info print() const; // Const, because host memory unmodified

    Info nnew( const Index nrows, const Index ncols ); // possibly unnecessary in C++
    Info dup( const DenseMatrix& C ) {}
    Info clear() {}
    Info nrows( Index& nrows ) const;
    Info ncols( Index& ncols ) const;
    Info nvals( Index& nvals ) const;

    private:
    Index nrows_;
    Index ncols_;
    Index nvals_;

		// Dense format
		T* h_denseVal;
		T* d_denseVal;

    // Keep track of whether host values are up-to-date with device values 
		bool need_update;
  };

	template <typename T>
  Info DenseMatrix<T>::build( const std::vector<T>& values )
	{
    nvals_ = nrows_*ncols_;
    need_update = false;

	  // Host alloc
		h_denseVal = (T*)malloc(nrows_*ncols_*sizeof(T));

    // Device alloc
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_denseVal, nrows_*ncols_*sizeof(T)));
    CUDA_SAFE_CALL(cudaMemset( d_denseVal, (T) 0, nrows_*ncols_*sizeof(T)));

		// Host copy
		for( graphblas::Index i=0; i<nrows_*ncols_; i++ )
				h_denseVal[i] = values[i];

    // Device memcpy
    CUDA_SAFE_CALL(cudaMemcpy(d_denseVal, h_denseVal, nrows_*ncols_*sizeof(T),
				cudaMemcpyHostToDevice));
		//CUDA_SAFE_CALL(cudaDeviceSynchronize());

		//printArrayDevice( "B matrix GPU", d_denseVal );
		return GrB_SUCCESS;
	}

  template <typename T>
  Info DenseMatrix<T>::print() const
	{
		if( need_update ) {
		  CUDA_SAFE_CALL(cudaMemcpy(h_denseVal, d_denseVal, 
				  nvals_*sizeof(T), cudaMemcpyDeviceToHost));
		}

    printArray( "denseVal", h_denseVal );
		return GrB_SUCCESS;
	}

	template <typename T>
	Info DenseMatrix<T>::nnew( const Index nrows, const Index ncols )
	{
		nrows_ = nrows;
		ncols_ = ncols;
		return GrB_SUCCESS;
	}

	template <typename T>
	Info DenseMatrix<T>::nrows( Index& nrows ) const
	{
		nrows = nrows_;
		return GrB_SUCCESS;
	}

	template <typename T>
	Info DenseMatrix<T>::ncols( Index& ncols ) const
	{
		ncols = ncols_;
		return GrB_SUCCESS;
	}

	template <typename T>
	Info DenseMatrix<T>::nvals( Index& nvals ) const
	{
		nvals = nvals_;
		return GrB_SUCCESS;
	}
} // backend
} // graphblas

#endif  // GRB_BACKEND_APSPIE_COOMATRIX_HPP
