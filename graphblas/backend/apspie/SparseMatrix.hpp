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
#include "graphblas/util.hpp"

namespace graphblas
{
namespace backend
{
  template <typename T>
  class DenseMatrix;

  template <typename T>
  class SparseMatrix
  {
    public:
    SparseMatrix()
        : nrows_(0), ncols_(0), nvals_(0), 
        h_csrColInd_(NULL), h_csrRowPtr_(NULL), h_csrVal_(NULL),
        d_csrColInd_(NULL), d_csrRowPtr_(NULL), d_csrVal_(NULL) {}

    SparseMatrix( const Index nrows, const Index ncols )
        : nrows_(nrows), ncols_(ncols), nvals_(0),
        h_csrColInd_(NULL), h_csrRowPtr_(NULL), h_csrVal_(NULL),
        d_csrColInd_(NULL), d_csrRowPtr_(NULL), d_csrVal_(NULL) 
		{
			//allocate();
		}

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

    Info extractTuples( std::vector<Index>& row_indices,
                        std::vector<Index>& col_indices,
                        std::vector<T>&     values );

    // Mutators
    // private method for setting nrows and ncols
    Info nnew( const Index nrows, const Index ncols );
    // private method for allocation
    Info allocate(); // 3 ways to allocate: (1) dup, (2) build, (3) spgemm 
    Info clear();    // 1 way to free: (1) clear
    Info print(); 
    Info printCSR( const char* str ); // private method for pretty printing
    Info cpuToGpu();
    Info gpuToCpu();

    // Accessors
    Info nrows( Index& nrows ) const;
    Info ncols( Index& ncols ) const;
    Info nvals( Index& nvals ) const;
    Info printStats() const;

    private:
    Index nrows_;
    Index ncols_;
    Index nvals_;

    // CSR format
    Index* h_csrColInd_;
    Index* h_csrRowPtr_;
    T*     h_csrVal_;
    Index* d_csrColInd_;
    Index* d_csrRowPtr_;
    T*     d_csrVal_;

    // CSC format
    // TODO: add CSC support. 
    // -this will be useful and necessary for direction-optimized SpMV
    /*Index* h_cscRowInd_;
    Index* h_cscColPtr_;
    T*     h_cscVal_;
    Index* d_cscRowInd_;
    Index* d_cscColPtr_;
    T*     d_csrVal_;*/

    // Keep track of whether host values are up-to-date with device values
    bool need_update_;

    template <typename c, typename a, typename b>
    friend Info spmm( DenseMatrix<c>&        C,
                      const Semiring&        op,
                      const SparseMatrix<a>& A,
                      const DenseMatrix<b>&  B );

    // For testing
    template <typename c, typename m, typename a, typename b>
    friend Info spmm( DenseMatrix<c>&        C,
                      const SparseMatrix<m>& mask,
                      const BinaryOp&        accum,
                      const Semiring&        op,
                      const SparseMatrix<a>& A,
                      const DenseMatrix<b>&  B,
                      const Descriptor&      desc );

    template <typename c, typename a, typename b>
    friend Info spmv( DenseMatrix<c>&        C,
                      const Semiring&        op,
                      const SparseMatrix<a>& A,
                      const DenseMatrix<b>&  B );

    template <typename c, typename m, typename a, typename b>
    friend Info spmv( DenseMatrix<c>&        C,
                      const SparseMatrix<m>& mask,
                      const BinaryOp&        accum,
                      const Semiring&        op,
                      const SparseMatrix<a>& A,
                      const DenseMatrix<b>&  B,
                      const Descriptor&      desc );

    template <typename c, typename a, typename b>
    friend Info cusparse_spmm( DenseMatrix<c>&        C,
                               const Semiring&        op,
                               const SparseMatrix<a>& A,
                               const DenseMatrix<b>&  B );

    template <typename c, typename a, typename b>
    friend Info cusparse_spmm2( DenseMatrix<c>&        C,
                                const Semiring&        op,
                                const SparseMatrix<a>& A,
                                const DenseMatrix<b>&  B );

    template <typename c, typename a, typename b>
    friend Info mergepath_spmm( DenseMatrix<c>&        C,
                                const Semiring&        op,
                                const SparseMatrix<a>& A,
                                const DenseMatrix<b>&  B );

    template <typename c, typename a, typename b>
    friend Info cusparse_spgemm( SparseMatrix<c>&       C,
                                 const Semiring&        op,
                                 const SparseMatrix<a>& A,
                                 const SparseMatrix<b>& B );

    template <typename c, typename a, typename b>
    friend Info cusparse_spgemm_analyze( SparseMatrix<c>&       C,
                                 const Semiring&        op,
                                 const SparseMatrix<a>& A,
                                 const SparseMatrix<b>& B );

    template <typename c, typename a, typename b>
    friend Info cusparse_spgemm_compute( SparseMatrix<c>&       C,
                                 const Semiring&        op,
                                 const SparseMatrix<a>& A,
                                 const SparseMatrix<b>& B );

    template <typename c, typename a, typename b>
    friend Info cusparse_spgemm2( SparseMatrix<c>&       C,
                                 const Semiring&        op,
                                 const SparseMatrix<a>& A,
                                 const SparseMatrix<b>& B );

    template <typename c, typename a, typename b>
    friend Info cusparse_spgemm2_compute( SparseMatrix<c>&       C,
                                 const Semiring&        op,
                                 const SparseMatrix<a>& A,
                                 const SparseMatrix<b>& B );
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
    Info err;
    nvals_ = nvals;
    need_update_ = false;

    err = allocate();

    // Convert to CSR/CSC
    Index temp, row, col, dest, cumsum=0;

    // Set all rowPtr to 0
    for( Index i=0; i<=nrows_; i++ )
      h_csrRowPtr_[i] = 0;
    // Go through all elements to see how many fall in each row
    for( Index i=0; i<nvals_; i++ ) {
      row = row_indices[i];
      if( row>=nrows_ ) return GrB_INDEX_OUT_OF_BOUNDS;
      h_csrRowPtr_[ row ]++;
    }
    // Cumulative sum to obtain rowPtr
    for( Index i=0; i<nrows_; i++ ) {
      temp = h_csrRowPtr_[i];
      h_csrRowPtr_[i] = cumsum;
      cumsum += temp;
    }
    h_csrRowPtr_[nrows_] = nvals;

    // Store colInd and val
    for( Index i=0; i<nvals_; i++ ) {
      row = row_indices[i];
      dest= h_csrRowPtr_[row];
      col = col_indices[i];
      if( col>=ncols_ ) return GrB_INDEX_OUT_OF_BOUNDS;
      h_csrColInd_[dest] = col;
      h_csrVal_[dest]    = values[i];
      h_csrRowPtr_[row]++;
    }
    cumsum = 0;
    
    // Undo damage done to rowPtr
    for( Index i=0; i<=nrows_; i++ ) {
      temp = h_csrRowPtr_[i];
      h_csrRowPtr_[i] = cumsum;
      cumsum = temp;
    }

    err = cpuToGpu();

    return err;
  }

  template <typename T>
  Info SparseMatrix<T>::extractTuples( std::vector<Index>& row_indices,
                                       std::vector<Index>& col_indices,
                                       std::vector<T>&     values )
  {
    Info err = gpuToCpu();
    row_indices.clear();
    col_indices.clear();
    values.clear();

    for( Index row=0; row<nrows_; row++ ) {
      for( Index ind=h_csrRowPtr_[row]; ind<h_csrRowPtr_[row+1]; ind++ ) {
        row_indices.push_back(row);
        col_indices.push_back(h_csrColInd_[ind]);
        values.push_back(     h_csrVal_[ind]);
      }
    }

    return err;
  }

  template <typename T>
  Info SparseMatrix<T>::nnew( const Index nrows, const Index ncols )
  {
    nrows_ = nrows;
    ncols_ = ncols;
		//allocate();
    return GrB_SUCCESS;
  }

  template <typename T>
  Info SparseMatrix<T>::allocate()
  {
    // Host malloc
    if( nrows_!=0 && h_csrRowPtr_ == NULL ) 
      h_csrRowPtr_ = (Index*)malloc((nrows_+1)*sizeof(Index));
    if( nvals_!=0 && h_csrColInd_ == NULL )
      h_csrColInd_ = (Index*)malloc(nvals_*sizeof(Index));
    if( nvals_!=0 && h_csrVal_ == NULL )
      h_csrVal_    = (T*)    malloc(nvals_*sizeof(T));

    // Device malloc
    if( nrows_!=0 && d_csrRowPtr_==NULL )
      CUDA(cudaMalloc((void**)&d_csrRowPtr_, 
          (nrows_+1)*sizeof(Index)));
    if( nvals_!=0 && d_csrColInd_==NULL )
      CUDA(cudaMalloc((void**)&d_csrColInd_, nvals_*sizeof(Index)));
    if( nvals_!=0 && d_csrVal_==NULL )
      CUDA(cudaMalloc((void**)&d_csrVal_,    nvals_*sizeof(T))); 
   
    if( h_csrRowPtr_==NULL ) return GrB_OUT_OF_MEMORY;
    if( h_csrColInd_==NULL ) return GrB_OUT_OF_MEMORY;
    if( h_csrVal_==NULL )    return GrB_OUT_OF_MEMORY;
    if( d_csrRowPtr_==NULL ) return GrB_OUT_OF_MEMORY;
    if( d_csrColInd_==NULL ) return GrB_OUT_OF_MEMORY;
    if( d_csrVal_==NULL )    return GrB_OUT_OF_MEMORY;

    return GrB_SUCCESS;
  }

  template <typename T>
  Info SparseMatrix<T>::clear()
  {
    if( h_csrRowPtr_ )
    {
      free( h_csrRowPtr_ );
      h_csrRowPtr_ = NULL;
    }
    if( h_csrColInd_ ) 
    {
      free( h_csrColInd_ );
      h_csrColInd_ = NULL;
    }
    if( h_csrVal_ )
    {
      free( h_csrVal_ );
      h_csrVal_ = NULL;
    }
    if( d_csrRowPtr_ )
    {
      CUDA(cudaFree( d_csrRowPtr_ ));
      d_csrRowPtr_ = NULL;
    }
    if( d_csrColInd_ )
    {
      CUDA(cudaFree( d_csrColInd_ ));
      d_csrColInd_ = NULL;
    }
    if( d_csrVal_ )
    {
      CUDA(cudaFree( d_csrVal_ ));
      d_csrVal_ = NULL;
    }
    return GrB_SUCCESS;
  }

  template <typename T>
  Info SparseMatrix<T>::print()
  {
    Info err = gpuToCpu();
    printArray( "csrColInd", h_csrColInd_ );
    printArray( "csrRowPtr", h_csrRowPtr_ );
    printArray( "csrVal",    h_csrVal_ );
    printCSR( "pretty print" );
    return err;
  }

  template <typename T>
  Info SparseMatrix<T>::printCSR( const char* str )
  {
    Index row_length = std::min(20, nrows_);
    Index col_length = std::min(20, ncols_);
    std::cout << str << ":\n";

    for( Index row=0; row<row_length; row++ ) {
      Index col_start = h_csrRowPtr_[row];
      Index col_end   = h_csrRowPtr_[row+1];
      for( Index col=0; col<col_length; col++ ) {
        Index col_ind = h_csrColInd_[col_start];
        if( col_start<col_end && col_ind==col ) {
          std::cout << "x ";
          col_start++;
        } else {
          std::cout << "0 ";
        }
      }
      std::cout << std::endl;
    }
    return GrB_SUCCESS;
  }

  // Copies graph to GPU
  template <typename T>
  Info SparseMatrix<T>::cpuToGpu()
  {
    CUDA( cudaMemcpy( d_csrRowPtr_, h_csrRowPtr_, (nrows_+1)*sizeof(Index),
        cudaMemcpyHostToDevice ) );
    CUDA( cudaMemcpy( d_csrColInd_, h_csrColInd_, nvals_*sizeof(Index),
        cudaMemcpyHostToDevice ) );
    CUDA( cudaMemcpy( d_csrVal_,    h_csrVal_,    nvals_*sizeof(T),
        cudaMemcpyHostToDevice ) );
    CUDA( cudaDeviceSynchronize() );
    return GrB_SUCCESS;
  }

  // Copies graph to CPU
  template <typename T>
  Info SparseMatrix<T>::gpuToCpu()
  {
    if( need_update_ )
    {
      CUDA( cudaMemcpy( h_csrRowPtr_, d_csrRowPtr_, (nrows_+1)*sizeof(Index),
          cudaMemcpyDeviceToHost ) );
      CUDA( cudaMemcpy( h_csrColInd_, d_csrColInd_, nvals_*sizeof(Index),
          cudaMemcpyDeviceToHost ) );
      CUDA( cudaMemcpy( h_csrVal_,    d_csrVal_,    nvals_*sizeof(T),
          cudaMemcpyDeviceToHost ) );
      CUDA( cudaDeviceSynchronize() );
    }
    need_update_ = false;
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

  template <typename T>
  Info SparseMatrix<T>::printStats() const
  {
    double row_mean = double( nvals_ )/nrows_;
    double variance = 0.f;
    double row_skew = 0.f;
    int    vars[33];
    int    big      = 0;
    for( int i=0; i<33; i++ )
      vars[i] = 0;
    for( Index row=0; row<nrows_; row++ )
    {
      Index length  = h_csrRowPtr_[row+1]-h_csrRowPtr_[row];
      double delta  = double(length) - row_mean;
      variance     += delta*delta;
      row_skew     += delta*delta*delta;
      if( length<32 ) vars[length]++;
      else vars[32]++;
      if( length>=10000 ) big++;
    }
    variance       /= nrows_;
    double row_std  = sqrt(variance);
    row_skew        = row_skew/nrows_/pow(row_std, 3.0);
    double row_var  = row_std/row_mean;

    for( int i=0; i<33; i++ )
      std::cout << vars[i] << ", ";
    std::cout << big << ", " << row_mean << ", " << row_std << ", " << row_var << ", " << row_skew << ", ";

    return GrB_SUCCESS;
    /*std::vector<int> count(32,0);
    for( Index i=0; i<nrows_; i++ )
    {
      int diff = h_csrRowPtr_[i+1]-h_csrRowPtr_[i];
      count[diff&31]++;
    }

    printArray( "count", count, 32 );
    return GrB_SUCCESS;*/
  }
} // backend
} // graphblas

#endif  // GRB_BACKEND_APSPIE_SPARSEMATRIX_HPP
