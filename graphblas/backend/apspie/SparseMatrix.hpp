#ifndef GRB_BACKEND_APSPIE_SPARSEMATRIX_HPP
#define GRB_BACKEND_APSPIE_SPARSEMATRIX_HPP

#include <vector>
#include <iostream>
#include <typeinfo>
#include <map>
#include <cassert>
#include <algorithm>

#include <cuda.h>
#include <cuda_runtime.h>

#include "graphblas/util.hpp"

#include "graphblas/backend/apspie/apspie.hpp"
#include "graphblas/backend/apspie/Matrix.hpp"

namespace graphblas
{
namespace backend
{
  template <typename T>
  class DenseMatrix;

  template <typename T>
  class Vector;

  template <typename T>
  class SparseMatrix
  {
    public:
    SparseMatrix()
        : nrows_(0), ncols_(0), nvals_(0), ncapacity_(0), nempty_(0), 
          h_csrColInd_(NULL), h_csrRowPtr_(NULL), h_csrVal_(NULL),
          d_csrColInd_(NULL), d_csrRowPtr_(NULL), d_csrVal_(NULL),
          buffer_(NULL),      need_update_(0) {}

    SparseMatrix( Index nrows, Index ncols )
        : nrows_(nrows), ncols_(ncols), nvals_(0), ncapacity_(0), nempty_(0),
          h_csrColInd_(NULL), h_csrRowPtr_(NULL), h_csrVal_(NULL),
          d_csrColInd_(NULL), d_csrRowPtr_(NULL), d_csrVal_(NULL),
          buffer_(NULL),      need_update_(0) {}

    // C API Methods
    Info nnew(  Index nrows, Index ncols );
    Info dup(   const SparseMatrix* rhs );
    Info clear();     // 1 way to free: (1) clear
    Info nrows( Index* nrows_t ) const;
    Info ncols( Index* ncols_t ) const;
    Info nvals( Index* nvals_t ) const;
    Info build( const std::vector<Index>* row_indices,
                const std::vector<Index>* col_indices,
                const std::vector<T>*     values,
                Index                     nvals,
                const BinaryOp            dup );
    Info build( const std::vector<T>* values,
                Index nvals );
    Info setElement(     Index row_index,
                         Index col_index );
    Info extractElement( T*    val,
                         Index row_index,
                         Index col_index );
    Info extractTuples(  std::vector<Index>* row_indices,
                         std::vector<Index>* col_indices,
                         std::vector<T>*     values,
                         Index*              n );
    Info extractTuples(  std::vector<T>* values,
                         Index*          n );

    // Handy methods
    const T operator[]( Index ind );
    Info print( bool forceUpdate=false ); 
    Info check();
    Info setNrows( Index nrows );
    Info setNcols( Index ncols );
    Info resize(   Index nrows, 
                   Index ncols );
    template <typename U>
    Info fill( Index axis, 
               Index nvals, 
               U     start );
    template <typename U>
    Info fillAscending( Index axis, 
                        Index nvals, 
                        U     start );

    private:
    Info allocate();  // 3 ways to allocate: (1) dup, (2) build, (3) spgemm
                      //                     (4) fill,(5) fillAscending
    Info printCSR( const char* str ); // private method for pretty printing
    Info cpuToGpu();
    Info gpuToCpu( bool forceUpdate=false );

    private:
    const T kcap_ratio_ = 1.3;
    const T kresize_ratio_ = 1.3;

    Index nrows_;
    Index ncols_;
    Index nvals_;    // 3 ways to set: (1) dup (2) build (3) nnew
    Index ncapacity_;
    Index nempty_;

    // CSR format
    Index* h_csrColInd_;
    Index* h_csrRowPtr_;
    T*     h_csrVal_;

    // GPU CSR
    Index* d_csrColInd_;
    Index* d_csrRowPtr_;
    T*     d_csrVal_;

    // GPU variables
    void*  buffer_;
    size_t buffer_size_;
    bool   need_update_;
    // CSC format
    // TODO: add CSC support. 
    // -this will be useful and necessary for direction-optimized SpMV
    /*Index* h_cscRowInd;
    Index* h_cscColPtr;
    T*     h_cscVal;*/

    template <typename a, typename b>
    friend Info cubReduce( Vector<b>&             B,
                           const SparseMatrix<a>& A,
                           void*                  d_buffer,
                           size_t                 buffer_size );

  };

  template <typename T>
  Info SparseMatrix<T>::nnew( Index nrows, Index ncols )
  {
    nrows_ = nrows;
    ncols_ = ncols;

    //Info err = allocate();
    return GrB_SUCCESS;//err;
  }

  template <typename T>
  Info SparseMatrix<T>::dup( const SparseMatrix* rhs )
  {
    if( nrows_ != rhs->nrows_ ) return GrB_DIMENSION_MISMATCH;
    if( ncols_ != rhs->ncols_ ) return GrB_DIMENSION_MISMATCH;
    nvals_ = rhs->nvals_;

    Info err = allocate();
    if( err != GrB_SUCCESS ) return err;

    //std::cout << "copying " << nrows_+1 << " rows\n";
    //std::cout << "copying " << nvals_+1 << " rows\n";

    CUDA( cudaMemcpy( d_csrRowPtr_, rhs->d_csrRowPtr_, (nrows_+1)*sizeof(Index),
        cudaMemcpyDeviceToDevice ) );
    CUDA( cudaMemcpy( d_csrColInd_, rhs->d_csrColInd_, nvals_*sizeof(Index),
        cudaMemcpyDeviceToDevice ) );
    CUDA( cudaMemcpy( d_csrVal_,    rhs->d_csrVal_,    nvals_*sizeof(T),
        cudaMemcpyDeviceToDevice ) );
    CUDA( cudaDeviceSynchronize() );

    need_update_ = true;
    return GrB_SUCCESS; 
  }

  template <typename T>
  Info SparseMatrix<T>::clear()
  {
    if( h_csrRowPtr_ ) {
      free( h_csrRowPtr_ );
      h_csrRowPtr_ = NULL;
    }
    if( h_csrColInd_ ) {
      free( h_csrColInd_ );
      h_csrColInd_ = NULL;
    }
    if( h_csrVal_ ) {
      free( h_csrVal_ );
      h_csrVal_ = NULL;
    }

    if( d_csrRowPtr_ ) {
      CUDA( cudaFree(d_csrRowPtr_) );
      d_csrRowPtr_ = NULL;
    }
    if( d_csrColInd_ ) {
      CUDA( cudaFree(d_csrColInd_) );
      d_csrColInd_ = NULL;
    }
    if( d_csrVal_ ) {
      CUDA( cudaFree(d_csrVal_) );
      d_csrVal_ = NULL;
    }
    ncapacity_ = 0;

    return GrB_SUCCESS;
  }

  template <typename T>
  inline Info SparseMatrix<T>::nrows( Index* nrows_t ) const
  {
    if( nrows_t==NULL ) return GrB_NULL_POINTER;
    *nrows_t = nrows_;
    return GrB_SUCCESS;
  }

  template <typename T>
  inline Info SparseMatrix<T>::ncols( Index* ncols_t ) const
  {
    if( ncols_t==NULL ) return GrB_NULL_POINTER;
    *ncols_t = ncols_;
    return GrB_SUCCESS;
  }

  template <typename T>
  inline Info SparseMatrix<T>::nvals( Index* nvals_t ) const
  {
    if( nvals_t==NULL ) return GrB_NULL_POINTER;
    *nvals_t = nvals_;
    return GrB_SUCCESS;
  }

  template <typename T>
  Info SparseMatrix<T>::build( const std::vector<Index>* row_indices,
                               const std::vector<Index>* col_indices,
                               const std::vector<T>*     values,
                               Index                     nvals,
                               const BinaryOp            dup )
  {
    nvals_ = nvals;
    Info err = allocate();
    if( err != GrB_SUCCESS ) return err;
    Index temp, row, col, dest, cumsum=0;

    // Convert to CSR if tranpose is false
    //            CSC if tranpose is true
    /*std::vector<Index> &row_indices = transpose ? col_indices_t : 
      row_indices_t;
    std::vector<Index> &col_indices = transpose ? row_indices_t :
      col_indices_t;

    customSort<T>( row_indices, col_indices, values );*/
    
    // Set all rowPtr to 0
    for( Index i=0; i<=nrows_; i++ )
      h_csrRowPtr_[i] = 0;
    // Go through all elements to see how many fall in each row
    for( Index i=0; i<nvals_; i++ ) {
      row = (*row_indices)[i];
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
      row = (*row_indices)[i];
      dest= h_csrRowPtr_[row];
      col = (*col_indices)[i];
      if( col>=ncols_ ) return GrB_INDEX_OUT_OF_BOUNDS;
      h_csrColInd_[dest] = col;
      h_csrVal_[dest]    = (*values)[i];
      h_csrRowPtr_[row]++;
    }
    cumsum = 0;
    
    // Undo damage done to rowPtr
    for( Index i=0; i<nrows_; i++ ) {
      temp = h_csrRowPtr_[i];
      h_csrRowPtr_[i] = cumsum;
      cumsum = temp;
    }
    temp = h_csrRowPtr_[nrows_];
    h_csrRowPtr_[nrows_] = cumsum;
    cumsum = temp;

    err = cpuToGpu();

    return err;
  }

  template <typename T>
  Info Sparsematrix<T>::build( const std::vector<T>* values,
                               Index                 nvals )
  {
    return GrB_SUCCESS;
  }

  template <typename T>
  Info Matrix<T>::setElement( Index row_index,
                              Index col_index )
  {
    return GrB_SUCCESS;
  }

  template <typename T>
  Info Matrix<T>::extractElement( T*    val,
                                  Index row_index,
                                  Index col_index )
  {
    return GrB_SUCCESS;
  }

  template <typename T>
  Info SparseMatrix<T>::extractTuples( std::vector<Index>* row_indices,
                                       std::vector<Index>* col_indices,
                                       std::vector<T>*     values,
                                       Index*              n )
  {
    Info err = gpuToCpu();
    row_indices->clear();
    col_indices->clear();
    values->clear();

    if( n==NULL ) return GrB_NULL_POINTER;
    if( *n>nvals_ )
    {
      err = GrB_UNINITIALIZED_OBJECT;
      *n  = nvals_;
    }
    else if( *n<nvals_ )
      err = GrB_INSUFFICIENT_SPACE;

    Index count = 0;
    for( Index row=0; row<nrows_; row++ ) {
      for( Index ind=h_csrRowPtr_[row]; ind<h_csrRowPtr_[row+1]; ind++ ) {
        if( h_csrVal_[ind]!=0 && count<*n )
        {
          count++;
          row_indices.push_back(row);
          col_indices.push_back(h_csrColInd_[ind]);
          values.push_back(     h_csrVal_[ind]);
        }
      }
    }

    return err;
  }

  template <typename T>
  Info SparseMatrix<T>::extractTuples( std::vector<T>* values, Index* n )
  {
    return GrB_SUCCESS;
  }

  template <typename T>
  const T SparseMatrix<T>::operator[]( Index ind )
  {
    gpuToCpu(true);
    if( ind>=nvals_ ) std::cout << "Error: index out of bounds!\n";

    return h_csrColInd_[ind];
  }

  template <typename T>
  Info SparseMatrix<T>::print( bool forceUpdate )
  {
    Info err = gpuToCpu( forceUpdate );
    printArray( "csrColInd", h_csrColInd_, std::min(nvals_,40) );
    printArray( "csrRowPtr", h_csrRowPtr_, std::min(nrows_+1,40) );
    printArray( "csrVal",    h_csrVal_,    std::min(nvals_,40) );
    printCSR( "pretty print" );
    return err;
  }

  template <typename T>
  Info SparseMatrix<T>::check()
  {
    Info err = gpuToCpu();
    std::cout << "Begin check:\n";
    //printArray( "rowptr", h_csrRowPtr_ );
    //printArray( "colind", h_csrColInd_+23 );
    // Check csrRowPtr is monotonically increasing
    for( Index row=0; row<nrows_; row++ )
    {
      //std::cout << "Comparing " << h_csrRowPtr_[row+1] << " >= " << h_csrRowPtr_[row] << std::endl;
      assert( h_csrRowPtr_[row+1]>=h_csrRowPtr_[row] );
    }

    // Check that: 1) there are no -1's in ColInd
    //             2) monotonically increasing
    for( Index row=0; row<nrows_; row++ )
    {
      Index row_start = h_csrRowPtr_[row];
      Index row_end   = h_csrRowPtr_[row+1];
      Index p_end     = h_csrRowPtr_[row+1];
      //std::cout << row << " " << row_end-row_start << std::endl;
      //printArray( "colind", h_csrColInd_+row_start, p_end-row_start );
      //printArray( "val", h_csrVal_+row_start, p_end-row_start );
      for( Index col=row_start; col<row_end-1; col++ )
      {
        //std::cout << "Comparing " << h_csrColInd_[col+1] << " >= " << h_csrColInd_[col] << std::endl;
        assert( h_csrColInd_[col]!=-1 );
        assert( h_csrColInd_[col+1]>=h_csrColInd_[col] );
        assert( h_csrVal_[col]>0 );
      }
      for( Index col=row_end; col<p_end; col++ )
        assert( h_csrColInd_[col]==-1 );
    }
    return err;
  }

  template <typename T>
  Info SparseMatrix<T>::setNrows( Index nrows )
  {
    nrows_ = nrows;
    return GrB_SUCCESS;
  }

  template <typename T>
  Info SparseMatrix<T>::setNcols( Index ncols )
  {
    ncols_ = ncols;
    return GrB_SUCCESS;
  }

  // Note: has different meaning from sequential resize
  //      -that one makes SparseMatrix bigger
  //      -this one accounts for smaller nrows
  template <typename T>
  Info SparseMatrix<T>::resize( Index nrows, Index ncols )
  {
    if( nrows<=nrows_ )
      nrows_ = nrows;
    else return GrB_PANIC;
    if( ncols<=ncols_ )
      ncols_ = ncols;
    else return GrB_PANIC;

    return GrB_SUCCESS;
  }

  template <typename T>
  template <typename U>
  Info SparseMatrix<T>::fill( Index axis, 
                              Index nvals,
                              U     start )
  {
    Info err;
    err = allocate();

    if( axis==0 )
      for( Index i=0; i<nvals; i++ )
        h_csrRowPtr_[i] = (Index) start;
    else if( axis==1 )
      for( Index i=0; i<nvals; i++ )
        h_csrColInd_[i] = (Index) start;
    else if( axis==2 )
      for( Index i=0; i<nvals; i++ )
        h_csrVal_[i] = (T) start;

    err = cpuToGpu();
    return err;
  }

  template <typename T>
  template <typename U>
  Info SparseMatrix<T>::fillAscending( Index axis, 
                                       Index nvals,
                                       U     start )
  {
    Info err;
    err = allocate();

    if( axis==0 )
      for( Index i=0; i<nvals; i++ )
        h_csrRowPtr_[i] = i+(Index) start;
    else if( axis==1 )
      for( Index i=0; i<nvals; i++ )
        h_csrColInd_[i] = i+(Index) start;
    else if( axis==2 )
      for( Index i=0; i<nvals; i++ )
        h_csrVal_[i] = (T)i+start;

    err = cpuToGpu();
    return err;
  }

  template <typename T>
  Info SparseMatrix<T>::allocate()
  {
    // Allocate
    ncapacity_ = kcap_ratio_*nvals_;

    // Host malloc
    if( nrows_!=0 && h_csrRowPtr_ == NULL ) 
      h_csrRowPtr_ = (Index*)malloc((nrows_+1)*sizeof(Index));
    else
    {
      //std::cout << "hrow: " << nrows_ << " " << (h_csrRowPtr_==NULL) << std::endl;
      //return GrB_UNINITIALIZED_OBJECT;
    }
    if( nvals_!=0 && h_csrColInd_ == NULL )
      h_csrColInd_ = (Index*)malloc(ncapacity_*sizeof(Index));
    else
    {
      //std::cout << "hcol: " << nvals_ << " " << (h_csrColInd_==NULL) << std::endl;
      //return GrB_UNINITIALIZED_OBJECT;
    }
    if( nvals_!=0 && h_csrVal_ == NULL )
      h_csrVal_    = (T*)    malloc(ncapacity_*sizeof(T));
    else
    {
      //std::cout << "hval: " << nvals_ << " " << (h_csrVal_==NULL) << std::endl;
      //return GrB_UNINITIALIZED_OBJECT;
    }

    // GPU malloc
    if( nrows_!=0 && d_csrRowPtr_ == NULL )
      CUDA( cudaMalloc( &d_csrRowPtr_, (nrows_+1)*sizeof(Index)) );
    else
    {
      //std::cout << "drow: " << nrows_ << " " << (d_csrRowPtr_==NULL) << std::endl;
      //return GrB_UNINITIALIZED_OBJECT;
    }
    if( nvals_!=0 && d_csrColInd_ == NULL )
      CUDA( cudaMalloc( &d_csrColInd_, ncapacity_*sizeof(Index)) );
    else
    {
      //std::cout << "dcol: " << nvals_ << " " << (d_csrColInd_==NULL) << std::endl;
      //return GrB_UNINITIALIZED_OBJECT;
    }
    if( nvals_!=0 && d_csrVal_ == NULL )
      CUDA( cudaMalloc( &d_csrVal_, ncapacity_*sizeof(T)) );
    else
    {
      //std::cout << "dval: " << nvals_ << " " << (d_csrVal_==NULL) << std::endl;
      //return GrB_UNINITIALIZED_OBJECT;
    }

    if( h_csrRowPtr_==NULL || h_csrColInd_==NULL || h_csrVal_==NULL ||
        d_csrRowPtr_==NULL || d_csrColInd_==NULL || d_csrVal_==NULL ) 
      return GrB_OUT_OF_MEMORY;

    return GrB_SUCCESS;
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
        if( col_start<col_end && col_ind==col && h_csrVal_[col_start]>0 ) {
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
  Info SparseMatrix<T>::gpuToCpu( bool forceUpdate )
  {
    if( need_update_ || forceUpdate )
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

}  // backend
}  // graphblas

#endif  // GRB_BACKEND_APSPIE_SPARSEMATRIX_HPP
