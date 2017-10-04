#ifndef GRB_BACKEND_SEQUENTIAL_SPARSEMATRIX_HPP
#define GRB_BACKEND_SEQUENTIAL_SPARSEMATRIX_HPP

#include <vector>
#include <iostream>
#include <typeinfo>

#include "graphblas/backend/sequential/Matrix.hpp"
#include "graphblas/backend/sequential/sequential.hpp"
#include "graphblas/backend/sequential/util.hpp"
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
        csrColInd(NULL), csrRowPtr(NULL), csrVal(NULL) {}

    SparseMatrix( const Index nrows, const Index ncols )
        : nrows_(nrows), ncols_(ncols), nvals_(0),
        csrColInd(NULL), csrRowPtr(NULL), csrVal(NULL) {}

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
                        std::vector<T>&     values ) const;

    // Mutators
    // private method for setting nrows and ncols
    Info nnew( const Index nrows, const Index ncols );
    // private method for allocation
    Info allocate();  
    Info clear();
    Info print(); 
    Info printCSR( const char* str ); // private method for pretty printing

    // Accessors
    Info nrows( Index& nrows ) const;
    Info ncols( Index& ncols ) const;
    Info nvals( Index& nvals ) const;
    Info count() const;

    private:
    Index nrows_;
    Index ncols_;
    Index nvals_;

    // CSR format
    Index* csrColInd;
    Index* csrRowPtr;
    T*     csrVal;

    // CSC format
    // TODO: add CSC support. 
    // -this will be useful and necessary for direction-optimized SpMV
    /*Index* cscRowInd;
    Index*   cscColPtr;
    T*       cscVal;*/

    // TODO: add sequential single-threaded spmm
    template <typename c, typename a, typename b>
    friend Info spmm( DenseMatrix<c>&        C,
                      const Semiring&        op,
                      const SparseMatrix<a>& A,
                      const DenseMatrix<b>&  B );

    // TODO: add sequential single-threaded spmm
    // For testing
    template <typename c, typename a, typename b>
    friend Info spmm( DenseMatrix<c>&        C,
                      const Semiring&        op,
                      const SparseMatrix<a>& A,
                      const DenseMatrix<b>&  B,
                      const int TA,
                      const int TB,
                      const int NT,
                      const bool ROW_MAJOR );

    // TODO: add mkl_spmm
    //template <typename c, typename a, typename b>
    //friend Info mkl_spmm( DenseMatrix<c>&        C,
    //                      const Semiring&        op,
    //                      const SparseMatrix<a>& A,
    //                      const DenseMatrix<b>&  B );

    template <typename c, typename a, typename b>
    friend Info mkl_spgemm( SparseMatrix<c>&       C,
                            const Semiring&        op,
                            const SparseMatrix<a>& A,
                            const SparseMatrix<b>& B );

    template <typename c, typename a, typename b>
    friend Info mkl_spgemm_analyze( SparseMatrix<c>&       C,
                            const Semiring&        op,
                            const SparseMatrix<a>& A,
                            const SparseMatrix<b>& B );

    template <typename c, typename a, typename b>
    friend Info mkl_spgemm_compute( SparseMatrix<c>&       C,
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
    nvals_ = nvals;

    allocate();

    // Convert to CSR/CSC
    Index temp, row, col, dest, cumsum=0;

    // Set all rowPtr to 0
    for( Index i=0; i<=nrows_; i++ )
      csrRowPtr[i] = 0;
    // Go through all elements to see how many fall in each row
    for( Index i=0; i<nvals_; i++ ) {
      row = row_indices[i];
      if( row>=nrows_ ) return GrB_INDEX_OUT_OF_BOUNDS;
      csrRowPtr[ row ]++;
    }
    // Cumulative sum to obtain rowPtr
    for( Index i=0; i<nrows_; i++ ) {
      temp = csrRowPtr[i];
      csrRowPtr[i] = cumsum;
      cumsum += temp;
    }
    csrRowPtr[nrows_] = nvals;

    // Store colInd and val
    for( Index i=0; i<nvals_; i++ ) {
      row = row_indices[i];
      dest= csrRowPtr[row];
      col = col_indices[i];
      if( col>=ncols_ ) return GrB_INDEX_OUT_OF_BOUNDS;
      csrColInd[dest] = col+1;      // One-based indexing uses col+1
      //csrColInd[dest] = col;      // Zero-based indexing uses col+1
      csrVal[dest]    = values[i];
      csrRowPtr[row]++;
    }
    cumsum = 0;
    
    // Undo damage done to rowPtr
    for( Index i=0; i<=nrows_; i++ ) {
      temp = csrRowPtr[i];
      csrRowPtr[i] = cumsum+1;      // One-based indexing
      //csrRowPtr[i] = cumsum;      // Zero-based indexing
      cumsum = temp;
    }

    return GrB_SUCCESS;
  }

  template <typename T>
  Info SparseMatrix<T>::extractTuples( std::vector<Index>& row_indices,
                                       std::vector<Index>& col_indices,
                                       std::vector<T>&     values ) const
  {
    row_indices.clear();
    col_indices.clear();
    values.clear();

    for( Index row=0; row<nrows_; row++ ) {
      for( Index ind=csrRowPtr[row]; ind<csrRowPtr[row+1]; ind++ ) {
        row_indices.push_back(row);
        col_indices.push_back(csrColInd[ind]);
        values.push_back(     csrVal[ind]);
      }
    }

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
  Info SparseMatrix<T>::allocate()
  {
    // Host malloc
    if( nrows_!=0 && csrRowPtr == NULL ) 
      csrRowPtr = (Index*)malloc((nrows_+1)*sizeof(Index));
    if( nvals_!=0 && csrColInd == NULL )
      csrColInd = (Index*)malloc(nvals_*sizeof(Index));
    if( nvals_!=0 && csrVal == NULL )
      csrVal    = (T*)    malloc(nvals_*sizeof(T));

    if( csrRowPtr==NULL ) return GrB_OUT_OF_MEMORY;
    if( csrColInd==NULL ) return GrB_OUT_OF_MEMORY;
    if( csrVal==NULL )    return GrB_OUT_OF_MEMORY;

    return GrB_SUCCESS;
  }

  template <typename T>
  Info SparseMatrix<T>::clear()
  {
    if( csrRowPtr ) free( csrRowPtr );
    if( csrColInd ) free( csrColInd );
    if( csrVal )    free( csrVal );
    return GrB_SUCCESS;
  }

  template <typename T>
  Info SparseMatrix<T>::print()
  {
    printArray( "csrColInd", csrColInd );
    printArray( "csrRowPtr", csrRowPtr );
    printArray( "csrVal",    csrVal );
    printCSR( "pretty print" );
    return GrB_SUCCESS;
  }

  template <typename T>
  Info SparseMatrix<T>::printCSR( const char* str )
  {
    Index length = std::min(20, nrows_);
    std::cout << str << ":\n";

    for( Index row=0; row<length; row++ ) {
      Index col_start = csrRowPtr[row];
      Index col_end   = csrRowPtr[row+1];
      for( Index col=0; col<length; col++ ) {
        Index col_ind = csrColInd[col_start];
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
  Info SparseMatrix<T>::count() const
  {
    std::vector<int> count(32,0);
    for( Index i=0; i<nrows_; i++ )
    {
      int diff = h_csrRowPtr_[i+1]-h_csrRowPtr_[i];
      count[diff&31]++;
    }

    printArray( "count", count, 32 );
    return GrB_SUCCESS;
  }
} // backend
} // graphblas

#endif  // GRB_BACKEND_SEQUENTIAL_SPARSEMATRIX_HPP
