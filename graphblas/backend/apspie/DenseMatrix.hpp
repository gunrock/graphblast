#ifndef GRB_BACKEND_APSPIE_DENSEMATRIX_HPP
#define GRB_BACKEND_APSPIE_DENSEMATRIX_HPP

#include <vector>
#include <iostream>
#include <typeinfo>

#include "graphblas/types.hpp"

#include "graphblas/backend/apspie/Matrix.hpp"

namespace graphblas
{
namespace backend
{
  template <typename T>
  class SparseMatrix;

  template <typename T>
  class DenseMatrix
  {
    public:
    DenseMatrix() 
        : nrows_(0), ncols_(0), nvals_(0), h_denseVal_(NULL) {}
    DenseMatrix( const Index nrows, const Index ncols ) 
        : nrows_(nrows), ncols_(ncols), nvals_(nrows*ncols), h_denseVal_(NULL) {}

    // Destructor
    // TODO
    ~DenseMatrix() {};

    // C API Methods
    //
    // Mutators
    Info dup( const DenseMatrix& rhs );
    // assume values are in column major format
    Info build( const std::vector<T>* values, Index nvals );
    // private method for setting nrows and ncols
    Info nnew( const Index nrows, const Index ncols );
    // private method for allocation
    Info allocate();  
    Info clear();
    Info setNrows( const Index nrows );
    Info setNcols( const Index ncols );

    Info extractTuples( std::vector<T>& values );
    Info print(); // Not const, because host memory modified

    // Accessors
    // private method for pretty printing
    Info printDense() const;
    Info getNrows( Index& nrows ) const;
    Info getNcols( Index& ncols ) const;
    Info getNvals( Index& nvals ) const;

    private:
    Index nrows_;
    Index ncols_;
    Index nvals_;

    // Dense format
    T* h_denseVal_;

    template <typename c, typename a, typename b>
    friend Info spmm( DenseMatrix<c>&        C,
                      const SparseMatrix<a>& A,
                      const DenseMatrix<b>&  B );

    // For testing
    template <typename c, typename a, typename b>
    friend Info spmm( DenseMatrix<c>&        C,
                      const SparseMatrix<a>& A,
                      const DenseMatrix<b>&  B,
                      const int TA,
                      const int TB,
                      const int NT,
                      const bool ROW_MAJOR );
    
    template <typename c, typename a, typename b>
    friend Info cusparse_spmm( DenseMatrix<c>&        C,
                               const SparseMatrix<a>& A,
                               const DenseMatrix<b>&  B );
  };

  template <typename T>
  Info DenseMatrix<T>::dup( const DenseMatrix& rhs )
  {
    if( nrows_ != rhs.nrows_ ) return GrB_DIMENSION_MISMATCH;
    if( ncols_ != rhs.ncols_ ) return GrB_DIMENSION_MISMATCH;
    nvals_ = rhs.nvals_;

    Info err = allocate();
    if( err != GrB_SUCCESS ) return err;

    //std::cout << "copying " << nrows_+1 << " rows\n";
    //std::cout << "copying " << nvals_+1 << " rows\n";

    memcpy( h_denseVal_, rhs.h_denseVal_, (nvals_+1)*sizeof(T) );

    return GrB_SUCCESS;
  }

  template <typename T>
  Info DenseMatrix<T>::build( const std::vector<T>* values, Index nvals )
  {
    if( nvals > nvals_ ) return GrB_DIMENSION_MISMATCH;

    Info err = allocate();
    if( err != GrB_SUCCESS ) return err;

    // Host copy
    for( graphblas::Index i=0; i<nvals_; i++ )
      h_denseVal_[i] = values[i];

    return GrB_SUCCESS;
  }

  template <typename T>
  Info DenseMatrix<T>::nnew( const Index nrows, const Index ncols )
  {
    nrows_ = nrows;
    ncols_ = ncols;
    nvals_ = nrows_*ncols_;
    return GrB_SUCCESS;
  }

  template <typename T>
  Info DenseMatrix<T>::allocate()
  {
    // Host alloc
    h_denseVal_ = (T*)malloc(nvals_*sizeof(T));
    for( Index i=0; i<nvals_; i++ )
      h_denseVal_[i] = (T) 0;

    return GrB_SUCCESS;
  }

  template <typename T>
  Info DenseMatrix<T>::clear()
  {
    if( h_denseVal_ ) free( h_denseVal_ );
    return GrB_SUCCESS;
  }

  template <typename T>
  Info DenseMatrix<T>::extractTuples( std::vector<T>& values )
  {
    values.clear();

    for( Index i=0; i<nvals_; i++ ) {
      values.push_back( h_denseVal_[i] );
    }
    return GrB_SUCCESS;
  }

  template <typename T>
  Info DenseMatrix<T>::setNrows( const Index nrows )
  {
    nrows_ = nrows;
    return GrB_SUCCESS;
  }

  template <typename T>
  Info DenseMatrix<T>::setNcols( const Index ncols )
  {
    ncols_ = ncols;
    return GrB_SUCCESS;
  }

  template <typename T>
  Info DenseMatrix<T>::print()
  {
    printArray( "denseVal", h_denseVal_ );
    printDense();
    return GrB_SUCCESS;
  }

  template <typename T>
  Info DenseMatrix<T>::printDense() const
  {
    int length=std::min(20,nrows_);
    for( int row=0; row<length; row++ ) {
      for( int col=0; col<length; col++ ) {
        // Print row major order matrix in row major order
        //#ifdef ROW_MAJOR
        if( h_denseVal_[row*ncols_+col]!=0.0 ) std::cout << "x ";
        else std::cout << "0 ";
        //#endif
        // Print column major order matrix in row major order (Transposition)
        //#ifdef COL_MAJOR
        //if( h_denseVal_[col*nrows_+row]!=0.0 ) std::cout << "x ";
        //else std::cout << "0 ";
        //#endif
      }
      std::cout << std::endl;
    }
    return GrB_SUCCESS;
  }

  template <typename T>
  Info DenseMatrix<T>::getNrows( Index& nrows ) const
  {
    nrows = nrows_;
    return GrB_SUCCESS;
  }

  template <typename T>
  Info DenseMatrix<T>::getNcols( Index& ncols ) const
  {
    ncols = ncols_;
    return GrB_SUCCESS;
  }

  template <typename T>
  Info DenseMatrix<T>::getNvals( Index& nvals ) const
  {
    nvals = nvals_;
    return GrB_SUCCESS;
  }
} // backend
} // graphblas

#endif  // GRB_BACKEND_APSPIE_DENSEMATRIX_HPP
