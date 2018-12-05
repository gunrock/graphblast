#ifndef GRB_BACKEND_SEQUENTIAL_DENSEMATRIX_HPP
#define GRB_BACKEND_SEQUENTIAL_DENSEMATRIX_HPP

#include <vector>
#include <iostream>
#include <typeinfo>

#include "graphblas/backend/sequential/Matrix.hpp"
//#include "graphblas/backend/sequential/SparseMatrix.hpp"
#include "graphblas/backend/sequential/sequential.hpp"
#include "graphblas/backend/sequential/util.hpp"
#include "graphblas/types.hpp"

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
        : nrows_(0), ncols_(0), nvals_(0), denseVal(NULL){}
    DenseMatrix( const Index nrows, const Index ncols ) 
        : nrows_(nrows), ncols_(ncols), nvals_(nrows*ncols), denseVal(NULL){}

    // Assignment Constructor
    // // TODO: Replaces dup in C++
    void operator=( const DenseMatrix& rhs ) {}

    // Destructor
    // TODO
    ~DenseMatrix() {};

    // C API Methods
    //
    // Mutators
    // assume values are in column major format
    Info build( const std::vector<T>& values );
    // private method for setting nrows and ncols
    Info nnew( const Index nrows, const Index ncols );
    // private method for allocation
    Info allocate();  
    Info clear();

    // Accessors
    Info extractTuples( std::vector<T>& values ) const;
    Info print() const; // Const, because host memory unmodified
    // private method for pretty printing
    Info printDense() const;
    Info nrows( Index& nrows ) const;
    Info ncols( Index& ncols ) const;
    Info nvals( Index& nvals ) const;

    private:
    Index nrows_;
    Index ncols_;
    Index nvals_;

    // Dense format
    T* denseVal;
    T* d_denseVal;

    // TODO:
    template <typename c, typename a, typename b>
    friend Info spmm( DenseMatrix<c>&        C,
                      const Semiring&        op,
                      const SparseMatrix<a>& A,
                      const DenseMatrix<b>&  B );

    // TODO:
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
    // TODO:  
    template <typename c, typename a, typename b>
    friend Info mkl_spmm( DenseMatrix<c>&        C,
                          const Semiring&        op,
                          const SparseMatrix<a>& A,
                          const DenseMatrix<b>&  B );
  };

  template <typename T>
  Info DenseMatrix<T>::build( const std::vector<T>& values )
  {
    allocate();

    // Host copy
    for( graphblas::Index i=0; i<nvals_; i++ )
        denseVal[i] = values[i];

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
    denseVal = (T*)malloc(nvals_*sizeof(T));
    for( Index i=0; i<nvals_; i++ )
      denseVal[i] = (T) 0;

    return GrB_SUCCESS;
  }

  template <typename T>
  Info DenseMatrix<T>::clear()
  {
    if( denseVal ) free( denseVal );
    return GrB_SUCCESS;
  }

  template <typename T>
  Info DenseMatrix<T>::extractTuples( std::vector<T>& values ) const
  {
    values.clear();

    for( Index i=0; i<nvals_; i++ ) {
      values.push_back( denseVal[i] );
    }
    return GrB_SUCCESS;
  }

  template <typename T>
  Info DenseMatrix<T>::print() const
  {
    printArray( "denseVal", denseVal );
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
        #ifdef ROW_MAJOR
        if( denseVal[row*ncols_+col]!=0.0 ) std::cout << "x ";
        else std::cout << "0 ";
        #endif
        // Print column major order matrix in row major order (Transposition)
        #ifdef COL_MAJOR
        if( denseVal[col*nrows_+row]!=0.0 ) std::cout << "x ";
        else std::cout << "0 ";
        #endif
      }
      std::cout << std::endl;
    }
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

#endif  // GRB_BACKEND_SEQUENTIAL_DENSEMATRIX_HPP
