#ifndef GRB_BACKEND_APSPIE_DENSEMATRIX_HPP
#define GRB_BACKEND_APSPIE_DENSEMATRIX_HPP

#include <vector>
#include <iostream>
#include <typeinfo>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "graphblas/backend/apspie/Matrix.hpp"
//#include "graphblas/backend/apspie/SparseMatrix.hpp"
#include "graphblas/backend/apspie/apspie.hpp"
#include "graphblas/backend/apspie/util.hpp"
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
        : nrows_(0), ncols_(0), nvals_(0), h_denseVal(NULL), d_denseVal(NULL){}
    DenseMatrix( const Index nrows, const Index ncols ) 
        : nrows_(nrows), ncols_(ncols), nvals_(nrows*ncols), 
        h_denseVal(NULL), d_denseVal(NULL) {}

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
    Info setMajor( const Major major_type );

    // Accessors
    Info extractTuples( std::vector<T>& values );
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
    T* h_denseVal;
    T* d_denseVal;

    // Keeps track of GrB_ROWMAJOR or GrB_COLMAJOR
    Major major_type_;

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
    friend Info cusparse_spmm( DenseMatrix<c>&        C,
                               const Semiring&        op,
                               const SparseMatrix<a>& A,
                               const DenseMatrix<b>&  B );
    
    template <typename c, typename a, typename b>
    friend Info mergepath_spmm( DenseMatrix<c>&        C,
                                const Semiring&        op,
                                const SparseMatrix<a>& A,
                                const DenseMatrix<b>&  B );
  };

  template <typename T>
  Info DenseMatrix<T>::build( const std::vector<T>& values )
  {
    need_update_ = false;

    allocate();

    // Host copy
    for( graphblas::Index i=0; i<nvals_; i++ )
        h_denseVal[i] = values[i];

    // Device memcpy
    CUDA(cudaMemcpy(d_denseVal, h_denseVal, nvals_*sizeof(T),
        cudaMemcpyHostToDevice));

    //printArrayDevice( "B matrix GPU", d_denseVal );
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
    h_denseVal = (T*)malloc(nvals_*sizeof(T));
    for( Index i=0; i<nvals_; i++ )
      h_denseVal[i] = (T) 0;

    // Device alloc
    CUDA(cudaMalloc((void**)&d_denseVal, nvals_*sizeof(T)));
    CUDA(cudaMemcpy(d_denseVal, h_denseVal, nvals_*sizeof(T), 
        cudaMemcpyHostToDevice));

    return GrB_SUCCESS;
  }

  template <typename T>
  Info DenseMatrix<T>::clear()
  {
    if( h_denseVal ) free( h_denseVal );
    if( d_denseVal ) CUDA(cudaFree( d_denseVal ));
    return GrB_SUCCESS;
  }

  template <typename T>
  Info DenseMatrix<T>::setMajor( const Major major_type )
  {
    major_type_ = major_type;
    return GrB_SUCCESS;
  }

  template <typename T>
  Info DenseMatrix<T>::extractTuples( std::vector<T>& values )
  {
    values.clear();

    if( need_update_ )
      CUDA(cudaMemcpy(h_denseVal, d_denseVal, 
          nvals_*sizeof(T), cudaMemcpyDeviceToHost));
    
    for( Index i=0; i<nvals_; i++ ) {
      values.push_back( h_denseVal[i] );
    }
    return GrB_SUCCESS;
  }

  template <typename T>
  Info DenseMatrix<T>::print() const
  {
    if( need_update_ )
      CUDA(cudaMemcpy(h_denseVal, d_denseVal, 
          nvals_*sizeof(T), cudaMemcpyDeviceToHost));

    printArray( "denseVal", h_denseVal );
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
        if( major_type_ == GrB_ROWMAJOR ) {
          if( h_denseVal[row*ncols_+col]!=0.0 ) std::cout << "x ";
          else std::cout << "0 ";
        // Print column major order matrix in row major order (Transposition)
        } else if (major_type_ == GrB_COLMAJOR ) {
          if( h_denseVal[col*nrows_+row]!=0.0 ) std::cout << "x ";
          else std::cout << "0 ";
        }
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

#endif  // GRB_BACKEND_APSPIE_DENSEMATRIX_HPP
