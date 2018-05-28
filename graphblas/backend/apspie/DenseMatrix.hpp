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
        : nrows_(0), ncols_(0), nvals_(0), h_denseVal_(NULL), d_denseVal_(NULL),
          major_type_(GrB_ROWMAJOR), need_update_(false) {}
    DenseMatrix( const Index nrows, const Index ncols ) 
        : nrows_(nrows), ncols_(ncols), nvals_(nrows*ncols), h_denseVal_(NULL),
          d_denseVal_(NULL), major_type_(GrB_ROWMAJOR), need_update_(false) 
		{
			//allocate();
		}

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
    T* h_denseVal_;
    T* d_denseVal_;

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
  };

  template <typename T>
  Info DenseMatrix<T>::build( const std::vector<T>& values )
  {
    need_update_ = false;

    allocate();

    // Host copy
    //std::cout << values.size() << " " << nvals_ << std::endl;
    for( graphblas::Index i=0; i<nvals_; i++ )
        h_denseVal_[i] = values[i];

    // Device memcpy
    CUDA(cudaMemcpy(d_denseVal_, h_denseVal_, nvals_*sizeof(T),
        cudaMemcpyHostToDevice));

    return GrB_SUCCESS;
  }

  template <typename T>
  Info DenseMatrix<T>::nnew( const Index nrows, const Index ncols )
  {
    nrows_ = nrows;
    ncols_ = ncols;
    nvals_ = nrows_*ncols_;
		//allocate();
    return GrB_SUCCESS;
  }

  template <typename T>
  Info DenseMatrix<T>::allocate()
  {
    // Host alloc
    if( nvals_!=0 && h_denseVal_ == NULL )
      h_denseVal_ = (T*)malloc(nvals_*sizeof(T));

    for( Index i=0; i<nvals_; i++ )
      h_denseVal_[i] = (T) 0;

    if( nvals_!=0 && d_denseVal_ == NULL )
		{
			printMemory( "d_denseVal" );
      CUDA( cudaMalloc( &d_denseVal_, nvals_*sizeof(T) ) );
    }

    return GrB_SUCCESS;
  }

  template <typename T>
  Info DenseMatrix<T>::clear()
  {
    if( h_denseVal_ ) free( h_denseVal_ );
    if( d_denseVal_ ) CUDA(cudaFree( d_denseVal_ ));
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
      CUDA(cudaMemcpy(h_denseVal_, d_denseVal_, 
          nvals_*sizeof(T), cudaMemcpyDeviceToHost));
    
    for( Index i=0; i<nvals_; i++ ) {
      values.push_back( h_denseVal_[i] );
    }
    return GrB_SUCCESS;
  }

  template <typename T>
  Info DenseMatrix<T>::print() const
  {
    if( need_update_ )
      CUDA(cudaMemcpy(h_denseVal_, d_denseVal_, 
          nvals_*sizeof(T), cudaMemcpyDeviceToHost));

    printArray( "denseVal", h_denseVal_, nvals_ );
    printDense();
    return GrB_SUCCESS;
  }

  template <typename T>
  Info DenseMatrix<T>::printDense() const
  {
    int row_length=std::min(20,nrows_);
    int col_length=std::min(20,ncols_);

    for( int row=0; row<row_length; row++ ) {
      for( int col=0; col<col_length; col++ ) {
        // Print row major order matrix in row major order
        if( major_type_ == GrB_ROWMAJOR ) {
          if( h_denseVal_[row*ncols_+col]!=0.0 ) std::cout << "x ";
          else std::cout << "0 ";
        // Print column major order matrix in row major order (Transposition)
        } else if (major_type_ == GrB_COLMAJOR ) {
          if( h_denseVal_[col*nrows_+row]!=0.0 ) std::cout << "x ";
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
