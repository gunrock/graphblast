#ifndef GRB_BACKEND_APSPIE_MATRIX_HPP
#define GRB_BACKEND_APSPIE_MATRIX_HPP

#include <vector>
#include <iostream>
#include <map>

namespace graphblas
{
namespace backend
{
  template <typename T>
  class SparseMatrix;

  template <typename T>
  class DenseMatrix;

  template <typename T>
  class Matrix
  {
    public:
    // Default Constructor, Standard Constructor 
    // Alternative to new in C++
    Matrix() : nrows_(0), ncols_(0), nvals_(0), sparse_(0,0), dense_(0,0),
               mat_type_(GrB_SPARSE) {}
    Matrix( Index nrows, Index ncols )
        : nrows_(nrows), ncols_(ncols), nvals_(0), sparse_(nrows,ncols), 
          dense_(nrows,ncols), mat_type_(GrB_SPARSE) {}

    // Default Destructor is good enough for this layer
    ~Matrix() {}

    // C API Methods

    // Mutators
    Info nnew(  Index nrows, Index ncols );
    Info dup(   const Matrix* rhs );
    Info clear();
    Info nrows( Index* nrows_t );
    Info ncols( Index* ncols_t );
    Info nvals( Index* nvals_t );
    template <typename BinaryOpT>
    Info build( const std::vector<Index>* row_indices,
                const std::vector<Index>* col_indices,
                const std::vector<T>*     values,
                Index                     nvals,
                BinaryOpT                 dup,
                const char*               fname );
    Info build( const char*           fname ); 
    Info build( const std::vector<T>* values, 
                Index                 nvals );
    Info build( Index* row_ptr,
                Index* col_ind,
                T*     values,
                Index  nvals );
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
                         Index* n );

    // Handy methods
    const T operator[]( Index ind );
    Info print( bool force_update );
    Info check();
    Info setNrows( Index nrows );
    Info setNcols( Index ncols );
    Info resize(   Index nrows,
                   Index ncols );
    Info setStorage( Storage  mat_type );
    Info getStorage( Storage* mat_type ) const;
    Info getFormat( SparseMatrixFormat* format) const;
    template <typename MatrixT>
    MatrixT* getMatrix() const;

    template <typename U>
    Info fill( Index axis,
               Index nvals,
               U     start );
    template <typename U>
    Info fillAscending( Index axis,
                        Index nvals,
                        U     start );

    private:
    Index nrows_;
    Index ncols_;
    Index nvals_;

    SparseMatrix<T> sparse_;
    DenseMatrix<T>  dense_;

    // Keeps track of whether matrix is Sparse or Dense
    Storage mat_type_;

  };

  // Transfer nrows ncols to Sparse/DenseMatrix data member
  template <typename T>
  Info Matrix<T>::nnew( Index nrows, Index ncols )
  {
    CHECK( sparse_.nnew(nrows, ncols) );
    CHECK(  dense_.nnew(nrows, ncols) );
    return GrB_SUCCESS;
  }

  template <typename T>
  Info Matrix<T>::dup( const Matrix* rhs )
  {
    mat_type_ = rhs->mat_type_;
    if( mat_type_ == GrB_SPARSE )
      return sparse_.dup(&rhs->sparse_);
    else if( mat_type_ == GrB_SPARSE )
      return dense_.dup(&rhs->dense_);
    std::cout << "Error: Failed to call dup!\n";
    return GrB_UNINITIALIZED_OBJECT;
  }

  template <typename T>
  Info Matrix<T>::clear() 
  {
    mat_type_ = GrB_UNKNOWN;
    nvals_    = 0;
    CHECK( sparse_.clear() );
    CHECK(  dense_.clear() );
    return GrB_SUCCESS;
  }

  template <typename T>
  inline Info Matrix<T>::nrows( Index* nrows_t )
  {
    Index nrows;
    if(      mat_type_ == GrB_SPARSE ) CHECK( sparse_.nrows(&nrows) );
    else if( mat_type_ == GrB_DENSE  ) CHECK(  dense_.nrows(&nrows) );
    else nrows = nrows_;

    // Update nrows_ with latest value
    nrows_   = nrows;
    *nrows_t = nrows;
    return GrB_SUCCESS;
  }

  template <typename T>
  inline Info Matrix<T>::ncols( Index* ncols_t )
  {
    Index ncols;
    if(      mat_type_ == GrB_SPARSE ) CHECK( sparse_.ncols(&ncols) );
    else if( mat_type_ == GrB_DENSE  ) CHECK(  dense_.ncols(&ncols) );
    else ncols = ncols_;

    // Update ncols_ with latest value
    ncols_   = ncols;
    *ncols_t = ncols;
    return GrB_SUCCESS;
  }

  template <typename T>
  inline Info Matrix<T>::nvals( Index* nvals_t ) 
  {
    Index nvals;
    if(      mat_type_ == GrB_SPARSE ) CHECK( sparse_.nvals(&nvals) );
    else if( mat_type_ == GrB_DENSE  ) CHECK(  dense_.nvals(&nvals) );
    else nvals = nvals_;

    // Update nvals_ with latest value
    nvals_   = nvals;
    *nvals_t = nvals;
    return GrB_SUCCESS;
  }

  // Option: Not const to allow sorting
  template <typename T>
  template <typename BinaryOpT>
  Info Matrix<T>::build( const std::vector<Index>* row_indices,
                         const std::vector<Index>* col_indices,
                         const std::vector<T>*     values,
                         Index                     nvals,
                         BinaryOpT                 dup,
                         const char*               fname )
  {
    mat_type_ = GrB_SPARSE;
    return sparse_.build( row_indices, col_indices, values, nvals, dup, fname );
  }

  template <typename T>
  Info Matrix<T>::build( const char* fname )
  {
    mat_type_ = GrB_SPARSE;
    return sparse_.build( fname );
  }

  template <typename T>
  Info Matrix<T>::build( const std::vector<T>* values, Index nvals )
  {
    mat_type_ = GrB_DENSE;
    return dense_.build( values, nvals );
  }

  template <typename T>
  Info Matrix<T>::build( Index* row_ptr,
                         Index* col_ind,
                         T*     values,
                         Index  nvals )
  {
    mat_type_ = GrB_SPARSE;
    return sparse_.build(row_ptr, col_ind, values, nvals);
  }

  template <typename T>
  Info Matrix<T>::setElement( Index row_index,
                              Index col_index )
  {
    if( mat_type_ == GrB_SPARSE ) 
      return sparse_.setElement( row_index, col_index );
    else if( mat_type_ == GrB_DENSE )
      return dense_.setElement( row_index, col_index );
    return GrB_UNINITIALIZED_OBJECT;
  }

  template <typename T>
  Info Matrix<T>::extractElement( T*    val,
                                  Index row_index,
                                  Index col_index )
  {
    if( mat_type_ == GrB_SPARSE ) 
      return sparse_.extractElement( val, row_index, col_index );
    else if( mat_type_ == GrB_DENSE ) 
      return dense_.extractElement( val, row_index, col_index );
    return GrB_UNINITIALIZED_OBJECT;
  }

  template <typename T>
  Info Matrix<T>::extractTuples( std::vector<Index>* row_indices,
                                 std::vector<Index>* col_indices,
                                 std::vector<T>*     values,
                                 Index*              n )
  {
    if( mat_type_ == GrB_SPARSE ) 
      return sparse_.extractTuples( row_indices, col_indices, values, n );
    else
      return GrB_UNINITIALIZED_OBJECT;
  }

  template <typename T>
  Info Matrix<T>::extractTuples( std::vector<T>* values, Index* n )
  {
    if( mat_type_ == GrB_DENSE ) 
      return dense_.extractTuples( values, n );
    else
      return GrB_UNINITIALIZED_OBJECT;
  }

  template <typename T>
  const T Matrix<T>::operator[]( Index ind )
  {
    if( mat_type_ == GrB_SPARSE )
      return sparse_[ind];
    else std::cout << "Error: operator[] not defined for dense matrices!\n";
    return 0.;
  }

  template <typename T>
  Info Matrix<T>::print( bool force_update )
  {
    if( mat_type_ == GrB_SPARSE ) return sparse_.print( force_update );
    else if( mat_type_ == GrB_DENSE ) return dense_.print( force_update );
    return GrB_UNINITIALIZED_OBJECT;
  }

  // Error checking function
  template <typename T>
  Info Matrix<T>::check()
  {
    if( mat_type_ == GrB_SPARSE )
      return sparse_.check();
    return GrB_UNINITIALIZED_OBJECT;
  }

  template <typename T>
  Info Matrix<T>::setNrows( Index nrows )
  {
    CHECK( sparse_.setNrows(nrows) );
    CHECK(  dense_.setNrows(nrows) );
    return GrB_SUCCESS;
  }

  template <typename T>
  Info Matrix<T>::setNcols( Index ncols )
  {
    CHECK( sparse_.setNcols(ncols) );
    CHECK(  dense_.setNcols(ncols) );
    return GrB_SUCCESS;
  }

  template <typename T>
  Info Matrix<T>::resize( Index nrows, Index ncols )
  {
    if( mat_type_ == GrB_SPARSE ) return sparse_.resize( nrows, ncols );
    return GrB_UNINITIALIZED_OBJECT;
  }

  // Private method that sets mat_type, clears and allocates
  template <typename T>
  Info Matrix<T>::setStorage( Storage mat_type )
  {
    mat_type_ = mat_type;
    if( mat_type_ == GrB_SPARSE ) {
      //CHECK( sparse_.clear() );
      CHECK( sparse_.allocate() );
    } else if( mat_type_ == GrB_DENSE ) {
      //CHECK( dense_.clear() );
      CHECK( dense_.allocate() );
    }
    return GrB_SUCCESS;
  }

  template <typename T>
  inline Info Matrix<T>::getStorage( Storage* mat_type ) const
  {
    *mat_type = mat_type_;
    return GrB_SUCCESS;
  }

  template <typename T>
  inline Info Matrix<T>::getFormat( SparseMatrixFormat* format ) const
  {
    if (mat_type_ == GrB_SPARSE) return sparse_.getFormat(format);
    else std::cout << 
        "Error: Sparse matrix format is not defined for dense matrix!\n";
    return GrB_SUCCESS;
  }

  template <typename T>
  template <typename MatrixT>
  MatrixT* Matrix<T>::getMatrix() const
  {
    if( mat_type_ == GrB_SPARSE )     return &sparse_;
    else if( mat_type_ == GrB_DENSE ) return &dense_;
    return NULL;
  }

  template <typename T>
  template <typename U>
  Info Matrix<T>::fill( Index axis, Index nvals, U start )
  {
    if( mat_type_ == GrB_SPARSE ) return sparse_.fill( axis, nvals, start );
    return GrB_UNINITIALIZED_OBJECT;
  }

  template <typename T>
  template <typename U>
  Info Matrix<T>::fillAscending( Index axis, 
                                 Index nvals, 
                                 U     start )
  {
    if( mat_type_ == GrB_SPARSE )
      return sparse_.fillAscending( axis, nvals, start );
    return GrB_UNINITIALIZED_OBJECT;
  }

}  // backend
}  // graphblas

#endif  // GRB_BACKEND_APSPIE_MATRIX_HPP
