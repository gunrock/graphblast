#ifndef GRB_BACKEND_SEQUENTIAL_DENSEMATRIX_HPP
#define GRB_BACKEND_SEQUENTIAL_DENSEMATRIX_HPP

#include <vector>

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
        : nrows_(nrows), ncols_(ncols), nvals_(nrows*ncols) {}

	// C API Methods
	Info build( const std::vector<T>& values );

    private:
    Index nrows_;
    Index ncols_;
    Index nvals_;

    std::vector<T> values_;
  };

  template <typename T>
  Info DenseMatrix<T>::build( const std::vector<T>& values )
  {
      if( values.size() > nvals_ )
		  return GrB_INDEX_OUT_OF_BOUNDS;
	  values_ = values;
	  return GrB_SUCCESS;
  }

} // backend
} // graphblas

#endif  // GRB_BACKEND_SEQUENTIAL_COOMATRIX_HPP
