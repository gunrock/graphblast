#ifndef GRB_BACKEND_SEQUENTIAL_COOMATRIX_HPP
#define GRB_BACKEND_SEQUENTIAL_COOMATRIX_HPP

#include <vector>

namespace graphblas
{
namespace backend
{
  template <typename T>
  class CooMatrix
  {
    public:
    CooMatrix() : nrows_(0), ncols_(0) {}
	CooMatrix( const Index nrows, const Index ncols ) : nrows_(nrows), ncols_(ncols) {}

	// C API Methods
	Info build( const std::vector<Index>& row_indices,
				const std::vector<Index>& col_indices,
				const std::vector<T>& values,
				const Index nvals,
				const CooMatrix& mask,
				const BinaryOp& dup );

    private:
    Index nrows_;
    Index ncols_;
    Index nvals_;

    std::vector<Index> row_indices_;
    std::vector<Index> col_indices_;
    std::vector<T> values_;
  };

  template <typename T>
  Info CooMatrix<T>::build( const std::vector<Index>& row_indices,
                            const std::vector<Index>& col_indices,
                            const std::vector<T>& values,
                            const Index nvals,
                            const CooMatrix& mask,
                            const BinaryOp& dup) {}

} // backend
} // graphblas

#endif  // GRB_BACKEND_SEQUENTIAL_COOMATRIX_HPP
