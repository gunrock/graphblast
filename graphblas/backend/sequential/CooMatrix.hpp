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
    CooMatrix();
    CooMatrix( Index num_rows, Index num_cols );

    private:
    Index num_row_;
    Index num_col_;
    Index num_nnz_;

    std::vector<Index> row_ind_;
    std::vector<Index> col_ind_;
    std::vector<T> values;
  };

  template <typename T>
  CooMatrix<T>::CooMatrix() : num_row_(0), num_col_(0) {}

  template <typename T>
  CooMatrix<T>::CooMatrix( Index num_row,
                           Index num_col ) : num_row_(num_row), num_col_(num_col) {}

} // backend
} // graphblas

#endif  // GRB_BACKEND_SEQUENTIAL_COOMATRIX_HPP
