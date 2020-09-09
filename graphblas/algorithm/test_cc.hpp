#ifndef GRAPHBLAS_ALGORITHM_TEST_CC_HPP_
#define GRAPHBLAS_ALGORITHM_TEST_CC_HPP_

#include <vector>
#include <stack>

namespace graphblas {
namespace algorithm {

// Runs simple CPU-based reference Connected Components (CC) implementation.
// Returns number of components and the connected component label of each node
// in h_cc_cpu starting from 1.
int SimpleReferenceCc(Index             nrows,
                      const Index*      h_csrRowPtr,
                      const Index*      h_csrColInd,
                      std::vector<int>* h_cc_cpu,
                      int               seed) {
  // Initialize labels to 0 (unlabeled).
  for (Index i = 0; i < nrows; ++i)
    (*h_cc_cpu)[i] = 0;

  CpuTimer cpu_timer;
  cpu_timer.Start();

  int current_label = 0;
  std::stack<Index> work_stack;
  for (Index i = 0; i < nrows; ++i) {
    if ((*h_cc_cpu)[i] == 0) {
      current_label++;
    }
    work_stack.push(i);
    while (!work_stack.empty()) {
      Index current = work_stack.top();
      work_stack.pop();

      if ((*h_cc_cpu)[current] == 0) {
        (*h_cc_cpu)[current] = current_label;
        Index row_start = h_csrRowPtr[current];
        Index row_end   = h_csrRowPtr[current+1];
        for (; row_start < row_end; ++row_start) {
          Index col = h_csrColInd[row_start];
          int label = (*h_cc_cpu)[col];
          if (label == 0) {
            work_stack.push(col);
          }
        }
      }
    }
  }

  cpu_timer.Stop();
  float elapsed = cpu_timer.ElapsedMillis();

  std::cout << "CPU CC finished in " << elapsed << " msec.";
}

int SimpleVerifyCc(Index                   nrows,
                   const Index*            h_csrRowPtr,
                   const Index*            h_csrColInd,
                   const std::vector<int>& h_cc_cpu,
                   bool                    suppress_zero) {
  int num_error = 0;
  int max_label = 0;

  for (Index row = 0; row < nrows; ++row) {
    int row_label = h_cc_cpu[row];
    if (row_label > max_label)
      max_label = row_label;

    if (row_label == 0 && num_error == 0 && !suppress_zero)
      std::cout << "\nINCORRECT: [" << row << "]: has no component.\n";

    Index row_start = h_csrRowPtr[row];
    Index row_end   = h_csrRowPtr[row+1];
    for (; row_start < row_end; ++row_start) {
      Index col = h_csrColInd[row_start];
      int col_label = h_cc_cpu[col];
      if (col_label != row_label) {
        if (num_error == 0) {
          std::cout << "\nINCORRECT: [" << row << "]: ";
          std::cout << row_label << " != " << col_label << " [" << col <<
            "]\n";
        }
        num_error++;
      }
    }
  }
  if (num_error == 0)
    std::cout << "\nCORRECT\n";
  else
    std::cout << num_error << " errors occurred.\n";
  std::cout << "Connected components found with " << max_label;
  std::cout << " components.\n";
}
}  // namespace algorithm
}  // namespace graphblas

#endif  // GRAPHBLAS_ALGORITHM_TEST_CC_HPP_
