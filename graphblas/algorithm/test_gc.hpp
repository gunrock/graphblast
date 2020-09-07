#ifndef GRAPHBLAS_ALGORITHM_TEST_GC_HPP_
#define GRAPHBLAS_ALGORITHM_TEST_GC_HPP_

#include <vector>
#include <random>

namespace graphblas {
namespace algorithm {

// A simple CPU-based reference Graph Coloring implementation of the greedy
// First-Fit Implementation (FFI) algorithm.
// Returns number of colors and each node's color in h_gc_cpu starting from 1.
int SimpleReferenceGc(Index             nrows,
                      const Index*      h_csrRowPtr,
                      const Index*      h_csrColInd,
                      std::vector<int>* h_gc_cpu,
                      int               seed,
                      int               max_colors) {
  // Initialize color labels to 0 (uncolored).
  for (Index i = 0; i < nrows; ++i)
    (*h_gc_cpu)[i] = 0;

  // Initialize random number generator.
  std::mt19937 gen(seed);

  std::vector<Index> order(nrows);
  std::iota(order.begin(), order.end(), 0);
  std::shuffle(order.begin(), order.end(), gen);

  // Perform Graph Coloring.
  CpuTimer cpu_timer;
  cpu_timer.Start();

  for (Index i = 0; i < nrows; ++i) {
    Index row = order[i];
    std::vector<bool> min_array(max_colors, false);

    Index row_start = h_csrRowPtr[row];
    Index row_end   = h_csrRowPtr[row+1];
    for (; row_start < row_end; ++row_start) {
      Index col = h_csrColInd[row_start];
      int color = (*h_gc_cpu)[col];
      min_array[color] = true;
    }
    for (int color = 1; color < max_colors; ++color) {
      if (!min_array[color]) {
        (*h_gc_cpu)[row] = color;
        break;
      }
    }
  }

  cpu_timer.Stop();
  float elapsed = cpu_timer.ElapsedMillis();

  std::cout << "CPU GC finished in " << elapsed << " msec.";
}

int SimpleVerifyGc(Index                   nrows,
                   const Index*            h_csrRowPtr,
                   const Index*            h_csrColInd,
                   const std::vector<int>& h_gc_cpu,
                   bool                    suppress_zero) {
  int num_error = 0;
  int max_color = 0;

  for (Index row = 0; row < nrows; ++row) {
    int row_color = h_gc_cpu[row];
    if (row_color > max_color)
      max_color = row_color;

    if (row_color == 0 && num_error == 0 && !suppress_zero)
      std::cout << "\nINCORRECT: [" << row << "]: has no color.\n";

    Index row_start = h_csrRowPtr[row];
    Index row_end   = h_csrRowPtr[row+1];
    for (; row_start < row_end; ++row_start) {
      Index col = h_csrColInd[row_start];
      int col_color = h_gc_cpu[col];
      if (col_color == row_color) {
        if (num_error == 0) {
          std::cout << "\nINCORRECT: [" << row << "]: ";
          std::cout << row_color << " == " << col_color << " [" << col <<
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
  std::cout << "Graph coloring found with " << max_color << " colors.\n";
}
}  // namespace algorithm
}  // namespace graphblas

#endif  // GRAPHBLAS_ALGORITHM_TEST_GC_HPP_
