#ifndef GRAPHBLAS_ALGORITHM_TEST_MIS_HPP_
#define GRAPHBLAS_ALGORITHM_TEST_MIS_HPP_

#include <vector>

namespace graphblas {
namespace algorithm {

// A simple CPU-based reference Maximal Independent Set implementation of the
// greedy algorithm
int SimpleReferenceMis(Index             nrows,
                       const Index*      h_csrRowPtr,
                       const Index*      h_csrColInd,
                       std::vector<int>* h_mis_cpu,
                       int               seed) {
  // initialize distances
  for (Index i = 0; i < nrows; ++i)
    (*h_mis_cpu)[i] = 0;

  // initialize random number generator
  std::mt19937 gen(seed);

  // initialize candidate list
  std::vector<bool> candidates(nrows, true);

  std::vector<Index> order(nrows);
  std::iota(order.begin(), order.end(), 0);
  std::shuffle(order.begin(), order.end(), gen);

  // perform Maximal Independent Set
  CpuTimer cpu_timer;
  cpu_timer.Start();

  for (Index i = 0; i < nrows; ++i) {
    Index row = order[i];

    Index row_start = h_csrRowPtr[row];
    Index row_end   = h_csrRowPtr[row+1];
    bool viable = candidates[row];
    if (viable) {
      (*h_mis_cpu)[row] = 1;
      candidates[row] = false;
      for (; row_start < row_end; ++row_start) {
        Index col = h_csrColInd[row_start];
        candidates[col] = false;
      }
    }
  }

  cpu_timer.Stop();
  float elapsed = cpu_timer.ElapsedMillis();

  std::cout << "CPU MIS finished in " << elapsed << " msec.";
}

/*!
 * Verification for Maximal Independent Set must check for 2 things:
 * 1) that the set is independent
 *   -i.e. no two vertices marked 1 are neighbours
 * 2) that the set is maximal
 *   -i.e. that the set and its neighbors form a cover
 */
int SimpleVerifyMis(Index                   nrows,
                    const Index*            h_csrRowPtr,
                    const Index*            h_csrColInd,
                    const std::vector<int>& h_mis_cpu) {
  int flag = 0;
  int set_size = 0;
  std::vector<bool> discovered(nrows, false);

  for (Index row = 0; row < nrows; ++row) {
    int row_color = h_mis_cpu[row];
    if (row_color == 1) {
      set_size++;
      discovered[row] = true;

      Index row_start = h_csrRowPtr[row];
      Index row_end   = h_csrRowPtr[row+1];
      for (; row_start < row_end; ++row_start) {
        Index col = h_csrColInd[row_start];
        int col_color = h_mis_cpu[col];
        if (col_color == row_color && flag == 0) {
          std::cout << "\nINCORRECT: [" << row << "]: ";
          std::cout << row_color << " == " << col_color << " [" << col <<
            "]\n";
        }

        if (col_color == row_color)
          flag++;

        discovered[col] = true;
      }
    }
  }
  for (Index row = 0; row < nrows; ++row) {
    if (!discovered[row] && flag == 0) {
      std::cout << "\nINCORRECT: [" << row << "]: ";
      std::cout << "Vertex not found!\n";
    }

    if (!discovered[row])
      flag++;
  }

  if (flag == 0)
    std::cout << "\nCORRECT\n";
  else
    std::cout << flag << " errors occurred.\n";
  std::cout << "Maximal independent set found with " << set_size << " elements.\n";
}
}  // namespace algorithm
}  // namespace graphblas

#endif  // GRAPHBLAS_ALGORITHM_TEST_MIS_HPP_
