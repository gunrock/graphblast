#ifndef GRAPHBLAS_ALGORITHM_TEST_PR_HPP_
#define GRAPHBLAS_ALGORITHM_TEST_PR_HPP_

#include <queue>
#include <limits>
#include <utility>
#include <vector>
#include <functional>

namespace graphblas {
namespace algorithm {

// A simple CPU-based reference PR ranking implementation
template <typename T>
int SimpleReferencePr(Index        nrows,
                      const Index* h_rowPtr,
                      const Index* h_colInd,
                      T*           h_val,
                      T*           source_path,
                      float        alpha,
                      float        eps,
                      int          max_niter) {
  // Initialize distances
  for (Index i = 0; i < nrows; ++i)
    source_path[i] = 1.f/nrows;
  Index search_depth = 0;

  // Initialize out-degrees array
  std::vector<T> outdegrees(nrows, 0.f);
  for (Index i = 0; i < nrows; ++i)
    outdegrees[i] = h_rowPtr[i+1] - h_rowPtr[i];

  // Initialize pagerank
  std::vector<T> pagerank(nrows, 0.f);
  T resultant = 0.f;

  // Perform PR
  CpuTimer cpu_timer;
  cpu_timer.Start();
  for (int i = 0; i < max_niter; ++i) {
    for (Index node = 0; node < nrows; ++node)
      pagerank[node] = (1.f-alpha)/nrows;

    for (Index node = 0; node < nrows; ++node) {
      // Contribution
      T contrib = source_path[node]/outdegrees[node];

      // Locate adjacency list
      Index edges_begin = h_rowPtr[node];
      Index edges_end   = h_rowPtr[node + 1];

      for (Index edge = edges_begin; edge < edges_end; ++edge) {
        Index neighbor = h_colInd[edge];
        pagerank[neighbor] += alpha*contrib;
      }
    }
    //printArray("cpu_pagerank", pagerank, nrows);

    resultant = 0.f;
    for (Index node = 0; node < nrows; ++node) {
      T diff = source_path[node] - pagerank[node];
      resultant += diff*diff;
      source_path[node] = pagerank[node];
    }

    if (fabs(resultant) < eps)
      break;

    search_depth++;
  }

  cpu_timer.Stop();
  float elapsed = cpu_timer.ElapsedMillis();

  printArray("output", source_path, nrows);
  printf("CPU PR finished in %lf msec. Search depth is: %d. Resultant: %f\n", elapsed, search_depth, resultant);

  return search_depth;
}
}  // namespace algorithm
}  // namespace graphblas

#endif  // GRAPHBLAS_ALGORITHM_TEST_PR_HPP_
