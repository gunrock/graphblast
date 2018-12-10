#ifndef GRAPHBLAS_ALGORITHM_TEST_BFS_HPP_
#define GRAPHBLAS_ALGORITHM_TEST_BFS_HPP_

#include <deque>

namespace graphblas {
namespace algorithm {

// A simple CPU-based reference BFS ranking implementation
template <typename T>
int SimpleReferenceBfs(Index        nrows,
                       const Index* h_rowPtr,
                       const Index* h_colInd,
                       T*           source_path,
                       Index*       predecessor,
                       Index        src,
                       Index        stop) {
  // Initialize distances
  for (Index i = 0; i < nrows; ++i)
    source_path[i] = 0;
  source_path[src] = 1;
  Index search_depth = 1;

  // Initialize queue for managing previously-discovered nodes
  std::deque<Index> frontier;
  frontier.push_back(src);

  // Perform BFS
  CpuTimer cpu_timer;
  cpu_timer.Start();
  while (!frontier.empty()) {
    // Dequeue node from frontier
    Index dequeued_node = frontier.front();
    frontier.pop_front();
    Index neighbor_dist = source_path[dequeued_node] + 1;
    if (neighbor_dist > stop)
      break;

    // Locate adjacency list
    int edges_begin = h_rowPtr[dequeued_node];
    int edges_end = h_rowPtr[dequeued_node + 1];

    for (int edge = edges_begin; edge < edges_end; ++edge) {
      // Lookup neighbor and enqueue if undiscovered
      Index neighbor = h_colInd[edge];
      if (source_path[neighbor] == 0) {
        source_path[neighbor] = neighbor_dist;
        if (search_depth < neighbor_dist)
          search_depth = neighbor_dist;
        frontier.push_back(neighbor);
      }
    }
  }

  cpu_timer.Stop();
  float elapsed = cpu_timer.ElapsedMillis();

  printf("CPU BFS finished in %lf msec. Search depth is: %d\n", elapsed, search_depth);

  return search_depth;
}
}  // namespace algorithm
}  // namespace graphblas

#endif  // GRAPHBLAS_ALGORITHM_TEST_BFS_HPP_
