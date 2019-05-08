#ifndef GRAPHBLAS_ALGORITHM_TEST_TC_HPP_
#define GRAPHBLAS_ALGORITHM_TEST_TC_HPP_

#include <queue>
#include <limits>
#include <utility>
#include <vector>
#include <functional>

namespace graphblas {
namespace algorithm {

// Count intersection of two sorted lists
template <typename T>
inline void CountIntersection(const Index* h_colInd,
                              Index        list_begin1,
                              Index        list_end1,
                              Index        list_begin2,
                              Index        list_end2,
                              Index        node1,
                              Index        node2,
                              T*           ntris) {
  while (list_begin1 < list_end1 && list_begin2 < list_end2) {
    Index list1 = h_colInd[list_begin1];
    Index list2 = h_colInd[list_begin2];
    if (list1 < list2) {
      ++list_begin1;
    } else if (list1 > list2) {
      ++list_begin2;
    } else {
      //std::cout << node1 << " -> " << node2 << " -> " << list1 << std::endl;
      *ntris += 1;
      ++list_begin1;
      ++list_begin2;
    }
  }
}

// A simple CPU-based reference TC implementation
template <typename T>
int SimpleReferenceTc(Index        nrows,
                      const Index* h_rowPtr,
                      const Index* h_colInd,
                      T*           ntris) {
  // Initialize number of triangles
  *ntris = 0;

  // Perform TC
  CpuTimer cpu_timer;
  cpu_timer.Start();

  for (Index node = 0; node < nrows; ++node) {
    // Locate adjacency list
    Index edges_begin = h_rowPtr[node];
    Index edges_end   = h_rowPtr[node + 1];

    // Edge represents closing edge of the wedge from node -> neighbor
    //
    //    node -> neighbor
    //        \   /
    //          ?
    //
    // We need to count size of intersection between
    // A(node, :) and A^T(:, neighbor)
    //
    // Since A^T is transpose of A, we can use the single CSR data structure to
    // solve triangle count
    for (Index edge = edges_begin; edge < edges_end; ++edge) {
      Index neighbor = h_colInd[edge];
      Index start    = h_rowPtr[neighbor];
      Index end      = h_rowPtr[neighbor+1];
      CountIntersection(h_colInd, edges_begin, edges_end, start, end, node,
          neighbor, ntris);   
    }
  }

  cpu_timer.Stop();
  float elapsed = cpu_timer.ElapsedMillis();

  printf("CPU TC finished in %lf msec. Number of triangles: %d\n", elapsed,
      *ntris);

  return 0;
}
}  // namespace algorithm
}  // namespace graphblas

#endif  // GRAPHBLAS_ALGORITHM_TEST_TC_HPP_
