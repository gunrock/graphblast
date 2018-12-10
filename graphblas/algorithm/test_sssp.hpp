#ifndef GRB_ALGORITHM_TESTSSSP_HPP
#define GRB_ALGORITHM_TESTSSSP_HPP

#include <queue>
#include <limits>
#include <utility>

namespace graphblas
{
namespace algorithm
{
  // A simple CPU-based reference SSSP ranking implementation
  template <typename T>
  int SimpleReferenceSssp( Index        nrows, 
                           const Index* h_rowPtr, 
                           const Index* h_colInd,
                           T*           h_val,
                           T*           source_path,
                           Index        src,
                           Index        stop)
  {
    typedef std::pair<T, Index> DistanceIndex;
    
    std::vector<bool> processed(nrows, false);

    //initialize distances
    for (Index i = 0; i < nrows; ++i)
    {
      source_path[i] = std::numeric_limits<T>::max();
    }
    source_path[src] = 0.f;
    Index search_depth = 0;

    // Initialize queue for managing previously-discovered nodes
    std::priority_queue<DistanceIndex, std::vector<DistanceIndex>, 
        std::greater<DistanceIndex>> frontier;
    frontier.push(std::make_pair<T, Index>(static_cast<T>(0.f), 
        static_cast<Index>(src)));

    //
    // Perform SSSP
    //

    CpuTimer cpu_timer;
    cpu_timer.Start();
    while (!frontier.empty())
    {
      Index frontier_size = frontier.size();
      for (int i = 0; i < frontier_size; ++i)
      {
        // Dequeue node from frontier
        DistanceIndex dequeued_node = frontier.top();
        T distance = dequeued_node.first;
        Index node = dequeued_node.second;
        frontier.pop();
        processed[node] = true;

        // Locate adjacency list
        int edges_begin = h_rowPtr[node];
        int edges_end   = h_rowPtr[node + 1];

        for (int edge = edges_begin; edge < edges_end; ++edge) 
        {
          // Lookup neighbor and enqueue if undiscovered
          Index neighbor = h_colInd[edge];
          T distance_to_neighbor = h_val[edge];
          if (!processed[neighbor] && 
              distance_to_neighbor != std::numeric_limits<T>::max()) 
          {
            T new_distance = distance + distance_to_neighbor;
            if (new_distance < source_path[neighbor])
            {
              source_path[neighbor] = new_distance;
              frontier.push(std::make_pair<T, Index>(
                  static_cast<T>(new_distance), static_cast<Index>(neighbor)));
            }
          }
        }
      }
      search_depth++;
    }

    cpu_timer.Stop();
    float elapsed = cpu_timer.ElapsedMillis();

    printArray("output", source_path, nrows);
    printf("CPU SSSP finished in %lf msec. Search depth is: %d\n", elapsed, search_depth);

    return search_depth;
  }

}  // namespace algorithm
}  // namespace graphblas

#endif  // GRB_ALGORITHM_TESTSSSP_HPP
