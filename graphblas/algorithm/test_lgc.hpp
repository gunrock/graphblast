#ifndef GRAPHBLAS_ALGORITHM_TEST_LGC_HPP_
#define GRAPHBLAS_ALGORITHM_TEST_LGC_HPP_

#include <queue>
#include <deque>
#include <utility>
#include <vector>

namespace graphblas {
namespace algorithm {

// A simple CPU-based reference LGC ranking implementation
template <typename T>
void SimpleReferenceLgc(Index        nrows,
                        const Index* h_rowPtrA,
                        const Index* h_colIndA,
                        const T*     h_val,
                        T*           pagerank,
                        Index        src,
                        double       alpha,
                        double       eps,
                        int          max_niter) {
  std::vector<T> residual(nrows, 0.f);
  std::vector<T> residual2(nrows, 0.f);
  std::vector<T> degrees(nrows, 0.f);

  // Initialize distances
  for (Index i = 0; i < nrows; ++i) {
    pagerank[i] = 0.f;
    degrees[i]  = h_rowPtrA[i+1] - h_rowPtrA[i];
  }
  residual[src] = 1.f;
  residual2[src] = 1.f;

  printf("alpha: %lf\n", alpha);
  printArray("degrees cpu", degrees, nrows);

  std::deque<Index> frontier;
  frontier.push_back(src);

  // Perform LGC
  CpuTimer cpu_timer;
  cpu_timer.Start();
  for (int i = 0; i < max_niter; ++i) {
    Index frontier_size = frontier.size();

    for (auto it = frontier.begin(); it != frontier.end(); it++) {
      Index v = *it;
      // p = p + alpha * r .* f
      pagerank[v] += alpha * residual[v];

      // r = (1 - alpha)*r/2
      residual2[v] = (1 - alpha)*residual[v]/2;
    }

    for (int j = 0; j < frontier_size; ++j) {
      Index v = frontier.front();
      frontier.pop_front();

      residual[v] = (1 - alpha)*residual[v]/2;

      // For each edge w such that (v,w) in E:
      //   r[w] = r[w] + (1 - alpha)*r[v]/2d[v]
      Index row_start = h_rowPtrA[v];
      Index row_end   = h_rowPtrA[v+1];
      for (; row_start < row_end; ++row_start) {
        Index w = h_colIndA[row_start];
        residual2[w] += residual[v]/degrees[v];
      }
    }
    residual = residual2;

    // f = {v : r[v] >= d[v] * eps}
    for (Index v = 0; v < residual.size(); ++v) {
      if (residual[v] >= degrees[v] * eps)
        frontier.push_back(v);
    }
    printArray("residual", residual, nrows);
  }

  cpu_timer.Stop();
  float elapsed = cpu_timer.ElapsedMillis();

  printf("CPU LGC finished in %lf msec\n", elapsed);
}

// A simple CPU-based reference LGC ranking implementation
template <typename T>
void SimpleReferenceLgcDense(Index        nrows,
                             const Index* h_rowPtrA,
                             const Index* h_colIndA,
                             const T*     h_val,
                             T*           pagerank,
                             Index        src,
                             double       alpha,
                             double       eps,
                             int          max_niter) {
  std::vector<T> residual(nrows, 0.f);
  std::vector<T> residual2(nrows, 0.f);
  std::vector<T> degrees(nrows, 0.f);

  // Initialize distances
  for (Index i = 0; i < nrows; ++i) {
    pagerank[i] = 0.f;
    degrees[i]  = h_rowPtrA[i+1] - h_rowPtrA[i];
  }
  residual[src] = 1.f;
  residual2[src] = 1.f;

  printf("alpha: %lf\n", alpha);
  printArray("degrees cpu", degrees, nrows);

  // Perform LGC
  CpuTimer cpu_timer;
  cpu_timer.Start();
  for (int i = 0; i < max_niter; ++i) {
    // f = {v : r[v] >= d[v] * eps}
    // p = p + alpha * r .* f
    /*for (int v = 0; v < nrows; ++v)
      if (residual[v] >= degrees[v] * eps)
        pagerank[v] += residual[v] * alpha;*/

    // For each edge w such that (v,w) in E:
    //   r[w] = r[w] + (1 - alpha)*r[v]/2d[v]
    for (int v = 0; v < nrows; ++v)
      if (residual[v] >= degrees[v] * eps)
        residual2[v] = (1 - alpha)*residual[v] / 2;

    for (int v = 0; v < nrows; ++v) {
      if (residual[v] >= degrees[v] * eps) {
        pagerank[v] += residual[v] * alpha;

        residual[v] = (1 - alpha)*residual[v] / 2;

        Index row_start = h_rowPtrA[v];
        Index row_end   = h_rowPtrA[v+1];
        for (; row_start < row_end; ++row_start) {
          Index w = h_colIndA[row_start];
          residual2[w] += residual[v] / degrees[v];
        }
      }
    }
    residual = residual2;
    // printArray("pagerank", pagerank, nrows);
    printArray("residual", residual, nrows);

    /*// r = (1 - alpha)*r/2
    for (int v = 0; v < nrows; ++v)
    {
      if (residual[v] >= degrees[v] * eps)
        residual[v] = (1 - alpha)*residual2[v] / 2;
    }*/
  }

  cpu_timer.Stop();
  float elapsed = cpu_timer.ElapsedMillis();

  printf("CPU LGC finished in %lf msec\n", elapsed);
}
}  // namespace algorithm
}  // namespace graphblas

#endif  // GRAPHBLAS_ALGORITHM_TEST_LGC_HPP_
