#ifndef GRB_ALGORITHM_TESTLGC_HPP
#define GRB_ALGORITHM_TESTLGC_HPP

namespace graphblas
{
namespace algorithm
{
  // A simple CPU-based reference LGC ranking implementation
  template <typename T>
  void SimpleReferenceLgc( Index        nrows, 
                           const Index* h_rowPtrA, 
                           const Index* h_colIndA,
                           const T*     h_val,
                           T*           pagerank,
                           Index        src,
                           double       alpha,
                           double       eps,
                           int          max_niter )
  {
    std::vector<T> residual(nrows, 0.f);
    std::vector<T> residual2(nrows, 0.f);
    std::vector<T> degrees(nrows, 0.f);

    //initialize distances
    for (Index i = 0; i < nrows; ++i)
    {
      pagerank[i] = 0.f;
      degrees[i]  = h_rowPtrA[i+1] - h_rowPtrA[i];
    }
    residual[src] = 1.f;
    residual2[src] = 1.f;

    printf("alpha: %lf\n", alpha);
    printArray("degrees cpu", degrees, nrows);

    //
    // Perform LGC
    //

    CpuTimer cpu_timer;
    cpu_timer.Start();
    for (int i = 0; i < max_niter; ++i)
    {
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

      for (int v = 0; v < nrows; ++v)
      {
        if (residual[v] >= degrees[v] * eps)
        {
          pagerank[v] += residual[v] * alpha;

          residual[v] = (1 - alpha)*residual[v] / 2;

          Index row_start = h_rowPtrA[v];
          Index row_end   = h_rowPtrA[v+1];
          for (; row_start < row_end; ++row_start)
          {
            Index w = h_colIndA[row_start];
            residual2[w] += residual[v] / degrees[v];
          }
        }
      }
      residual = residual2;
      //printArray("pagerank", pagerank, nrows);
      //printArray("residual", residual, nrows);

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

#endif  // GRB_ALGORITHM_TESTLGC_HPP
