#ifndef GRB_ALGORITHM_GC_HPP
#define GRB_ALGORITHM_GC_HPP

#include "graphblas/algorithm/testGc.hpp"
#include "graphblas/backend/cuda/util.hpp" // GpuTimer

namespace graphblas
{
  template <typename T_in1, typename T_out=T_in1>
  struct set_random
  {
    set_random()
    {
      seed_ = getEnv("GRB_SEED", 0);
      srand(seed_);
    }

    inline GRB_HOST_DEVICE T_out operator()(T_in1 lhs)
    { return rand(); }

    int seed_;
  };

namespace algorithm
{
  // Use float for now for both v and A
  float gc( Vector<int>*         v,
            const Matrix<float>* A,
            int                  seed,
            int                  max_colors,
            Descriptor*          desc )
  {
    Index A_nrows;
    CHECK( A->nrows(&A_nrows) );

    // Colors vector (v)
    // 0: no color, 1 ... n: color
    CHECK( v->fill(0) );

    // Frontier vector (f)
    Vector<float> f(A_nrows);

    // Weight vectors (w)
    Vector<float> w(A_nrows);
    CHECK( w.fill(0.f) );

    // Neighbor max (m)
    Vector<float> m(A_nrows);

    // Neighbor color (n)
    Vector<int> n(A_nrows);

    // Dense array (d)
    Vector<int> d(max_colors);

    // Ascending array (ascending)
    Vector<int> ascending(max_colors);
    CHECK( ascending.fillAscending(max_colors) );

    // Array for finding smallest color (min_array)
    Vector<int> min_array(max_colors);

    // Set seed
    setEnv("GRB_SEED", seed);

    desc->set(GrB_BACKEND, GrB_SEQUENTIAL);
    apply<float,float,float>(&w, GrB_NULL, GrB_NULL, set_random<float>(), &w, desc);
    CHECK( w.print() );
    desc->set(GrB_BACKEND, GrB_CUDA);

    float iter = 1;
    float succ = 0.f;
    int   min_color = 0;
    Index unvisited = A_nrows;
    backend::GpuTimer gpu_tight;

    if( desc->descriptor_.timing_>0 )
      gpu_tight.Start();
    do
    {
      if( desc->descriptor_.debug() )
      {
        std::cout << "=====Iteration " << iter - 1 << "=====\n";
        CHECK( v->print() );
        CHECK( w.print() );
        CHECK( f.print() );
        CHECK( m.print() );
      }
      if( desc->descriptor_.timing_==2 )
      {
        gpu_tight.Stop();
        if (iter > 1)
        {
          std::string vxm_mode = (desc->descriptor_.lastmxv_ == GrB_PUSHONLY) ?
              "push" : "pull";
          std::cout << iter - 1 << ", " << succ << "/" << A_nrows << ", "
              << unvisited << ", " << vxm_mode << ", "
              << gpu_tight.ElapsedMillis() << "\n";
        }
        unvisited -= (int)succ;
        gpu_tight.Start();
      }
      
      // find max of neighbors
      vxm<float, float, float, float>(&m, &w, GrB_NULL, 
          MaximumMultipliesSemiring<float>(), &w, A, desc);

      // find all largest nodes that are uncolored
      eWiseMult<float, float, float, float>(&f, GrB_NULL, GrB_NULL,
          PlusGreaterSemiring<float>(), &w, &m, desc);

      // stop when frontier is empty
      reduce<float, float>(&succ, GrB_NULL, PlusMonoid<float>(), &f, desc);

      if (succ == 0)
        break;

      // find neighbors of frontier
      vxm<float, int, float, float>(&m, v, GrB_NULL,
          LogicalOrAndSemiring<float>(), &f, A, desc);

      // get color
      eWiseMult<int, float, float, int>(&n, GrB_NULL, GrB_NULL,
          PlusMultipliesSemiring<float, int, int>(), &m, v, desc);

      // prepare dense array
      CHECK( d.fill(0) );

      // scatter nodes into a dense array
      scatter<int, float, int, int>(&d, GrB_NULL, &n, (int)max_colors, desc);

      // TODO(@ctcyang): this eWiseMult and reduce could be changed into single
      // reduce with argmin Monoid
      // map boolean bit array to element id
      eWiseMult<int, int, int, int>(&min_array, GrB_NULL, GrB_NULL,
          MinimumPlusSemiring<int>(), &d, &ascending, desc);
      CHECK( min_array.setElement(max_colors, 0) );

      // compute min color
      reduce<int, int>(&min_color, GrB_NULL, MinimumMonoid<int>(),
          &min_array, desc);

      // assign new color
      assign<int, float>(v, &f, GrB_NULL, min_color, GrB_ALL, A_nrows, desc);

      // get rid of colored nodes in candidate list
      assign<float, float>(&w, &f, GrB_NULL, (float)0.f, GrB_ALL, A_nrows,
          desc);

      iter++;
      if (desc->descriptor_.debug())
        std::cout << "succ: " << succ << " " << (int)succ << std::endl;
      if (iter > desc->descriptor_.max_niter_)
        break;
    } while (succ > 0);
    if( desc->descriptor_.timing_>0 )
    {
      gpu_tight.Stop();
      std::string vxm_mode = (desc->descriptor_.lastmxv_ == GrB_PUSHONLY) ?
          "push" : "pull";
      std::cout << iter - 1 << ", " << succ << "/" << A_nrows << ", "
          << unvisited << ", " << vxm_mode << ", "
          << gpu_tight.ElapsedMillis() << "\n";
      return gpu_tight.ElapsedMillis();
    }
    return 0.f;
  }

  template <typename T, typename a>
  int gcCpu( Index        source,
             Matrix<a>*   A,
             T*           h_bfs_cpu,
             Index        depth,
             bool         transpose=false )
  {
    Index* reference_check_preds = NULL;
    int max_depth;

    if( transpose )
      max_depth = SimpleReferenceGc<T>( A->matrix_.nrows_, 
          A->matrix_.sparse_.h_cscColPtr_, A->matrix_.sparse_.h_cscRowInd_, 
          h_bfs_cpu, reference_check_preds, source, depth);
    else
      max_depth = SimpleReferenceGc<T>( A->matrix_.nrows_, 
          A->matrix_.sparse_.h_csrRowPtr_, A->matrix_.sparse_.h_csrColInd_, 
          h_bfs_cpu, reference_check_preds, source, depth);

    return max_depth; 
  }

}  // algorithm
}  // graphblas

#endif  // GRB_ALGORITHM_GC_HPP
