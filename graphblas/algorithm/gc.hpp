#ifndef GRAPHBLAS_ALGORITHM_GC_HPP_
#define GRAPHBLAS_ALGORITHM_GC_HPP_

#include <string>
#include <vector>

#include "graphblas/algorithm/test_gc.hpp"
#include "graphblas/algorithm/mis.hpp"
#include "graphblas/algorithm/common.hpp"
#include "graphblas/backend/cuda/util.hpp"

namespace graphblas {
namespace algorithm {

// cuSPARSE implementation
float gcCusparse(Vector<int>*         v,
                 const Matrix<float>* A,
                 int                  seed,
                 int                  max_colors,
                 Descriptor*          desc) {
  Index A_nrows;
  CHECK(A->nrows(&A_nrows));

  // Colors vector (v)
  // 0: no color, 1 ... n: color
  CHECK(v->fill(0));

  backend::GpuTimer gpu_tight;
  float gpu_tight_time = 0.f;

  if (desc->descriptor_.timing_ > 0)
    gpu_tight.Start();
  graphColor(v, A, desc);
  if (desc->descriptor_.timing_ > 0) {
    gpu_tight.Stop();
    gpu_tight_time += gpu_tight.ElapsedMillis();
    return gpu_tight_time;
  }
  return 0.f;  
}

// Implementation of Independent Set graph coloring algorithm
float gcIS(Vector<int>*       v,
           const Matrix<int>* A,
           int                seed,
           int                max_colors,
           Descriptor*        desc) {
  Index A_nrows;
  CHECK(A->nrows(&A_nrows));

  // Colors vector (v)
  // 0: no color, 1 ... n: color
  CHECK(v->fill(0));

  // Frontier vector (f)
  Vector<int> f(A_nrows);

  // Weight vectors (w)
  Vector<int> w(A_nrows);
  CHECK(w.fill(0));

  // Neighbor max (m)
  Vector<int> m(A_nrows);

  // Set seed
  setEnv("GRB_SEED", seed);

  CHECK(desc->set(GrB_BACKEND, GrB_SEQUENTIAL));
  apply<int, int, int>(&w, GrB_NULL, GrB_NULL, set_random<int>(), &w, desc);
  CHECK(desc->set(GrB_BACKEND, GrB_CUDA));

  int iter = 1;
  int succ = 0;
  int min_color = 0;
  Index unvisited = A_nrows;
  backend::GpuTimer gpu_tight;
  float gpu_tight_time = 0.f;

  if (desc->descriptor_.timing_ > 0)
    gpu_tight.Start();
  do {
    if (desc->descriptor_.debug()) {
      std::cout << "=====GC Iteration " << iter - 1 << "=====\n";
      CHECK(v->print());
      CHECK(w.print());
      CHECK(f.print());
      CHECK(m.print());
    }
    if (desc->descriptor_.timing_ == 2) {
      gpu_tight.Stop();
      if (iter > 1) {
        std::string vxm_mode = (desc->descriptor_.lastmxv_ == GrB_PUSHONLY) ?
            "push" : "pull";
        std::cout << iter - 1 << ", " << succ << "/" << A_nrows << ", "
            << unvisited << ", " << vxm_mode << ", "
            << gpu_tight.ElapsedMillis() << "\n";
        gpu_tight_time += gpu_tight.ElapsedMillis();
      }
      unvisited -= succ;
      gpu_tight.Start();
    }

    // find max of neighbors
    vxm<int, int, int, int>(&m, GrB_NULL, GrB_NULL,
        MaximumMultipliesSemiring<int>(), &w, A, desc);
    //vxm<int, int, int, int>(&m, &w, GrB_NULL,
    //    MaximumMultipliesSemiring<int>(), &w, A, desc);

    // find all largest nodes that are uncolored
    // eWiseMult<float, float, float, float>(&f, GrB_NULL, GrB_NULL,
    //     PlusGreaterSemiring<float>(), &w, &m, desc);
    eWiseAdd<int, int, int, int>(&f, GrB_NULL, GrB_NULL,
        GreaterPlusSemiring<int>(), &w, &m, desc);

    // stop when frontier is empty
    reduce<int, int>(&succ, GrB_NULL, PlusMonoid<int>(), &f, desc);

    if (succ == 0) {
      break;
    }

    // assign new color
    assign<int, int, int, Index>(v, &f, GrB_NULL, iter, GrB_ALL, A_nrows, desc);

    // get rid of colored nodes in candidate list
    assign<int, int, int, Index>(&w, &f, GrB_NULL, static_cast<int>(0), GrB_ALL,
        A_nrows, desc);

    iter++;
    if (desc->descriptor_.debug())
      std::cout << "succ: " << succ << " " << static_cast<int>(succ) <<
          std::endl;
    if (iter > desc->descriptor_.max_niter_)
      break;
  } while (succ > 0);
  if (desc->descriptor_.timing_ > 0) {
    gpu_tight.Stop();
    std::string vxm_mode = (desc->descriptor_.lastmxv_ == GrB_PUSHONLY) ?
        "push" : "pull";
    if (desc->descriptor_.timing_ == 2)
      std::cout << iter << ", " << succ << "/" << A_nrows << ", "
          << unvisited << ", " << vxm_mode << ", "
          << gpu_tight.ElapsedMillis() << "\n";
    gpu_tight_time += gpu_tight.ElapsedMillis();
    return gpu_tight_time;
  }
  return 0.f;
}

// Implementation of Maximal Independent Set graph coloring algorithm
float gcMIS(Vector<int>*       v,
            const Matrix<int>* A,
            int                seed,
            int                max_colors,
            Descriptor*        desc) {
  Index A_nrows;
  CHECK(A->nrows(&A_nrows));

  // Colors vector (v)
  // 0: no color, 1 ... n: color
  CHECK(v->fill(0));

  // Frontier vector (f)
  Vector<int> f(A_nrows);

  // Weight vectors (w)
  Vector<int> w(A_nrows);
  CHECK(w.fill(0));

  // Temporary weight vector (w)
  Vector<int> temp_w(A_nrows);

  // Buffer vector (m)
  Vector<int> m(A_nrows);

  // Buffer vector (n)
  Vector<int> n(A_nrows);

  // Set seed
  setEnv("GRB_SEED", seed);

  desc->set(GrB_BACKEND, GrB_SEQUENTIAL);
  apply<int, int, int>(&w, GrB_NULL, GrB_NULL, set_random<int>(), &w, desc);
  desc->set(GrB_BACKEND, GrB_CUDA);

  int iter = 1;
  int succ = 0;
  int min_color = 0;
  Index unvisited = A_nrows;
  backend::GpuTimer gpu_tight;
  float gpu_tight_time = 0.f;

  if (desc->descriptor_.timing_ > 0)
    gpu_tight.Start();
  do {
    if (desc->descriptor_.debug()) {
      std::cout << "=====GC Iteration " << iter - 1 << "=====\n";
      CHECK(v->print());
      CHECK(w.print());
      CHECK(f.print());
      CHECK(m.print());
    }
    if (desc->descriptor_.timing_ == 2) {
      gpu_tight.Stop();
      if (iter > 1) {
        std::string vxm_mode = (desc->descriptor_.lastmxv_ == GrB_PUSHONLY) ?
            "push" : "pull";
        std::cout << iter - 1 << ", " << succ << "/" << A_nrows << ", "
            << unvisited << ", " << vxm_mode << ", "
            << gpu_tight.ElapsedMillis() << "\n";
        gpu_tight_time += gpu_tight.ElapsedMillis();
      }
      unvisited -= succ;
      gpu_tight.Start();
    }
    CHECK(temp_w.dup(&w));

    // find maximal independent set f of w on graph A
    misInner(&f, &temp_w, &n, &m, A, desc);

    // stop when frontier is empty
    reduce<int, int>(&succ, GrB_NULL, PlusMonoid<int>(), &f, desc);

    if (succ == 0) {
      break;
    }

    // assign new color
    assign<int, int, int, Index>(v, &f, GrB_NULL, iter, GrB_ALL, A_nrows, desc);

    // get rid of colored nodes in candidate list
    assign<int, int, int, Index>(&w, &f, GrB_NULL, static_cast<int>(0), GrB_ALL,
        A_nrows, desc);

    iter++;
    if (desc->descriptor_.debug())
      std::cout << "succ: " << succ << " " << static_cast<int>(succ) <<
          std::endl;
    if (iter > desc->descriptor_.max_niter_)
      break;
  } while (succ > 0);
  if (desc->descriptor_.timing_ > 0) {
    gpu_tight.Stop();
    std::string vxm_mode = (desc->descriptor_.lastmxv_ == GrB_PUSHONLY) ?
        "push" : "pull";
    if (desc->descriptor_.timing_ == 2)
      std::cout << iter << ", " << succ << "/" << A_nrows << ", "
          << unvisited << ", " << vxm_mode << ", "
          << gpu_tight.ElapsedMillis() << "\n";
    gpu_tight_time += gpu_tight.ElapsedMillis();
    return gpu_tight_time;
  }
  return 0.f;
}

// Implementation of Jones-Plassman graph coloring algorithm
float gcJP(Vector<int>*       v,
           const Matrix<int>* A,
           int                seed,
           int                max_colors,
           Descriptor*        desc) {
  Index A_nrows;
  CHECK(A->nrows(&A_nrows));

  // Colors vector (v)
  // 0: no color, 1 ... n: color
  CHECK(v->fill(0));

  // Frontier vector (f)
  Vector<int> f(A_nrows);

  // Weight vectors (w)
  Vector<int> w(A_nrows);
  CHECK(w.fill(0));

  // Temp weight vector (temp_w)
  Vector<int> temp_w(A_nrows);
  CHECK(temp_w.fill(0));
  Vector<int> temp_w2(A_nrows);
  CHECK(temp_w2.fill(0));

  // Neighbor max (m)
  Vector<int> m(A_nrows);

  // Neighbor color (n)
  Vector<int> n(A_nrows);

  // Dense array (d)
  Vector<int> d(max_colors);

  // Ascending array (ascending)
  Vector<int> ascending(max_colors);
  CHECK(ascending.fillAscending(max_colors));

  // Array for finding smallest color (min_array)
  Vector<int> min_array(max_colors);

  // Set seed
  setEnv("GRB_SEED", seed);

  desc->set(GrB_BACKEND, GrB_SEQUENTIAL);
  apply<int, int, int>(&w, GrB_NULL, GrB_NULL, set_random<int>(), &w, desc);
  desc->set(GrB_BACKEND, GrB_CUDA);

  int iter = 1;
  int succ = 0;
  int min_color = 0;
  Index unvisited = A_nrows;
  backend::GpuTimer gpu_tight;
  float gpu_tight_time = 0.f;

  if (desc->descriptor_.timing_ > 0)
    gpu_tight.Start();
  do {
    if (desc->descriptor_.debug()) {
      std::cout << "=====GC Iteration " << iter - 1 << "=====\n";
      CHECK(v->print());
      CHECK(w.print());
      CHECK(f.print());
      CHECK(m.print());
    }
    if (desc->descriptor_.timing_ == 2) {
      gpu_tight.Stop();
      if (iter > 1) {
        std::string vxm_mode = (desc->descriptor_.lastmxv_ == GrB_PUSHONLY) ?
            "push" : "pull";
        std::cout << iter - 1 << ", " << succ << "/" << A_nrows << ", "
            << unvisited << ", " << vxm_mode << ", "
            << gpu_tight.ElapsedMillis() << "\n";
        gpu_tight_time += gpu_tight.ElapsedMillis();
      }
      unvisited -= succ;
      gpu_tight.Start();
    }

    // find max of neighbors
    vxm<int, int, int, int>(&m, &w, GrB_NULL,
        MaximumMultipliesSemiring<int>(), &w, A, desc);

    // find all largest nodes that are uncolored
    // eWiseMult<float, float, float, float>(&f, GrB_NULL, GrB_NULL,
    //     PlusGreaterSemiring<float>(), &w, &m, desc);
    eWiseAdd<int, int, int, int>(&f, GrB_NULL, GrB_NULL,
        GreaterPlusSemiring<int>(), &w, &m, desc);

    // stop when frontier is empty
    reduce<int, int>(&succ, GrB_NULL, PlusMonoid<int>(), &f, desc);

    if (succ == 0) {
      break;
    }

    // find neighbors of frontier
    vxm<int, int, int, int>(&m, v, GrB_NULL,
        LogicalOrAndSemiring<int>(), &f, A, desc);

    // get color
    eWiseMult<int, int, int, int>(&n, GrB_NULL, GrB_NULL,
        PlusMultipliesSemiring<int, int, int>(), &m, v, desc);

    // prepare dense array
    CHECK(d.fill(0));

    // scatter nodes into a dense array
    scatter<int, int, int, int>(&d, GrB_NULL, &n, static_cast<int>(max_colors),
        desc);

    // TODO(@ctcyang): this eWiseMult and reduce could be changed into single
    // reduce with argmin Monoid
    // map boolean bit array to element id
    eWiseMult<int, int, int, int>(&min_array, GrB_NULL, GrB_NULL,
        MinimumPlusSemiring<int>(), &d, &ascending, desc);
    CHECK(min_array.setElement(max_colors, 0));

    // compute min color
    reduce<int, int>(&min_color, GrB_NULL, MinimumMonoid<int>(),
        &min_array, desc);

    // assign new color
    assign<int, int, int, Index>(v, &f, GrB_NULL, min_color, GrB_ALL, A_nrows,
        desc);

    // get rid of colored nodes in candidate list
    assign<int, int, int, Index>(&w, &f, GrB_NULL, static_cast<int>(0), GrB_ALL,
        A_nrows, desc);

    iter++;
    if (desc->descriptor_.debug())
      std::cout << "succ: " << succ << " " << static_cast<int>(succ) <<
          std::endl;
    if (iter > desc->descriptor_.max_niter_)
      break;
  } while (succ > 0);
  if (desc->descriptor_.timing_ > 0) {
    gpu_tight.Stop();
    std::string vxm_mode = (desc->descriptor_.lastmxv_ == GrB_PUSHONLY) ?
        "push" : "pull";
    if (desc->descriptor_.timing_ == 2)
      std::cout << iter << ", " << succ << "/" << A_nrows << ", "
          << unvisited << ", " << vxm_mode << ", "
          << gpu_tight.ElapsedMillis() << "\n";
    gpu_tight_time += gpu_tight.ElapsedMillis();
    return gpu_tight_time;
  }
  return 0.f;
}

template <typename a>
int gcCpu(Index             seed,
          Matrix<a>*        A,
          std::vector<int>* h_gc_cpu,
          int               max_colors) {
  SimpleReferenceGc(A->matrix_.nrows_, A->matrix_.sparse_.h_csrRowPtr_,
      A->matrix_.sparse_.h_csrColInd_, h_gc_cpu, seed, max_colors);
}

template <typename a>
int verifyGc(const Matrix<a>*        A,
             const std::vector<int>& h_gc_cpu,
             bool                    suppress_zero = false) {
  SimpleVerifyGc(A->matrix_.nrows_, A->matrix_.sparse_.h_csrRowPtr_,
      A->matrix_.sparse_.h_csrColInd_, h_gc_cpu, suppress_zero);
}
}  // namespace algorithm
}  // namespace graphblas

#endif  // GRAPHBLAS_ALGORITHM_GC_HPP_
