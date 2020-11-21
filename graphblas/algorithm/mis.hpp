#ifndef GRAPHBLAS_ALGORITHM_MIS_HPP_
#define GRAPHBLAS_ALGORITHM_MIS_HPP_

#include <string>
#include <vector>

#include "graphblas/algorithm/test_mis.hpp"
#include "graphblas/algorithm/common.hpp"
#include "graphblas/backend/cuda/util.hpp"

namespace graphblas {
namespace algorithm {

/*!
 * \brief Implementation of maximal independent set algorithm inner loop
 * \param v output vector which stores independent set as 1
 * \param w weight vector with candidates as nonzeroes
 * \param f temporary vector
 * \param m temporary vector
 * \param A adjacency matrix of graph
 * \param desc pointer to descriptor
 */
float misInner(Vector<int>*       v,
               Vector<int>*       w,
               Vector<int>*       f,
               Vector<int>*       m,
               const Matrix<int>* A,
               Descriptor*        desc) {
  Index A_nrows;
  CHECK(A->nrows(&A_nrows));

  // initialize v
  CHECK(v->fill(0));

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
      std::cout << "=====MIS Iteration " << iter - 1 << "=====\n";
      CHECK(v->print());
      CHECK(w->print());
      CHECK(f->print());
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
      gpu_tight.Start();
    }
    // find max of neighbors
    vxm<int, int, int, int>(m, w, GrB_NULL,
        MaximumMultipliesSemiring<int>(), w, A, desc);

    // find all largest nodes are candidates
    // eWiseMult<float, float, float, float>(&f, GrB_NULL, GrB_NULL,
    //     PlusGreaterSemiring<float>(), &w, &m, desc);
    eWiseAdd<int, int, int, int>(f, GrB_NULL, GrB_NULL,
        GreaterPlusSemiring<int>(), w, m, desc);

    // assign new members (frontier) to independent set v
    assign<int, int, int, Index>(v, f, GrB_NULL, static_cast<int>(1), GrB_ALL,
        A_nrows, desc);

    // get rid of new members in candidate list
    assign<int, int, int, Index>(w, f, GrB_NULL, static_cast<int>(0), GrB_ALL,
        A_nrows, desc);

    // check for stopping condition
    reduce<int, int>(&succ, GrB_NULL, PlusMonoid<int>(), f, desc);
    if (succ == 0)
        break;

    // remove neighbors of new members from candidates
    vxm<int, int, int, int>(m, w, GrB_NULL,
        LogicalOrAndSemiring<int>(), f, A,  desc);
    assign<int, int, int, Index>(w, m, GrB_NULL, static_cast<int>(0), GrB_ALL,
        A_nrows, desc);
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

// Implementation of maximal independent set algorithm
float mis(Vector<int>*       v,
          const Matrix<int>* A,
          int                seed,
          Descriptor*        desc) {
  Index A_nrows;
  CHECK(A->nrows(&A_nrows));

  // Colors vector (v)
  // 0: no color, 1 ... n: color
  CHECK(v->fill(0));

  // Weight vectors (w)
  Vector<int> w(A_nrows);
  CHECK(w.fill(0));

  // Frontier vector (f)
  Vector<int> f(A_nrows);

  // Neighbor max (m)
  Vector<int> m(A_nrows);

  // Set seed
  setEnv("GRB_SEED", seed);

  desc->set(GrB_BACKEND, GrB_SEQUENTIAL);
  apply<int, int, int>(&w, GrB_NULL, GrB_NULL, set_random<int>(), &w, desc);
  desc->set(GrB_BACKEND, GrB_CUDA);

  // find maximal independent set f of w on graph A
  float gpu_tight = misInner(v, &w, &f, &m, A, desc);

  if (desc->descriptor_.timing_ > 0)
    return gpu_tight;
  return 0.f;
}

template <typename a>
int misCpu(Index             seed,
           Matrix<a>*        A,
           std::vector<int>* h_gc_cpu) {
  SimpleReferenceMis(A->matrix_.nrows_, A->matrix_.sparse_.h_csrRowPtr_,
      A->matrix_.sparse_.h_csrColInd_, h_gc_cpu, seed);
}

template <typename a>
int verifyMis(const Matrix<a>*        A,
              const std::vector<int>& h_gc_cpu) {
  SimpleVerifyMis(A->matrix_.nrows_, A->matrix_.sparse_.h_csrRowPtr_,
      A->matrix_.sparse_.h_csrColInd_, h_gc_cpu);
}
}  // namespace algorithm
}  // namespace graphblas

#endif  // GRAPHBLAS_ALGORITHM_MIS_HPP_
