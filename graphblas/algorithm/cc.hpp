#ifndef GRAPHBLAS_ALGORITHM_CC_HPP_
#define GRAPHBLAS_ALGORITHM_CC_HPP_

#include <limits>
#include <vector>
#include <string>

#include "graphblas/algorithm/test_cc.hpp"
#include "graphblas/backend/cuda/util.hpp"

namespace graphblas {
namespace algorithm {

// mask = NULL, accumulator = GrB_MIN_UINT64, descriptor = NULL
Info reduce_assign(Vector<int>*              w,
                   Vector<int>*              src,
                   const std::vector<Index>* index,
                   Index                     num_elements) {
  Index nw = 0;
  Index ns = 0;
  CHECK(w->nvals(&nw));
  CHECK(src->nvals(&ns));
  std::vector<Index> ind(nw, 0);
  std::vector<int> w_val(nw, 0);
  std::vector<int> src_val(nw, 0);
  CHECK(w->extractTuples(&ind, &w_val, &nw));
  CHECK(src->extractTuples(&ind, &src_val, &ns));
  for (Index i = 0; i < num_elements; ++i) {
    if (src_val[i] < w_val[(*index)[i]]) {
      w_val[(*index)[i]] = src_val[i];
    }
  }
  CHECK(w->clear());
  CHECK(w->build(&ind, &w_val, nw, GrB_NULL));
  return GrB_SUCCESS;
}

//
// Code is based on the algorithm described in the following paper.
// Zhang, Azad, Hu. FastSV: FastSV: A Distributed-Memory Connected Component
// Algorithm with Fast Convergence (SIAM PP20).
//
// Modified by Tim Davis, Texas A&M University
//
// Code is ported from LAGraph: www.github.com/GraphBLAS/LAGraph
float cc(Vector<int>*       v,
         const Matrix<int>* A,
         int                seed,
         Descriptor*        desc) {
  Index A_nrows;
  CHECK(A->nrows(&A_nrows));

  // Difference vector.
  // mod in LAGraph.
  Vector<bool> diff(A_nrows);

  // Parent vector.
  // f in Zhang paper and LAGraph.
  Vector<int> parent(A_nrows);
  Vector<int> parent_temp(A_nrows);

  // Grandparent vector.
  // gf in Zhang paper.
  // gp in LAGraph.
  Vector<int> grandparent(A_nrows);
  Vector<int> grandparent_temp(A_nrows);

  // Min neighbor grandparent vector.
  // mngf in Zhang paper.
  // mngp in LAGraph.
  Vector<int> min_neighbor_parent(A_nrows);
  Vector<int> min_neighbor_parent_temp(A_nrows);

  // Initialize parent and min_neighbor_parent to:
  // [0]:0 [1]:1 [2]:2 [3]:3 [4]:4, etc.
  CHECK(parent.fillAscending(A_nrows));
  CHECK(min_neighbor_parent.dup(&parent));
  CHECK(min_neighbor_parent_temp.dup(&parent));
  CHECK(grandparent.dup(&parent));
  CHECK(grandparent_temp.dup(&parent));

  // Output vectors I and V in LAGraph respectively.
  std::vector<Index> index(A_nrows, 0);
  std::vector<Index> value(A_nrows, 0);
  Vector<Index> index_vec(A_nrows);
  Vector<Index> value_vec(A_nrows);
  CHECK(index_vec.fill(0));
  CHECK(value_vec.fill(0));

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
      std::cout << "=====CC Iteration " << iter - 1 << "=====\n";
      /*CHECK(v->print());
      CHECK(w.print());
      CHECK(f.print());
      CHECK(m.print());*/
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
    // Duplicate parent.
    CHECK(parent_temp.dup(&parent));

    // Hooking and shortcutting.
    // mngf[u] = A x gf
    //mxv<int, int, int, int>(&min_neighbor_parent, GrB_NULL, GrB_NULL,
    //    MinimumSelectSecondSemiring<int>(), A, &grandparent, desc);
    //CHECK(mask.clear());
    //eWiseMult<bool, bool, int, int>(&mask, GrB_NULL, GrB_NULL,
    //    CustomLessLessSemiring<int>(), &min_neighbor_parent, &parent, desc);
    //assign<int, bool, int, Index>(&hook_min_neighbor_parent, &mask, GrB_NULL,
    //    static_cast<int>(-1), GrB_ALL, A_nrows, desc);
    /*eWiseMult<int, bool, bool, int>(&parent_temp, GrB_NULL, GrB_NULL,
        PlusMultipliesSemiring<bool, int, int>(), &mask, &parent, desc);
    CHECK(desc->toggle(GrB_MASK));
    assign<int, bool, int, Index>(&parent_temp, &mask, GrB_NULL,
        static_cast<int>(-1), GrB_ALL, A_nrows, desc);
    CHECK(desc->toggle(GrB_MASK));
    CHECK(parent_temp.sparse2dense(-1));
    CHECK(parent_temp.dense2sparse(-1, desc));*/

    /*eWiseMult<int, bool, bool, int>(&hook_min_neighbor_parent, GrB_NULL,
        GrB_NULL, PlusMultipliesSemiring<bool, int, int>(), &mask,
        &min_neighbor_parent, desc);
    CHECK(desc->toggle(GrB_MASK));
    assign<int, bool, int, Index>(&hook_min_neighbor_parent, &mask, GrB_NULL,
        static_cast<int>(-1), GrB_ALL, A_nrows, desc);
    CHECK(desc->toggle(GrB_MASK));
    CHECK(hook_min_neighbor_parent.sparse2dense(-1));
    CHECK(hook_min_neighbor_parent.dense2sparse(-1, desc));
    CHECK(min_neighbor_parent.clear());
    CHECK(parent_temp.nvals(&num_hooks));*/

    // 1) Stochastic hooking.
    // mngf[u] = A x gf
    mxv<int, int, int, int>(&min_neighbor_parent_temp, GrB_NULL, GrB_NULL,
        MinimumSelectSecondSemiring<int>(), A, &grandparent, desc);
    eWiseAdd<int, bool, int, int>(&min_neighbor_parent, GrB_NULL, GrB_NULL,
        MinimumSelectSecondSemiring<int>(), &min_neighbor_parent,
        &min_neighbor_parent_temp, desc);
    // f[f[u]] = mngf[u]
    assignScatter<int, bool, int, int>(&parent, GrB_NULL, GrB_NULL,
        &min_neighbor_parent, &parent_temp, desc);

    // 2) Aggressive hooking.
    // f = min(f, mngf)
    eWiseAdd<int, bool, int, int>(&parent, GrB_NULL, GrB_NULL,
        MinimumPlusSemiring<int>(), &parent, &min_neighbor_parent, desc);

    // 3) Shortcutting.
    // f = min(f, gf)
    eWiseAdd<int, bool, int, int>(&parent, GrB_NULL, GrB_NULL,
        MinimumPlusSemiring<int>(), &parent, &parent_temp, desc);

    // 4) Calculate grandparents.
    // gf[u] = f[f[u]]
    extractGather<int, bool, int, int>(&grandparent, GrB_NULL, GrB_NULL,
        &parent, &parent, desc);

    // 5) Check termination.
    eWiseMult<bool, bool, int, int>(&diff, GrB_NULL, GrB_NULL,
        MinimumNotEqualToSemiring<int, int, bool>(), &grandparent_temp,
        &grandparent, desc);
    reduce<int, bool>(&succ, GrB_NULL, PlusMonoid<int>(), &diff, desc);
    CHECK(grandparent_temp.dup(&grandparent));

    if (succ == 0) {
      break;
    }
    iter++;
    if (desc->descriptor_.debug())
      std::cout << "succ: " << succ << " " << static_cast<int>(succ) <<
          std::endl;
    if (iter > desc->descriptor_.max_niter_)
      break;
  } while (succ > 0);

  // Copy result to output.
  CHECK(v->dup(&parent));
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
int ccCpu(Index             seed,
          Matrix<a>*        A,
          std::vector<int>* h_cc_cpu) {
  SimpleReferenceCc(A->matrix_.nrows_, A->matrix_.sparse_.h_csrRowPtr_,
      A->matrix_.sparse_.h_csrColInd_, h_cc_cpu, seed);
}

template <typename a>
int verifyCc(const Matrix<a>*        A,
             const std::vector<int>& h_cc_cpu,
             bool                    suppress_zero = false) {
  SimpleVerifyCc(A->matrix_.nrows_, A->matrix_.sparse_.h_csrRowPtr_,
      A->matrix_.sparse_.h_csrColInd_, h_cc_cpu, suppress_zero);
}
}  // namespace algorithm
}  // namespace graphblas

#endif  // GRAPHBLAS_ALGORITHM_CC_HPP_
