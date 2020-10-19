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

// Code is based on the algorithm described in the following paper.
// Azad, Buluç. LACC: a linear-algebraic algorithm for finding connected
// components in distributed memory (IPDPS 2019).
//
// Code is ported from LAGraph: www.github.com/GraphBLAS/LAGraph
float cc(Vector<int>*       v,
         const Matrix<int>* A,
         int                seed,
         Descriptor*        desc) {
  Index A_nrows;
  CHECK(A->nrows(&A_nrows));

  // Star membership vector.
  Vector<bool> star(A_nrows);
  Vector<bool> mask(A_nrows);

  // f in Azad paper.
  // parents in LAGraph.
  Vector<int> parent(A_nrows);

  // gf in Azad paper.
  // gp in LAGraph.
  Vector<int> grandparent(A_nrows);

  // f_n in Azad paper.
  // mnp in LAGraph.
  Vector<int> min_neighbor_parent(A_nrows);

  // hookMNP in LAGraph.
  Vector<int> hook_min_neighbor_parent(A_nrows);

  // f_h in Azad paper.
  // hookP in LAGraph.
  Vector<int> hook_parent(A_nrows);

  // Temporary vectors.
  // tmp, pNonstars and nsgp in LAGraph respectively.
  Vector<int> temp(A_nrows);
  Vector<int> nonstar_parent(A_nrows);
  Vector<int> nonstar_grandparent(A_nrows);

  // Initialize parent and min_neighbor_parent to:
  // [0]:0 [1]:1 [2]:2 [3]:3 [4]:4, etc.
  CHECK(parent.fillAscending(A_nrows));
  CHECK(min_neighbor_parent.dup(&parent));
  CHECK(star.fill(true));

  // Output vectors I and V in LAGraph respectively.
  std::vector<Index> index(A_nrows, 0);
  std::vector<Index> value(A_nrows, 0);
  Vector<Index> index_vec(A_nrows);
  Vector<Index> value_vec(A_nrows);
  CHECK(index_vec.fill(0));
  CHECK(value_vec.fill(0));

  // semiring & monoid
  /*GrB_Monoid Min, Add;
  GrB_Semiring Sel2ndMin; // (Sel2nd,Min) semiring
  LAGRAPH_OK (GrB_Monoid_new (&Min, GrB_MIN_UINT64, (GrB_Index) UINT_MAX));
  LAGRAPH_OK (GrB_Monoid_new (&Add, GrB_PLUS_UINT64, (GrB_Index) 0));
  LAGRAPH_OK (GrB_Semiring_new (&Sel2ndMin, Min, GrB_SECOND_UINT64));*/

  // nHooks, nStars, nNonStars in LAGraph respectively.
  Index num_hooks = 0;
  Index num_stars = 0;
  Index num_nonstars = 0;

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

    // ---------------------------------------------------------
    // CondHook(A, parent, star);
    // ---------------------------------------------------------
    mxv<int, int, int, int>(&min_neighbor_parent, GrB_NULL, GrB_NULL,
        MinimumSelectSecondSemiring<int>(), A, &parent, desc);
    CHECK(mask.clear());
    eWiseMult<bool, bool, int, int>(&mask, &star, GrB_NULL,
        CustomLessLessSemiring<int>(), &min_neighbor_parent, &parent, desc);
    //assign<int, bool, int, Index>(&hook_min_neighbor_parent, &mask, GrB_NULL,
    //    static_cast<int>(-1), GrB_ALL, A_nrows, desc);
    eWiseMult<int, bool, bool, int>(&hook_parent, GrB_NULL, GrB_NULL,
        PlusMultipliesSemiring<bool, int, int>(), &mask, &parent, desc);
    CHECK(desc->toggle(GrB_MASK));
    assign<int, bool, int, Index>(&hook_parent, &mask, GrB_NULL,
        static_cast<int>(-1), GrB_ALL, A_nrows, desc);
    CHECK(desc->toggle(GrB_MASK));
    CHECK(hook_parent.sparse2dense(-1));
    CHECK(hook_parent.dense2sparse(-1, desc));

    eWiseMult<int, bool, bool, int>(&hook_min_neighbor_parent, GrB_NULL,
        GrB_NULL, PlusMultipliesSemiring<bool, int, int>(), &mask,
        &min_neighbor_parent, desc);
    CHECK(desc->toggle(GrB_MASK));
    assign<int, bool, int, Index>(&hook_min_neighbor_parent, &mask, GrB_NULL,
        static_cast<int>(-1), GrB_ALL, A_nrows, desc);
    CHECK(desc->toggle(GrB_MASK));
    CHECK(hook_min_neighbor_parent.sparse2dense(-1));
    CHECK(hook_min_neighbor_parent.dense2sparse(-1, desc));

    CHECK(min_neighbor_parent.clear());
    CHECK(hook_parent.nvals(&num_hooks));
    assignScatter<int, bool, int, int>(&parent, GrB_NULL, GrB_NULL,
        &hook_min_neighbor_parent, &hook_parent, num_hooks, desc);
    /*CHECK(hook_parent.extractTuples(&index, &value, &num_hooks));
    CHECK(temp.nnew(num_hooks));
    // TODO(ctcyang): Need to implement extract variant.
    std::cout << "num_hooks: " << num_hooks << std::endl;
    printArray("index", index, num_hooks);
    printArray("value", value, num_hooks);
    extract<int, int, int>(&temp, GrB_NULL, GrB_NULL, &hook_min_neighbor_parent,
        &index, num_hooks, desc);
    reduce_assign(&parent, &temp, &value, num_hooks);
    CHECK(temp.clear());*/

    // Modify the star vector.
    // TODO(ctcyang): Need to implement constant assign variant.
    CHECK(value_vec.build(&value, num_hooks));
    assign<bool, bool, bool>(&star, GrB_NULL, GrB_NULL, false, &value_vec,
        num_hooks, desc);
    // Extract modified parent.
    // TODO(ctcyang): Need to implement extract variant.
    extract<int, int, int>(&temp, GrB_NULL, GrB_NULL, &parent, &value,
        num_hooks, desc);
    CHECK(temp.extractTuples(&index, &value, &num_hooks));
    // TODO(ctcyang): Need to implement constant assign variant.
    CHECK(value_vec.build(&value, num_hooks));
    assign<bool, bool, bool>(&star, GrB_NULL, GrB_NULL, false, &value_vec,
        num_hooks, desc);
    CHECK(parent.extractTuples(&index, &value, &A_nrows));
    // TODO(ctcyang): Need to implement extract variant.
    extract<bool, bool>(&mask, GrB_NULL, GrB_NULL, &star, &value, A_nrows,
        desc);
    // TODO(ctcyang): Need to check assign variant with logical_and as the
    // accumulate function exists.
    assignIndexed<bool, bool>(&star, GrB_NULL, logical_and<bool>(), &mask,
        GrB_ALL, A_nrows, desc);
    // Clean up.
    CHECK(hook_min_neighbor_parent.clear());
    CHECK(hook_parent.clear());
    CHECK(temp.clear());
    // ---------------------------------------------------------
    // UnCondHook(A, parent, star);
    // ---------------------------------------------------------
    // TODO(ctcyang): Add vector assign indices variant
    assignIndexed<int, int>(&nonstar_parent, GrB_NULL, GrB_NULL, &parent,
        GrB_ALL, A_nrows, desc);
    assign<int, bool, int, Index>(&nonstar_parent, &star, GrB_NULL,
        static_cast<int>(A_nrows), GrB_ALL, A_nrows, desc);
    mxv<int, bool, int, int>(&hook_min_neighbor_parent, &star, GrB_NULL,
        MinimumSelectSecondSemiring<int>(), A, &nonstar_parent, desc);
    // Select the valid elements (i.e. less than A_nrows) of
    // hook_min_neighbor_parent.
    assign<int, bool, int, Index>(&nonstar_parent, GrB_NULL, GrB_NULL, A_nrows,
        GrB_ALL, A_nrows, desc);
    eWiseMult<bool, int, int, int>(&mask, GrB_NULL, GrB_NULL,
        PlusLessSemiring<int, int, bool>(), &hook_min_neighbor_parent,
        &nonstar_parent, desc);
    eWiseMult<int, bool, int, int>(&hook_parent, &mask, GrB_NULL,
        MinimumSelectSecondSemiring<int>(), &hook_min_neighbor_parent, &parent,
        desc);
    CHECK(hook_parent.nvals(&num_hooks));
    CHECK(hook_parent.extractTuples(&index, &value, &num_hooks));
    CHECK(temp.nnew(num_hooks));
    extract<int, int, int>(&temp, GrB_NULL, GrB_NULL, &hook_min_neighbor_parent,
        &index, num_hooks, desc);
    // !!
    CHECK(value_vec.build(&value, num_hooks));
    assign<int, int, int>(&parent, GrB_NULL, GrB_NULL,
        static_cast<int>(A_nrows), &value_vec, num_hooks, desc);
    reduce_assign(&parent, &temp, &value, num_hooks);

    // Modify the star vector.
    // TODO(ctcyang): Add vector assign constant variant without GrB_ALL.
    CHECK(value_vec.build(&value, num_hooks));
    assign<bool, bool>(&star, GrB_NULL, GrB_NULL, false, &value_vec, num_hooks,
        desc);
    CHECK(parent.extractTuples(&index, &value, &A_nrows));
    extract<bool, bool>(&mask, GrB_NULL, GrB_NULL, &star, &value, A_nrows,
        desc);
    // TODO(ctcyang): Need to check assign variant with logical_and as the
    // accumulate function exists.
    assignIndexed<bool, bool>(&star, GrB_NULL, logical_and<int>(), &mask,
        GrB_ALL, A_nrows, desc);

    // Check termination.
    reduce<int, bool>(&num_stars, GrB_NULL, PlusMonoid<int>(), &star, desc);
    if (num_stars == A_nrows) {
      break;
    }

    // Clean up.
    CHECK(hook_min_neighbor_parent.clear());
    CHECK(hook_parent.clear());
    CHECK(nonstar_parent.clear());
    CHECK(temp.clear());
    // ---------------------------------------------------------
    // Shortcut(parent);
    // ---------------------------------------------------------
    CHECK(parent.extractTuples(&index, &value, &A_nrows));
    extract<int, int, int>(&grandparent, GrB_NULL, GrB_NULL, &parent, &value,
        A_nrows, desc);
    assignIndexed<int, int>(&parent, GrB_NULL, GrB_NULL, &grandparent, GrB_ALL,
        A_nrows, desc);
    // ---------------------------------------------------------
    // StarCheck(parent, star);
    // ---------------------------------------------------------
    // Calculate grandparent.
    CHECK(parent.extractTuples(&index, &value, &A_nrows));
    extract<int, int, int>(&grandparent, GrB_NULL, GrB_NULL, &parent, &value,
        A_nrows, desc);
    
    // Identify vertices whose parent and grandparent are different.
    eWiseMult<bool, int, int, int>(&mask, GrB_NULL, GrB_NULL,
        PlusNotEqualToSemiring<int, int, bool>(), &grandparent, &parent, desc);
    CHECK(nonstar_grandparent.nnew(A_nrows));
    assignIndexed<int, bool, int>(&nonstar_grandparent, &mask, GrB_NULL,
        &grandparent, GrB_ALL, A_nrows, desc);
    
    // Extract indices and values for assign.
    CHECK(nonstar_grandparent.nvals(&num_nonstars));
    CHECK(nonstar_grandparent.extractTuples(&index, &value, &num_nonstars));
    CHECK(star.fill(true));
    CHECK(value_vec.build(&value, num_nonstars));
    CHECK(index_vec.build(&index, num_nonstars));
    assign<bool, int>(&star, GrB_NULL, GrB_NULL, static_cast<bool>(false),
        &index_vec, num_nonstars, desc);
    assign<bool, int>(&star, GrB_NULL, GrB_NULL, static_cast<bool>(false),
        &value_vec, num_nonstars, desc);

    // Extract indices and values for assign
    CHECK(parent.extractTuples(&index, &value, &A_nrows));
    extract<bool, bool>(&mask, GrB_NULL, GrB_NULL, &star, &value, A_nrows,
        desc);
    assignIndexed<bool, bool>(&star, GrB_NULL, logical_and<bool>(), &mask,
        GrB_ALL, A_nrows, desc);

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
