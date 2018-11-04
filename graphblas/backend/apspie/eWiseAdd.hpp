#ifndef GRB_BACKEND_APSPIE_EWISEADD_HPP
#define GRB_BACKEND_APSPIE_EWISEADD_HPP

#include <iostream>

#include "graphblas/backend/apspie/kernels/kernels.hpp"

namespace graphblas
{
namespace backend
{

  /*
   * \brief 4 vector variants
   */

  // Sparse x sparse vector
  template <typename W, typename U, typename V, typename M,
            typename BinaryOpT,     typename SemiringT>
  Info eWiseAddInner( DenseVector<W>*       w,
                      const Vector<M>*       mask,
                      BinaryOpT              accum,
                      SemiringT              op,
                      const SparseVector<U>* u,
                      const SparseVector<V>* v,
                      Descriptor*            desc )
  {
    std::cout << "Error: eWiseAdd sparse-sparse not implemented yet!\n";
    return GrB_SUCCESS;
  }

  // Dense x dense vector
  template <typename W, typename U, typename V, typename M,
            typename BinaryOpT,     typename SemiringT>
  Info eWiseAddInner( DenseVector<W>*       w,
                      const Vector<M>*      mask,
                      BinaryOpT             accum,
                      SemiringT             op,
                      const DenseVector<U>* u,
                      const DenseVector<V>* v,
                      Descriptor*           desc )
  {
    // Get descriptor parameters for SCMP, REPL
    Desc_value scmp_mode, repl_mode;
    CHECK( desc->get(GrB_MASK, &scmp_mode) );
    CHECK( desc->get(GrB_OUTP, &repl_mode) );

    std::string accum_type = typeid(accum).name();
    // TODO: add accum and replace support
    // -have masked variants as separate kernel
    // -have scmp as template parameter
    // -accum and replace as parts in flow
    bool use_mask  = (mask != NULL);
    bool use_accum = (accum_type.size() > 1);
    bool use_scmp  = (scmp_mode == GrB_SCMP);
    bool use_repl  = (repl_mode == GrB_REPLACE);

    if( desc->debug() )
    {
      std::cout << "Executing eWiseAdd dense-dense\n";
      printState( use_mask, use_accum, use_scmp, use_repl, 0 );
    }

    // Get descriptor parameters for nthreads
    Desc_value nt_mode;
    CHECK( desc->get(GrB_NT, &nt_mode) );
    const int nt = static_cast<int>(nt_mode);

    // Get number of elements
    Index u_nvals;
    u->nvals(&u_nvals);

    DenseVector<U>* u_t = const_cast<DenseVector<U>*>(u);
    DenseVector<V>* v_t = const_cast<DenseVector<V>*>(v);

    if( use_mask && desc->mask() )
    {
      std::cout << "Error: Masked eWiseAdd dense-dense not implemented yet!\n";
    }
    else
    {
      dim3 NT, NB;
      NT.x = nt;
      NT.y = 1;
      NT.z = 1;
      NB.x = (u_nvals + nt - 1) / nt;
      NB.y = 1;
      NB.z = 1;

      eWiseAddDenseDenseKernel<<<NB, NT>>>(w->d_val_, NULL, extractAdd(op),
          u_t->d_val_, v_t->d_val_, u_nvals);
    }
    w->need_update_ = true;

    return GrB_SUCCESS;
  }

  // Sparse x dense vector
  template <typename W, typename U, typename V, typename M,
            typename BinaryOpT,     typename SemiringT>
  Info eWiseAddInner( DenseVector<W>*       w,
                      const Vector<M>*       mask,
                      BinaryOpT              accum,
                      SemiringT              op,
                      const SparseVector<U>* u,
                      const DenseVector<V>*  v,
                      Descriptor*            desc )
  {
    // Get descriptor parameters for SCMP, REPL
    Desc_value scmp_mode, repl_mode;
    CHECK( desc->get(GrB_MASK, &scmp_mode) );
    CHECK( desc->get(GrB_OUTP, &repl_mode) );

    std::string accum_type = typeid(accum).name();
    // TODO: add accum and replace support
    // -have masked variants as separate kernel
    // -have scmp as template parameter
    // -accum and replace as parts in flow
    bool use_mask  = (mask != NULL);
    bool use_accum = (accum_type.size() > 1);
    bool use_scmp  = (scmp_mode == GrB_SCMP);
    bool use_repl  = (repl_mode == GrB_REPLACE);

    if( desc->debug() )
    {
      std::cout << "Executing eWiseAdd sparse-dense\n";
      printState( use_mask, use_accum, use_scmp, use_repl, 0 );
    }

    // Get descriptor parameters for nthreads
    Desc_value nt_mode;
    CHECK( desc->get(GrB_NT, &nt_mode) );
    const int nt = static_cast<int>(nt_mode);

    // Get number of elements
    Index u_nvals;
    u->nvals(&u_nvals);

    if (use_mask && desc->mask())
    {
      std::cout << "Error: Masked eWiseAdd sparse-dense not implemented yet!\n";
    }
    else
    {
      if (v != w)
        w->dup(v);

      dim3 NT, NB;
      NT.x = nt;
      NT.y = 1;
      NT.z = 1;
      NB.x = (u_nvals + nt - 1) / nt;
      NB.y = 1;
      NB.z = 1;

      eWiseAddSparseDenseKernel<<<NB, NT>>>(w->d_val_, NULL, extractAdd(op),
          u->d_ind_, u->d_val_, u_nvals);
    }
    w->need_update_ = true;
    return GrB_SUCCESS;
  }

}  // backend
}  // graphblas

#endif  // GrB_BACKEND_APSPIE_EWISEADD_HPP
