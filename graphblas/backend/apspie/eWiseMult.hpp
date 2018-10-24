#ifndef GRB_BACKEND_APSPIE_EWISEMULT_HPP
#define GRB_BACKEND_APSPIE_EWISEMULT_HPP

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
  Info eWiseMultInner( SparseVector<W>*       w,
                       const Vector<M>*       mask,
                       BinaryOpT              accum,
                       SemiringT              op,
                       const SparseVector<U>* u,
                       const SparseVector<V>* v,
                       Descriptor*            desc )
  {
    std::cout << "Error: eWiseMult sparse-sparse not implemented yet!\n";
    return GrB_SUCCESS;
  }

  // Dense x dense vector (no mask)
  template <typename W, typename U, typename V, typename M,
            typename BinaryOpT,     typename SemiringT>
  Info eWiseMultInner( DenseVector<W>*       w,
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
      std::cout << "Executing eWiseMult dense-dense\n";
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
      Storage mask_type;
      CHECK( mask->getStorage(&mask_type) );
      if (mask_type != GrB_DENSE)
        return GrB_INVALID_OBJECT;

      const DenseVector<M>* mask_dense = &mask->dense_;

      dim3 NT, NB;
      NT.x = nt;
      NT.y = 1;
      NT.z = 1;
      NB.x = (u_nvals + nt - 1) / nt;
      NB.y = 1;
      NB.z = 1;

      eWiseMultKernel<<<NB, NT>>>(w->d_val_, NULL, mask_dense->d_val_, 
          op.identity(), extractMul(op), u_t->d_val_, v_t->d_val_, u_nvals);
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

      eWiseMultKernel<<<NB, NT>>>(w->d_val_, NULL, op.identity(),
          extractMul(op), u_t->d_val_, v_t->d_val_, u_nvals);
    }
    w->need_update_ = true;

    return GrB_SUCCESS;
  }

  // Dense x dense vector (sparse mask)
  template <typename W, typename U, typename V, typename M,
            typename BinaryOpT,     typename SemiringT>
  Info eWiseMultInner( SparseVector<W>*       w,
                       const SparseVector<M>* mask,
                       BinaryOpT              accum,
                       SemiringT              op,
                       const DenseVector<U>*  u,
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
      std::cout << "Executing eWiseMult dense-dense (sparse mask)\n";
      printState( use_mask, use_accum, use_scmp, use_repl, 0 );
    }

    // Get descriptor parameters for nthreads
    Desc_value nt_mode;
    CHECK( desc->get(GrB_NT, &nt_mode) );
    const int nt = static_cast<int>(nt_mode);

    // Get number of elements
    Index u_nvals;
    u->nvals(&u_nvals);

    if( use_mask && desc->mask() )
    {
      Index mask_nvals;
      mask->nvals(&mask_nvals);

      dim3 NT, NB;
      NT.x = nt;
      NT.y = 1;
      NT.z = 1;
      NB.x = (mask_nvals + nt - 1) / nt;
      NB.y = 1;
      NB.z = 1;

      eWiseMultKernel<<<NB, NT>>>(w->d_ind_, w->d_val_, mask->d_ind_,
            mask->d_val_, mask_nvals, NULL, op.identity(), extractMul(op),
            u->d_val_, v->d_val_);

      w->nvals_ = mask_nvals;
    }
    else
    {
      std::cout << "Error: Unmasked eWiseMult dense-dense should not generate sparse vector output!\n";
    }
    w->need_update_ = true;

    return GrB_SUCCESS;
  }

  // Sparse x dense vector
  template <typename W, typename U, typename V, typename M,
            typename BinaryOpT,     typename SemiringT>
  Info eWiseMultInner( SparseVector<W>*       w,
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
      std::cout << "Executing eWiseMult sparse-dense\n";
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
      Storage mask_type;
      mask->getStorage(&mask_type);
      if (mask_type == GrB_DENSE)
        std::cout << "Error: Masked eWiseMult sparse-dense with dense mask not implemented yet!\n";
      else if (mask_type == GrB_SPARSE)
      {
        const SparseVector<M>* mask_sparse = &mask->sparse_;
        Index mask_nvals;
        mask_sparse->nvals(&mask_nvals);

        dim3 NT, NB;
        NT.x = nt;
        NT.y = 1;
        NT.z = 1;
        NB.x = (mask_nvals + nt - 1) / nt;
        NB.y = 1;
        NB.z = 1;

        eWiseMultKernel<<<NB, NT>>>(w->d_ind_, w->d_val_, mask_sparse->d_ind_,
            mask_sparse->d_val_, mask_nvals, NULL, op.identity(), 
            extractMul(op), u->d_ind_, u->d_val_, u_nvals, v->d_val_);

        // Mask size is upper bound on output memory allocation
        w->nvals_ = mask_nvals;
      }
    }
    else
    {
      Index* w_ind;
      W*     w_val;
      if (u == w)
      {
        CHECK( desc->resize(u_nvals*sizeof(Index) + u_nvals*sizeof(W), 
            "buffer") );
        w_ind = (Index*) desc->d_buffer_;
        w_val = (W*)     desc->d_buffer_+u_nvals;
      }
      else
      {
        w_ind = w->d_ind_;
        w_val = w->d_val_;
      }

      dim3 NT, NB;
      NT.x = nt;
      NT.y = 1;
      NT.z = 1;
      NB.x = (u_nvals + nt - 1) / nt;
      NB.y = 1;
      NB.z = 1;

      eWiseMultKernel<<<NB, NT>>>(w_ind, w_val, NULL, op.identity(), 
          extractMul(op), u->d_ind_, u->d_val_, u_nvals, v->d_val_);

      // u size is upper bound on output memory allocation
      w->nvals_ = u_nvals;
      if (u == w)
      {
        CUDA_CALL( cudaMemcpy(u->d_ind_, w_ind, u_nvals*sizeof(Index), 
            cudaMemcpyDeviceToDevice) );
        CUDA_CALL( cudaMemcpy(u->d_val_, w_val, u_nvals*sizeof(W), 
            cudaMemcpyDeviceToDevice) );
      }
    }
    w->need_update_ = true;
    return GrB_SUCCESS;
  }

}  // backend
}  // graphblas

#endif  // GrB_BACKEND_APSPIE_EWISEMULT_HPP
