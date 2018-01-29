#ifndef GRB_BACKEND_APSPIE_ASSIGN_HPP
#define GRB_BACKEND_APSPIE_ASSIGN_HPP

#include <iostream>

#include "graphblas/backend/apspie/Descriptor.hpp"
#include "graphblas/backend/apspie/SparseMatrix.hpp"
#include "graphblas/backend/apspie/DenseMatrix.hpp"
#include "graphblas/backend/apspie/operations.hpp"
#include "graphblas/backend/apspie/kernels/assignDense.hpp"
#include "graphblas/backend/apspie/kernels/assignSparse.hpp"

namespace graphblas
{
namespace backend
{

  template <typename W, typename T, typename M>
  Info assignDense( DenseVector<W>*           w,
                    const Vector<M>*          mask,
                    const BinaryOp<W,W,W>*    accum,
                    T                         val,
                    const std::vector<Index>* indices,
                    Index                     nindices,
                    Descriptor*               desc )
  {
    // Get descriptor parameters for SCMP, REPL, TRAN
    Desc_value scmp_mode, repl_mode;
    CHECK( desc->get(GrB_MASK, &scmp_mode) );
    CHECK( desc->get(GrB_OUTP, &repl_mode) );

    // TODO: add accum and replace support
    // -have masked variants as separate kernel
    // -accum and replace as parts in flow
    // -no need to copy indices from cpuToGpu if user selected all indices
    bool use_mask = (mask!=NULL);
    bool use_accum= (accum!=NULL);            //TODO
    bool use_all  = (indices==NULL);
    bool use_scmp = (scmp_mode==GrB_SCMP);
    bool use_repl = (repl_mode==GrB_REPLACE); //TODO

    if( desc->debug() )
    {
      std::cout << "Executing assignDense\n";
      printState( use_mask, use_accum, use_scmp, use_repl, false );
    }

    Index* indices_t = NULL;
    if( !use_all && nindices>0 )
    {
      desc->resize(nindices*sizeof(Index), "buffer");
      indices_t = (Index*) desc->d_buffer_;
      CUDA( cudaMemcpy(indices_t, indices, nindices*sizeof(Index), 
          cudaMemcpyHostToDevice) );
    }

    //printState( use_mask, use_accum, use_scmp, use_repl, use_tran );

    if( use_mask )
    {
      // Get descriptor parameters for nthreads
      Desc_value nt_mode;
      CHECK( desc->get(GrB_NT, &nt_mode) );
      const int nt = static_cast<int>(nt_mode);
      dim3 NT, NB;
			NT.x = nt;
			NT.y = 1;
			NT.z = 1;
			NB.x = (w->nvals_+nt-1)/nt;
			NB.y = 1;
			NB.z = 1;

      // Mask type
      // 1) Dense mask
      // 2) Sparse mask
      // 3) Uninitialized
      Storage mask_vec_type;
      CHECK( mask->getStorage(&mask_vec_type) );

      if( mask_vec_type==GrB_DENSE )
      {
        // TODO: must allow user to specify identity for dense mask vectors
        if( use_scmp )
          assignDenseDenseMaskedKernel<true, true ,true ><<<NB,NT>>>( w->d_val_,
              w->nvals_, (mask->dense_).d_val_, accum, (W)val, indices_t, 
							nindices );
        else
          assignDenseDenseMaskedKernel<false,true ,true ><<<NB,NT>>>( w->d_val_,
              w->nvals_, (mask->dense_).d_val_, accum, (W)val, indices_t, 
							nindices );
      }
      else if( mask_vec_type==GrB_SPARSE )
      {
        if( use_scmp )
          assignDenseSparseMaskedKernel<true, true ,true ><<<NB,NT>>>( 
              w->d_val_, w->nvals_, (mask->sparse_).d_ind_, 
              (mask->sparse_).nvals_, accum, (W)val, indices_t, nindices );
        else
          assignDenseSparseMaskedKernel<false,true ,true ><<<NB,NT>>>( 
              w->d_val_, w->nvals_, (mask->sparse_).d_ind_, 
              (mask->sparse_).nvals_, accum, (W)val, indices_t, nindices );

        if( desc->debug() )
        {
          printDevice("mask_ind", (mask->sparse_).d_ind_, mask->sparse_.nvals_);
        }
      }
      else
      {
        return GrB_UNINITIALIZED_OBJECT;
      }

      if( desc->debug() )
      {
        printDevice("mask_val", (mask->sparse_).d_val_, mask->sparse_.nvals_);
        printDevice("w_val", w->d_val_, w->nvals_);
      }
    }
    else
    {
      std::cout << "Unmasked DeVec Assign Constant\n";
      std::cout << "Error: Feature not implemented yet!\n";
    }
    w->need_update_ = true;
    return GrB_SUCCESS;
  }

  template <typename W, typename T, typename M>
  Info assignSparse( SparseVector<W>*          w,
                     const Vector<M>*          mask,
                     const BinaryOp<W,W,W>*    accum,
                     T                         val,
                     const std::vector<Index>* indices,
                     Index                     nindices,
                     Descriptor*               desc )
  {
    std::cout << "SpVec Assign Constant\n";
    std::cout << "Error: Feature not implemented yet!\n";
    return GrB_SUCCESS;
  }
}  // backend
}  // graphblas

#endif  // GRB_BACKEND_APSPIE_ASSIGN_HPP
