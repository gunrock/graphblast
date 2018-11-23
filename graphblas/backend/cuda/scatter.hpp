#ifndef GRB_BACKEND_CUDA_SCATTER_HPP
#define GRB_BACKEND_CUDA_SCATTER_HPP

#include <iostream>

namespace graphblas
{
namespace backend
{
  // Dense vector variant
  template <typename W, typename M, typename U, typename T>
  Info scatterDense( DenseVector<W>*       w,
                     const Vector<M>*      mask,
                     const DenseVector<U>* u,
                     T                     val,
                     Descriptor*           desc ) 
  {
    bool use_mask  = (mask != NULL);

    // Get descriptor parameters for nthreads
    Desc_value nt_mode;
    CHECK( desc->get(GrB_NT, &nt_mode) );
    const int nt = static_cast<int>(nt_mode);

    // Get number of elements
    Index u_nvals;
    u->nvals(&u_nvals);

    if (use_mask)
    {
      std::cout << "Error: Masked variant scatter not implemented yet!\n";
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

      scatterKernel<<<NB, NT>>>(w->d_val_, u_nvals, u->d_val_, u_nvals, val);
    }
    w->need_update_ = true;

    return GrB_SUCCESS;
  }

  // Sparse vector variant
  template <typename W, typename M, typename U, typename T>
  Info scatterSparse( DenseVector<W>*        w,
                      const Vector<M>*       mask,
                      const SparseVector<U>* u,
                      T                      val,
                      Descriptor*            desc )
  {
    bool use_mask  = (mask != NULL);

    // Get descriptor parameters for nthreads
    Desc_value nt_mode;
    CHECK( desc->get(GrB_NT, &nt_mode) );
    const int nt = static_cast<int>(nt_mode);

    // Get number of elements
    Index u_nvals, w_nvals;
    u->nvals(&u_nvals);
    w->nvals(&w_nvals);

    if (use_mask)
    {
      std::cout << "Error: Masked variant scatter not implemented yet!\n";
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

      scatterKernel<<<NB, NT>>>(w->d_val_, w_nvals, u->d_val_, u_nvals, val);
    }
    w->need_update_ = true;

    return GrB_SUCCESS;
  }
}  // backend
}  // graphblas

#endif  // GRB_BACKEND_CUDA_SCATTER_HPP
