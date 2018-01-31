#ifndef GRB_BACKEND_APSPIE_SPMSPVINNER_HPP
#define GRB_BACKEND_APSPIE_SPMSPVINNER_HPP

#include <iostream>

#include <cuda.h>
#include <cusparse.h>

#include <moderngpu.cuh>
#include <cub.cuh>

#include "graphblas/backend/apspie/kernels/util.hpp"

namespace graphblas
{
namespace backend
{
  // Memory requirements: (4|V|+5|E|)*desc->memusage()
  //   -desc->memusage() is defined in graphblas/types.hpp
  //
  //  -> d_csrColBad    |V|*desc->memusage()
  //  -> d_csrColGood   |V|*desc->memusage()
  //  -> d_csrColDiff   |V|*desc->memusage()
  //  -> d_index        |V|*desc->memusage()
  //  -> d_csrVecInd    |E|*desc->memusage() (u_ind)
  //  -> d_csrSwapInd   |E|*desc->memusage()
  //  -> d_csrVecVal    |E|*desc->memusage()
  //  -> d_csrTempVal   |E|*desc->memusage() (u_val)
  //  -> d_csrSwapVal   |E|*desc->memusage()
  //  -> w_ind          |E|*desc->memusage()
  //  -> w_val          |E|*desc->memusage()
  //  -> d_temp_storage runtime constant
  template <typename W, typename a, typename U,
            typename BinaryOpT, typename SemiringT>
  Info spmspvApspie( Index*            w_ind,
                     W*                w_val,
                     Index*            w_nvals,
                     BinaryOpT         accum,
                     SemiringT         op,
                     Index             A_nrows,
                     Index             A_nvals,
                     const Index*      A_csrRowPtr,
                     const Index*      A_csrColInd,
                     const a*          A_csrVal,
                     const Index*      u_ind,
                     const U*          u_val,
                     const Index*      u_nvals,
                     Descriptor*       desc )
  {
    return GrB_SUCCESS;
  }

  // Memory requirements: 2|E|*desc->memusage()
  //   -desc->memusage() is defined in graphblas/types.hpp
  // 
  //  -> d_csrSwapInd   |E|*desc->memusage() [2*A_nrows: 1*|E|*desc->memusage()]
  //  -> d_csrSwapVal   |E|*desc->memusage() [2*A_nrows+ 2*|E|*desc->memusage()]
  //  -> d_temp_storage runtime constant
  //
  // TODO: can lower 2|E| * desc->memusage() memory requirement further by doing
  //       external memory sorting
  template <typename W, typename a, typename U,
            typename BinaryOpT, typename SemiringT>
  Info spmspvApspieLB( Index*            w_ind,
                       W*                w_val,
                       Index*            w_nvals,
                       BinaryOpT         accum,
                       SemiringT         op,
                       Index             A_nrows,
                       Index             A_nvals,
                       const Index*      A_csrRowPtr,
                       const Index*      A_csrColInd,
                       const a*          A_csrVal,
                       const Index*      u_ind,
                       const U*          u_val,
                       const Index*      u_nvals,
                       Descriptor*       desc )
  {
    // Get descriptor parameters for nthreads
    Desc_value ta_mode, tb_mode, nt_mode;
    CHECK( desc->get(GrB_TA, &ta_mode) );
    CHECK( desc->get(GrB_TB, &tb_mode) );
    CHECK( desc->get(GrB_NT, &nt_mode) );

    const int ta = static_cast<int>(ta_mode);
    const int tb = static_cast<int>(tb_mode);
    const int nt = static_cast<int>(nt_mode);

    dim3 NT, NB;
    NT.x = nt;
    NT.y = 1;
    NT.z = 1;
    NB.x = (*u_nvals+nt-1)/nt;
    NB.y = 1;
    NB.z = 1;

    //Step 0) Must compute how many elements are in the selected region in the
    //        worst-case. This is a global reduce.
    //  -> d_temp_nvals |V|
    //  -> d_scan       |V|
    int    size        = (float) A_nvals*desc->memusage()+1;
    void* d_temp_nvals = (void*)w_ind;
    void* d_scan       = (void*)w_val;
    void* d_temp       = desc->d_buffer_+ 2*A_nrows      *sizeof(Index);

    if( desc->struconly() )
      d_scan = desc->d_buffer_+(A_nrows+size)*sizeof(Index);

    if( desc->debug() )
    {
      assert( *u_nvals<A_nrows );
      std::cout << NT.x << " " << NB.x << std::endl;
    }

    indirectScanKernel<<<NB,NT>>>( (Index*)d_temp_nvals, A_csrRowPtr, u_ind, 
        *u_nvals );
    // Note: cannot use op.add_op() here
    mgpu::ScanPrealloc<mgpu::MgpuScanTypeExc>( (Index*)d_temp_nvals, *u_nvals,
        (Index)0, mgpu::plus<Index>(), (Index*)d_scan+(*u_nvals), w_nvals, 
        (Index*)d_scan, (Index*)d_temp, *(desc->d_context_) );

    if( desc->debug() )
    {
      printDevice( "d_temp_nvals", (Index*)d_temp_nvals, *u_nvals );
      printDevice( "d_scan",       (Index*)d_scan,       *u_nvals+1 );

      std::cout << "u_nvals: " << *u_nvals << std::endl;
      std::cout << "w_nvals: " << *w_nvals << std::endl;
    }

    if( desc->struconly() && !desc->sort() )
    {
      CUDA( cudaMemset(w_ind, 0, A_nrows*sizeof(Index)) );
      //CUDA( cudaMemsetAsync(w_ind, 0, A_nrows*sizeof(Index)) );
    }

    // No neighbors is one possible stopping condition
    if( *w_nvals==0 )
      return GrB_SUCCESS;

		//Step 1) Gather from CSR graph into one big array  |     |  |
    //Step 2) Vector Portion
		//   -IntervalExpand into frontier-length list
		//      1. Gather the elements indexed by d_csrVecInd
		//      2. Expand the elements to memory set by d_csrColGood
		//   -Element-wise multiplication with frontier
		//Step 3) Matrix Structure Portion
		//Step 4) Element-wise multiplication
		//Step 1-4) custom kernel method (1 single kernel)
		//  modify spmvCsrIndirectBinary() to stop after expand phase
    //  output: 1) expanded index array 2) expanded value array
    //  -> d_csrSwapInd |E| x desc->memusage()
    //  -> d_csrSwapVal |E| x desc->memusage()
    void* d_csrSwapInd;
    void* d_csrSwapVal;

    if( desc->struconly() )
    {
      d_csrSwapInd = desc->d_buffer_+   A_nrows      *sizeof(Index);
      d_temp       = desc->d_buffer_+(  A_nrows+size)*sizeof(Index);
    }
    else
    {
      d_csrSwapInd = desc->d_buffer_+ 2*A_nrows        *sizeof(Index);
      d_csrSwapVal = desc->d_buffer_+(2*A_nrows+  size)*sizeof(Index);
      d_temp       = desc->d_buffer_+(2*A_nrows+2*size)*sizeof(Index);
    }
		/*indirectGather<<<NB,NT>>>( (Index*)d_temp_nvals, A_csrRowPtr, u_ind, 
				*u_nvals );
    printDevice( "d_temp_nvals", (Index*)d_temp_nvals, *u_nvals );
		IntervalGather( *w_nvals, (Index*)d_temp_nvals, (Index*)d_scan, *u_nvals, 
        A_csrColInd, d_csrSwapInd, *(desc->d_context_) );
		IntervalGather( *w_nvals, (Index*)d_temp_nvals, (Index*)d_scan, *u_nvals, 
        A_csrVal, d_csrSwapVal, *(desc->d_context_) );*/

    // TODO: Add element-wise multiplication with frontier
    // -uses op.mul_op()
    
    //if( desc->prealloc() )
    //{
    /*  IntervalGatherIndirectPrealloc( *w_nvals, A_csrRowPtr, (Index*)d_scan, 
          *u_nvals, A_csrColInd, u_ind, (Index*)d_csrSwapInd, (Index*)d_temp, 
          *(desc->d_context_) );
      if( !desc->struconly() )
        IntervalGatherIndirectPrealloc( *w_nvals, A_csrRowPtr, (Index*)d_scan, 
            *u_nvals, A_csrVal, u_ind, (T*)d_csrSwapVal, (Index*)d_temp,
            *(desc->d_context_) );
    }
    else
    {*/
      IntervalGatherIndirect( *w_nvals, A_csrRowPtr, (Index*)d_scan, 
        *u_nvals, A_csrColInd, u_ind, (Index*)d_csrSwapInd, 
        *(desc->d_context_) );
      if( !desc->struconly() )
        IntervalGatherIndirect( *w_nvals, A_csrRowPtr, (Index*)d_scan, 
            *u_nvals, A_csrVal, u_ind, (T*)d_csrSwapVal,
            *(desc->d_context_) );
    //}
		//Step 4) Element-wise multiplication
    //mgpu::SpmspvCsrIndirectBinary(A_csrVal, A_csrColInd, *w_nvals, 
    //    A_csrRowPtr, A_nrows, u_ind, u_val, *u_nvals, false, w_ind, w_val, 
    //    w_nvals, (T)identity, mul_op, *(desc->d_context_));
    //CUDA( cudaDeviceSynchronize() );

    if( desc->debug() )
    {
      printDevice( "SwapInd", (Index*)d_csrSwapInd, *w_nvals );
      if( !desc->struconly() )
        printDevice( "SwapVal", (T*)    d_csrSwapVal, *w_nvals );
    }

		//Step 5) Sort step
    //  -> d_csrTempInd |E| x desc->memusage()
    //  -> d_csrTempVal |E| x desc->memusage()
    size_t temp_storage_bytes = 0;
    void* d_csrTempInd;
    void* d_csrTempVal;

    int endbit = sizeof(Index)*8;
    if( desc->endbit() )
      endbit = min(endbit, (int)log2((float)A_nrows)+1);

    if( desc->struconly() )
    {
      if( desc->sort() )
      {
        d_csrTempInd = desc->d_buffer_+(A_nrows+size)*sizeof(Index);
      
        if( !desc->split() )
          CUDA( cub::DeviceRadixSort::SortKeys(NULL, temp_storage_bytes, 
              (Index*)d_csrSwapInd, (Index*)d_csrTempInd, *w_nvals, 0, endbit));
        else
          temp_storage_bytes = desc->d_temp_size_;
        
        if( desc->debug() )
        {
          std::cout << temp_storage_bytes << " bytes required!\n";
        }

        desc->resize( temp_storage_bytes, "temp" );

        CUDA( cub::DeviceRadixSort::SortKeys(desc->d_temp_, temp_storage_bytes,
            (Index*)d_csrSwapInd, (Index*)d_csrTempInd, *w_nvals, 0, endbit) );

        if( desc->debug() )
          printDevice( "TempInd", (Index*)d_csrTempInd, *w_nvals );
      }
    }
    else
    {
      d_csrTempInd = desc->d_buffer_+(2*A_nrows+2*size)*sizeof(Index);
      d_csrTempVal = desc->d_buffer_+(2*A_nrows+3*size)*sizeof(Index);

      if( !desc->split() )
        CUDA( cub::DeviceRadixSort::SortPairs(NULL, temp_storage_bytes, 
            (Index*)d_csrSwapInd, (Index*)d_csrTempInd, (T*)d_csrSwapVal, 
            (T*)d_csrTempVal, *w_nvals, 0, endbit) ); 
      else
        temp_storage_bytes = desc->d_temp_size_;
 
      if( desc->debug() )
      {
        std::cout << temp_storage_bytes << " bytes required!\n";
      }

      desc->resize( temp_storage_bytes, "temp" );

      CUDA( cub::DeviceRadixSort::SortPairs(desc->d_temp_, temp_storage_bytes, 
          (Index*)d_csrSwapInd, (Index*)d_csrTempInd, (T*)d_csrSwapVal, 
          (T*)d_csrTempVal, *w_nvals, 0, endbit) );
      //MergesortKeys(d_csrVecInd, total, mgpu::less<int>(), desc->d_context_);

      if( desc->debug() )
      {
        printDevice( "TempInd", (Index*)d_csrTempInd, *w_nvals );
        printDevice( "TempVal", (T*)    d_csrTempVal, *w_nvals );
      }
    }

		if( desc->debug() )
    {
      printf("Endbit: %d\n", endbit);
      printf("Current iteration: %d nonzero vector, %d edges\n", *u_nvals, 
        *w_nvals);
    }

		//Step 6) Segmented Reduce By Key
    if( desc->struconly() )
    {
      if( !desc->sort() )
      {
        NB.x = (*w_nvals+nt-1)/nt;
        scatter<<<NB,NT>>>(w_ind, (Index*)d_csrSwapInd, (Index)1, *w_nvals);
        *w_nvals = A_nrows;

        if( desc->debug() )
          printDevice("scatter", w_ind, *w_nvals);
      }
      else
      {
        d_temp = desc->d_buffer_+(A_nrows+2*size)*sizeof(Index);

        Index  w_nvals_t = 0;
        /*if( desc->prealloc() )
          ReduceByKeyPrealloc( (Index*)d_csrTempInd, (T*)d_csrSwapInd, *w_nvals,
              op.identity(), op<GrB_ADD>(), mgpu::equal_to<int>(), w_ind, w_val,
              &w_nvals_t, (int*)0, (int*)d_temp, (int*)desc->d_temp_, 
              *(desc->d_context_) );
        else*/
        ReduceByKey( (Index*)d_csrTempInd, (T*)d_csrSwapInd, *w_nvals, 
            op.identity(), op.operator()<T,GrB_ADD>, mgpu::equal_to<T>(), w_ind, w_val, 
            &w_nvals_t, (int*)0, *(desc->d_context_) );
        *w_nvals         = w_nvals_t;
      }
    }
    else
    {
      Index  w_nvals_t = 0;
      ReduceByKey( (Index*)d_csrTempInd, (T*)d_csrTempVal, *w_nvals, 
          op.identity(), op<T,GrB_ADD>(), mgpu::equal_to<T>(), w_ind, w_val, 
          &w_nvals_t, (int*)0, *(desc->d_context_) );
      *w_nvals         = w_nvals_t;
    }

    return GrB_SUCCESS;
  }

  template <typename W, typename a, typename U,
            typename BinaryOpT, typename SemiringT>
  Info spmspvGunrockLB( Index*            w_ind,
                        W*                w_val,
                        Index*            w_nvals,
                        BinaryOpT         accum,
                        SemiringT         op,
                        Index             A_nrows,
                        Index             A_nvals,
                        const Index*      A_csrRowPtr,
                        const Index*      A_csrColInd,
                        const a*          A_csrVal,
                        const Index*      u_ind,
                        const U*          u_val,
                        const Index*      u_nvals,
                        Descriptor*       desc )
  {
    return GrB_SUCCESS;
  }

  template <typename W, typename a, typename U,
            typename BinaryOpT, typename SemiringT>
  Info spmspvGunrockTWC( Index*            w_ind,
                         W*                w_val,
                         Index*            w_nvals,
                         BinaryOpT         accum,
                         SemiringT         op,
                         Index             A_nrows,
                         Index             A_nvals,
                         const Index*      A_csrRowPtr,
                         const Index*      A_csrColInd,
                         const a*          A_csrVal,
                         const Index*      u_ind,
                         const U*          u_val,
                         const Index*      u_nvals,
                         Descriptor*       desc )
  {
    return GrB_SUCCESS;
  }

  //__global__ void filterKernel();

}  // backend
}  // graphblas

#endif  // GRB_BACKEND_APSPIE_KERNELS_SPMSPV_HPP
