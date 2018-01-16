#ifndef GRB_BACKEND_APSPIE_KERNELS_SPMSPV_HPP
#define GRB_BACKEND_APSPIE_KERNELS_SPMSPV_HPP

#include <iostream>

#include <cuda.h>
#include <cusparse.h>

#include <moderngpu.cuh>
#include <cub.cuh>

namespace graphblas
{
namespace backend
{
  __global__ void indirectScanKernel( Index*       d_temp_nvals,
			                                const Index* A_csrRowPtr, 
                                      const Index* u_ind, 
                                      Index        u_nvals )
  {
    int gid = blockIdx.x*blockDim.x+threadIdx.x;
    Index length = 0;

    if( gid<u_nvals )
    {
      Index row   = __ldg( u_ind+gid );

      Index start = __ldg( A_csrRowPtr+row   );
      Index end   = __ldg( A_csrRowPtr+row+1 );
      length      = end-start;

      d_temp_nvals[gid] = length;
      //if( tid<10 ) printf("%d: %d = %d - %d\n", length, start, end);
    }
  }

  __global__ void indirectGather( Index*       d_temp_nvals, 
																	const Index* A_csrRowPtr, 
			                            const Index* u_ind, 
																	Index        u_nvals )
	{
    int gid = blockIdx.x*blockDim.x+threadIdx.x;

		if( gid<u_nvals )
		{
      Index   row = __ldg( u_ind+gid );
			Index start = __ldg( A_csrRowPtr+row );
			d_temp_nvals[gid] = start;
		}
	}

  // Memory requirements: (4|V|+5|E|)*GrB_THRESHOLD
  //   -GrB_THRESHOLD is defined in graphblas/types.hpp
  //
  //  -> d_csrColBad    |V|*GrB_THRESHOLD
  //  -> d_csrColGood   |V|*GrB_THRESHOLD
  //  -> d_csrColDiff   |V|*GrB_THRESHOLD
  //  -> d_index        |V|*GrB_THRESHOLD
  //  -> d_csrVecInd    |E|*GrB_THRESHOLD (u_ind)
  //  -> d_csrSwapInd   |E|*GrB_THRESHOLD
  //  -> d_csrVecVal    |E|*GrB_THRESHOLD
  //  -> d_csrTempVal   |E|*GrB_THRESHOLD (u_val)
  //  -> d_csrSwapVal   |E|*GrB_THRESHOLD
  //  -> w_ind          |E|*GrB_THRESHOLD
  //  -> w_val          |E|*GrB_THRESHOLD
  //  -> d_temp_storage runtime constant
  template <typename W, typename a, typename U,
            typename AccumOp, typename MulOp, typename AddOp>
  Info spmspvApspie( Index*            w_ind,
                     W*                w_val,
                     Index*            w_nvals,
                     AccumOp           accum_op,
                     a                 identity,
                     MulOp             mul_op,
                     AddOp             add_op,
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
    /*Desc_value ta_mode, tb_mode, nt_mode;
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
    NB.x = (ta*A_nrows+nt-1)/nt;
    NB.y = 1;
    NB.z = 1;

		//Step 1) Gather from CSR graph into one big array  |     |  |
		// 1. Extracts the row lengths we are interested in 3  3  3  2  3  1
		//  -> d_csrColBad  |V|/2
		// 2. Scans them, giving the offset from 0          0  3  6  8
		//  -> d_csrColGood |V|/2
		// 3. Extracts the col indices starts we are interested in 0  6  9
		//  -> d_csrColBad  |V|/2
		// 4. Extracts the neighbour lists
		//  -> d_csrVecInd  |E|/2 (u_ind)
		//  -> d_csrVecVal  |E|/2
		IntervalGather( h_csrVecCount, u_ind, d->d_index, h_csrVecCount, d->d_csrColDiff, d->d_csrColBad, context );
		mgpu::Scan<mgpu::MgpuScanTypeExc>( d->d_csrColBad, h_csrVecCount, 0, mgpu::plus<int>(), (int*)0, &total, d->d_csrColGood, context );
		IntervalGather( h_csrVecCount, u_ind, d->d_index, h_csrVecCount, d_csrColPtr, d->d_csrColBad, context );

		//printf("Processing %d nodes frontier size: %d\n", h_csrVecCount, total);

    //Step 2) Vector Portion
		// a) naive method
		//   -IntervalExpand into frontier-length list
		//      1. Gather the elements indexed by d_csrVecInd
		//      2. Expand the elements to memory set by d_csrColGood
		//   -Element-wise multiplication with frontier
    //  -> d_csrTempVal |E|/2 (u_val)
    //  -> d_csrSwapVal |E|/2
		//IntervalGather( h_csrVecCount, d->d_csrVecInd, d->d_index, h_csrVecCount, d_randVec, d->d_csrTempVal, context );
		IntervalExpand( total, d->d_csrColGood, u_val, h_csrVecCount, d->d_csrSwapVal, context );

		//Step 3) Matrix Structure Portion
    //  -> d_csrVecInd  |E|/2
    //  -> d_csrTempVal |E|/2
		IntervalGather( total, d->d_csrColBad, d->d_csrColGood, h_csrVecCount, d_csrRowInd, d->d_csrVecInd, context );
		IntervalGather( total, d->d_csrColBad, d->d_csrColGood, h_csrVecCount, d_csrVal, d->d_csrTempVal, context );

		//Step 4) Element-wise multiplication
		elementMult<<<NBLOCKS, NTHREADS>>>( total, d->d_csrSwapVal, d->d_csrTempVal, d->d_csrVecVal );

		//Step 1-4) custom kernel method (1 single kernel)
		//  modify spmvCsrBinary() to use Indirect load and stop after expand phase
    //  output: 1) index array 2) value array

		//Step 5) Sort step
    //  -> d_csrSwapInd |E|/2
    //  -> d_csrVecVal  |E|/2
    //  -> d_csrSwapVal |E|/2
		cub::DeviceRadixSort::SortPairs( d->d_temp_storage, temp_storage_bytes, 
        d->d_csrVecInd, d->d_csrSwapInd, d->d_csrVecVal, d->d_csrSwapVal, 
        total );
		CUDA( cudaMalloc(&d->d_temp_storage, temp_storage_bytes) );
		cub::DeviceRadixSort::SortPairs( d->d_temp_storage, temp_storage_bytes, 
        d->d_csrVecInd, d->d_csrSwapInd, d->d_csrVecVal, d->d_csrSwapVal, 
        total );
		//MergesortKeys(d_csrVecInd, total, mgpu::less<int>(), context);

		//Step 6) Gather the rand values
		//gather<<<NBLOCKS,NTHREADS>>>( total, d_csrVecVal, d_randVec, d_csrVecVal );

		//Step 7) Segmented Reduce By Key
		ReduceByKey( d->d_csrSwapInd, d->d_csrSwapVal, total, (float)0, 
        mgpu::plus<float>(), mgpu::equal_to<int>(), w_ind, w_val, 
        &h_csrVecCount, (int*)0, context );

		//printf("Current iteration: %d nonzero vector, %d edges\n",  h_csrVecCount, total);

		//Step 8) Reset dense flag array
    //  -> d_mmResult  |V|
		//preprocessFlag<<<NBLOCKS,NTHREADS>>>( d_mmResult, m );

    //Step 9) Sparse Vector to Dense Vector
		//scatterFloat<<<NBLOCKS,NTHREADS>>>( h_csrVecCount, d->d_csrSwapInd, d->d_csrSwapVal, d_mmResult );
    //return total;*/
    return GrB_SUCCESS;
  }

  // Memory requirements: 2|E|*GrB_THRESHOLD
  //   -GrB_THRESHOLD is defined in graphblas/types.hpp
  // 
  //  -> d_csrSwapInd   |E|*GrB_THRESHOLD [2*A_nrows: 1*|E|*GrB_THRESHOLD]
  //  -> d_csrSwapVal   |E|*GrB_THRESHOLD [2*A_nrows+ 2*|E|*GrB_THRESHOLD]
  //  -> d_temp_storage runtime constant
  //
  // TODO: can lower 2|E|*GrB_THRESHOLD memory requirement further by doing 
  //       external memory sorting
  template <typename W, typename a, typename U,
            typename AccumOp, typename MulOp, typename AddOp>
  Info spmspvApspieLB( Index*            w_ind,
                       W*                w_val,
                       Index*            w_nvals,
                       AccumOp           accum_op,
                       a                 identity,
                       MulOp             mul_op,
                       AddOp             add_op,
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
    void* d_temp_nvals = (void*)w_ind;
    void* d_scan       = (void*)w_val;
    if( GrB_DEBUG )
    {
      assert( *u_nvals<A_nrows );
      std::cout << NT.x << " " << NB.x << std::endl;
    }

    indirectScanKernel<<<NB,NT>>>( (Index*)d_temp_nvals, A_csrRowPtr, u_ind, 
				*u_nvals );
    mgpu::Scan<mgpu::MgpuScanTypeExc>( (Index*)d_temp_nvals, *u_nvals, 0,
        mgpu::plus<int>(), (Index*)d_scan+(*u_nvals), w_nvals, (Index*)d_scan, 
        *(desc->d_context_) );

    if( GrB_DEBUG )
    {
      printDevice( "d_temp_nvals", (Index*)d_temp_nvals, *u_nvals );
      printDevice( "d_scan",       (Index*)d_scan,       *u_nvals+1 );

      std::cout << "u_nvals: " << *u_nvals << std::endl;
      std::cout << "w_nvals: " << *w_nvals << std::endl;
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
    //  -> d_csrSwapInd |E|xGrB_THRESHOLD
    //  -> d_csrSwapVal |E|xGrB_THRESHOLD
    int    size         = (float)  A_nvals*GrB_THRESHOLD+1;
    void* d_csrSwapInd = desc->d_buffer_+ 2*A_nrows      *sizeof(Index);
    void* d_csrSwapVal = desc->d_buffer_+(2*A_nrows+size)*sizeof(Index);
		/*indirectGather<<<NB,NT>>>( (Index*)d_temp_nvals, A_csrRowPtr, u_ind, 
				*u_nvals );
    printDevice( "d_temp_nvals", (Index*)d_temp_nvals, *u_nvals );
		IntervalGather( *w_nvals, (Index*)d_temp_nvals, (Index*)d_scan, *u_nvals, 
        A_csrColInd, d_csrSwapInd, *(desc->d_context_) );
		IntervalGather( *w_nvals, (Index*)d_temp_nvals, (Index*)d_scan, *u_nvals, 
        A_csrVal, d_csrSwapVal, *(desc->d_context_) );*/

		IntervalGatherIndirect( *w_nvals, A_csrRowPtr, (Index*)d_scan, *u_nvals, 
        A_csrColInd, u_ind, (Index*)d_csrSwapInd, *(desc->d_context_) );
		IntervalGatherIndirect( *w_nvals, A_csrRowPtr, (Index*)d_scan, *u_nvals, 
        A_csrVal, u_ind, (T*)d_csrSwapVal, *(desc->d_context_) );

		//Step 4) Element-wise multiplication
    //mgpu::SpmspvCsrIndirectBinary(A_csrVal, A_csrColInd, *w_nvals, 
    //    A_csrRowPtr, A_nrows, u_ind, u_val, *u_nvals, false, w_ind, w_val, 
    //    w_nvals, (T)identity, mul_op, *(desc->d_context_));
    //CUDA( cudaDeviceSynchronize() );

    if( GrB_DEBUG )
    {
      printDevice( "SwapInd", (Index*)d_csrSwapInd, *w_nvals );
      printDevice( "SwapVal", (T*)    d_csrSwapVal, *w_nvals );
    }

		//Step 5) Sort step
    //  -> d_csrTempInd |E|xGrB_THRESHOLD
    //  -> d_csrTempVal |E|xGrB_THRESHOLD
    size_t temp_storage_bytes;
    void* d_csrTempInd = desc->d_buffer_+(2*A_nrows+2*size)*sizeof(Index);
    void* d_csrTempVal = desc->d_buffer_+(2*A_nrows+3*size)*sizeof(Index);

    CUDA( cub::DeviceRadixSort::SortPairs(NULL, temp_storage_bytes, 
				(Index*)d_csrSwapInd, (Index*)d_csrTempInd, (T*)d_csrSwapVal, 
				(T*)d_csrTempVal, *w_nvals) );

    if( GrB_DEBUG )
    {
      std::cout << temp_storage_bytes << " bytes required!\n";
    }

    desc->resize( temp_storage_bytes, "temp" );

    CUDA( cub::DeviceRadixSort::SortPairs(desc->d_temp_, temp_storage_bytes, 
				(Index*)d_csrSwapInd, (Index*)d_csrTempInd, (T*)d_csrSwapVal, 
				(T*)d_csrTempVal, *w_nvals) );
    //desc->clear("temp");
		//MergesortKeys(d_csrVecInd, total, mgpu::less<int>(), desc->d_context_);
    if( GrB_DEBUG )
    {
      printDevice( "TempInd", (Index*)d_csrTempInd, *w_nvals );
      printDevice( "TempVal", (T*)    d_csrTempVal, *w_nvals );
    }

		if( GrB_DEBUG )
    {
      printf("Current iteration: %d nonzero vector, %d edges\n", *u_nvals, 
        *w_nvals);
    }

		//Step 6) Segmented Reduce By Key
    Index  w_nvals_t    = 0;
		ReduceByKey( (Index*)d_csrTempInd, (T*)d_csrTempVal, *w_nvals, (float)0, 
        add_op, mgpu::equal_to<int>(), w_ind, w_val, 
        &w_nvals_t, (int*)0, *(desc->d_context_) );
    *w_nvals            = w_nvals_t;

    return GrB_SUCCESS;
  }

  template <typename W, typename a, typename U,
            typename AccumOp, typename MulOp, typename AddOp>
  Info spmspvGunrockLB( Index*            w_ind,
                        W*                w_val,
                        Index*            w_nvals,
                        AccumOp           accum_op,
                        a                 identity,
                        MulOp             mul_op,
                        AddOp             add_op,
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
            typename AccumOp, typename MulOp, typename AddOp>
  Info spmspvGunrockTWC( Index*            w_ind,
                         W*                w_val,
                         Index*            w_nvals,
                         AccumOp           accum_op,
                         a                 identity,
                         MulOp             mul_op,
                         AddOp             add_op,
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
