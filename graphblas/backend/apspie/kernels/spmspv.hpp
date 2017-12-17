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

  template <bool UseScmp, bool UseAccum, bool UseRepl,
            typename W, typename a, typename U,
            typename AccumOp, typename MulOp, typename AddOp>
  void spmspvApspie( W*                w_val,
                     AccumOp           accum_op,
                     a                 identity,
                     MulOp             mul_op,
                     AddOp             add_op,
                     Index             A_nrows,
                     Index             A_nvals,
                     const Index*      A_csrRowPtr,
                     const Index*      A_csrColInd,
                     const a*          A_csrVal,
                     const U*          u_val,
                     const Descriptor* desc )
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
    NB.x = (ta*A_nrows+nt-1)/nt;
    NB.y = 1;
    NB.z = 1;

		//Step 1) Gather from CSR graph into one big array  |     |  |
		// 1. Extracts the row lengths we are interested in 3  3  3  2  3  1
		//  -> d_cscColBad
		// 2. Scans them, giving the offset from 0          0  3  6  8
		//  -> d_cscColGood
		// 3. Extracts the col indices we are interested in 0  6  9
		//  -> d_cscColBad
		// 4. Extracts the neighbour lists
		//  -> d_cscVecInd
		//  -> d_cscVecVal
		IntervalGather( h_cscVecCount, d->d_cscVecInd, d->d_index, h_cscVecCount, d->d_cscColDiff, d->d_cscColBad, context );
		mgpu::Scan<mgpu::MgpuScanTypeExc>( d->d_cscColBad, h_cscVecCount, 0, mgpu::plus<int>(), (int*)0, &total, d->d_cscColGood, context );
		IntervalGather( h_cscVecCount, d->d_cscVecInd, d->d_index, h_cscVecCount, d_cscColPtr, d->d_cscColBad, context );

		//printf("Processing %d nodes frontier size: %d\n", h_cscVecCount, total);

    //Step 2) Vector Portion
		// a) naive method
		//   -IntervalExpand into frontier-length list
		//      1. Gather the elements indexed by d_cscVecInd
		//      2. Expand the elements to memory set by d_cscColGood
		//   -Element-wise multiplication with frontier
		IntervalGather( h_cscVecCount, d->d_cscVecInd, d->d_index, h_cscVecCount, d_randVec, d->d_cscTempVal, context );
		IntervalExpand( total, d->d_cscColGood, d->d_cscTempVal, h_cscVecCount, d->d_cscSwapVal, context );

		//Step 3) Matrix Structure Portion
		IntervalGather( total, d->d_cscColBad, d->d_cscColGood, h_cscVecCount, d_cscRowInd, d->d_cscVecInd, context );
		IntervalGather( total, d->d_cscColBad, d->d_cscColGood, h_cscVecCount, d_cscVal, d->d_cscTempVal, context );

		//Step 4) Element-wise multiplication
		elementMult<<<NBLOCKS, NTHREADS>>>( total, d->d_cscSwapVal, d->d_cscTempVal, d->d_cscVecVal );

		//Step 1-4) custom kernel method (1 single kernel)
		//  modify spmvCsrBinary() to use Indirect load and stop after expand phase
    //  output: 1) index array 2) value array

		// Reset dense flag array
		preprocessFlag<<<NBLOCKS,NTHREADS>>>( d_mmResult, m );

		//Step 5) Sort step
		//IntervalGather( ceil(h_cscVecCount/2.0), everyOther->get(), d_index, ceil(h_cscVecCount/2.0), d_cscColGood, d_cscColBad, context );
		//SegSortKeysFromIndices( d_cscVecInd, total, d_cscColBad, ceil(h_cscVecCount/2.0), context );
		//LocalitySortKeys( d_cscVecInd, total, context );
		cub::DeviceRadixSort::SortPairs( d->d_temp_storage, temp_storage_bytes, d->d_cscVecInd, d->d_cscSwapInd, d->d_cscVecVal, d->d_cscSwapVal, total );
		//MergesortKeys(d_cscVecInd, total, mgpu::less<int>(), context);

		//Step 6) Gather the rand values
		//gather<<<NBLOCKS,NTHREADS>>>( total, d_cscVecVal, d_randVec, d_cscVecVal );

		//Step 7) Segmented Reduce By Key
		ReduceByKey( d->d_cscVecInd, d->d_cscVecVal, total, (float)0, mgpu::plus<float>(), mgpu::equal_to<int>(), d->d_cscSwapInd, d->d_cscSwapVal, &h_cscVecCount, (int*)0, context );

		//printf("Current iteration: %d nonzero vector, %d edges\n",  h_cscVecCount, total);

    //Step 8) Sparse Vector to Dense Vector
		//scatterFloat<<<NBLOCKS,NTHREADS>>>( h_cscVecCount, d->d_cscSwapInd, d->d_cscSwapVal, d_mmResult );
    //return total;
  }

  //__global__ void filterKernel();

}  // backend
}  // graphblas

#endif  // GRB_BACKEND_APSPIE_KERNELS_SPMSPV_HPP
