#ifndef GRB_BACKEND_APSPIE_KERNELS_SPMM_HPP
#define GRB_BACKEND_APSPIE_KERNELS_SPMM_HPP

#include <cuda.h>
#include <cstdio>
//#include <helper_math.h>

#include "graphblas/types.hpp"

//#define TA     32
//#define TB     32
//#define NT     64

namespace graphblas
{
namespace backend
{
  // Varies by B_ncols
  template<typename c, int TB, bool TRANS>
  __global__ void spmmRowKernel3( const Index A_nrows, 
      const Index B_ncols, const Index A_ncols, const Index A_nvals,
      const Index* A_csrRowPtr, const Index* A_csrColInd, const c* A_csrVal, 
      const c* B_denseVal, c* C_denseVal )
  {
    float vals[TB];
    int   col_all[TB];
    float val_all[TB];

    int thread_id = blockDim.x*blockIdx.x+threadIdx.x; // global thrd idx
    int warp_id   = thread_id>>5;                      // global warp idx
    int lane_id   = thread_id & (32 - 1);
    int row       = warp_id;
    const c* B_offset = B_denseVal+lane_id+(blockIdx.y<<5);
    int C_offset;
	  if( TRANS )
		  C_offset = (lane_id+(blockIdx.y<<5))*A_nrows+row;
		else
		  C_offset = (row*B_ncols)+lane_id+(blockIdx.y<<5);

    //if( threadIdx.x==0 )
    //  printf("row:%d\n", row);

    if( row < A_nrows )
    {
      int row_start = __ldg(A_csrRowPtr+row);
      int row_end   = __ldg(A_csrRowPtr+row+1);

      int   col = -1;
      float val = 0.f;
      float sum = 0.f;
      int   jj  = row_start+lane_id;

      //TODO: add popc() and ballot to query which to shfl
      if( blockIdx.y!=gridDim.y-1 )
      {
        for( int jj_start=row_start; jj_start<row_end; jj_start+=32 )
        {
          //#pragma unroll
          //for( int ii=0; ii<TB; ii++ )
          //  vals[ii] = 0.f;
          if( jj<row_end )
          {
            col = __ldg(A_csrColInd+jj)*B_ncols;
            val = __ldg(A_csrVal+jj);
          }
          else
          {
            col = 0;
            val = 0.f;
          }
          jj+=32;
          //if( warp_id==0 ) printf("tid:%d,col:%d,val:%f\n", threadIdx.x, col, val);
          for( int kk=0; kk<32; kk+=TB )
          {
            #pragma unroll
            for( int ii=0; ii<TB; ii++ )
            {
              col_all[ii] = __shfl(col, ii+kk);
              val_all[ii] = __shfl(val, ii+kk);
              //sum        += val_all[ii]*__ldg(B_offset+col_all[ii]);
              vals[   ii] = val_all[ii]*__ldg(B_offset+col_all[ii]);
              //vals[   ii] = __ldg(B_offset+col_all[ii]);
            }

            //if( warp_id==0 && blockIdx.y==0 )
            //  printf("row:%d,tid:%d,col_all:%d,ii:%d,load_id:%d,val:%f\n",row,thread_id,col_all>>6, ii, col_all+lane_id+((blockIdx.y&1)<<5), vals[ii]);

            #pragma unroll
            for( int ii=0; ii<TB; ii++ )
            {
              //val_all[ii] = __shfl(val, ii+kk);
              //sum += val_all[ii]*vals[ii];
              sum += vals[ii];
            //  if( threadIdx.x==1 && warp_id==0 && blockIdx.y==0 ) printf("tid:%d,ii:%d,val:%f\n", threadIdx.x, ii, vals[ii]);
            }
            //if( warp_id==0 && blockIdx.y==0 ) printf("tid:%d,val:%f\n", threadIdx.x, vals[0]);
          }
        }
        C_denseVal[C_offset] = sum;
      }
      else
      {
        int leftover = B_ncols - (blockIdx.y<<5);
        for( int jj_start=row_start; jj_start<row_end; jj_start+=32 )
        {
          //#pragma unroll
          //for( int ii=0; ii<TB; ii++ )
          //  vals[ii] = 0.f;
          if( jj<row_end )
          {
            col = __ldg(A_csrColInd+jj)*B_ncols;
            val = __ldg(A_csrVal+jj);
          }
          else
          {
            col = 0;
            val = 0.f;
          }
          jj+=32;
          //if( jj_start<row_start+32*5 && warp_id==0 ) printf("tid:%d,col:%d,val:%f\n", threadIdx.x, col, val);
          for( int kk=0; kk<32; kk+=TB )
          {
              #pragma unroll
              for( int ii=0; ii<TB; ii++ )
              {
                col_all[ii] = __shfl(col, ii+kk);
                val_all[ii] = __shfl(val, ii+kk);
                //sum        += val_all[ii]*__ldg(B_offset+col_all[ii]);
                if( lane_id<leftover )
                  vals[ii]  = val_all[ii]*__ldg(B_offset+col_all[ii]);
                else
                  vals[ii]  = 0.f;
                //vals[   ii] = __ldg(B_offset+col_all[ii]);
                //if( jj_start<row_start+32*5 && thread_id<2 && warp_id==0 && blockIdx.y==0 )
                  //printf("row:%d,tid:%d,ii:%d,val:%f\n",row,thread_id, ii, vals[ii]);
              }

              #pragma unroll
              for( int ii=0; ii<TB; ii++ )
              {
                //val_all[ii] = __shfl(val, ii+kk);
                //sum += val_all[ii]*vals[ii];
                sum += vals[ii];
              //  if( threadIdx.x==1 && warp_id==0 && blockIdx.y==0 ) printf("tid:%d,ii:%d,val:%f\n", threadIdx.x, ii, vals[ii]);
              }
              //if( jj_start<row_start+32*5 && warp_id==0 && blockIdx.y==0 ) printf("str tid:%d,val:%f\n", threadIdx.x, sum);
              //if( jj_start>row_end-32*5 && warp_id==0 && blockIdx.y==0 ) printf("end tid:%d,val:%f\n", threadIdx.x, sum);
          }
        }
        if( lane_id<leftover )
          C_denseVal[C_offset] = sum;
      }
    }
  } // spmmRowKernel3

  // In paper "Design Principles for Sparse Matrix Multiplication"
  template<typename c, int TB, bool TRANS>
  __global__ void spmmRowKernel2( const Index A_nrows, 
      const Index B_ncols, const Index A_ncols, const Index A_nvals,
      const Index* A_csrRowPtr, const Index* A_csrColInd, const c* A_csrVal, 
      const c* B_denseVal, c* C_denseVal )
  {
    float vals[TB];
    int   col_all[TB];
    float val_all[TB];

    int thread_id = blockDim.x*blockIdx.x+threadIdx.x; // global thrd idx
    int warp_id   = thread_id>>5;                      // global warp idx
    int lane_id   = thread_id & (32 - 1);
    int row       = warp_id;
    const c* B_offset = B_denseVal+lane_id+((blockIdx.y&1)<<5);
    int C_offset;
	  if( TRANS )
		  C_offset = (lane_id+(blockIdx.y<<5))*A_nrows+row;
		else
		  C_offset = (row*B_ncols)+lane_id+(blockIdx.y<<5);

    //if( threadIdx.x==0 )
    //  printf("row:%d\n", row);

    if( row < A_nrows )
    {
      int row_start = __ldg(A_csrRowPtr+row);
      int row_end   = __ldg(A_csrRowPtr+row+1);

      int   col = -1;
      float val = 0.f;
      float sum = 0.f;
      int   jj  = row_start+lane_id;

      //TODO: add popc() and ballot to query which to shfl
      for( int jj_start=row_start; jj_start<row_end; jj_start+=32 )
      {
        //#pragma unroll
        //for( int ii=0; ii<TB; ii++ )
        //  vals[ii] = 0.f;
        if( jj<row_end )
        {
          col = __ldg(A_csrColInd+jj)<<6;
          val = __ldg(A_csrVal+jj);
        }
        else
        {
          col = 0;
          val = 0.f;
        }
        jj+=32;
        //if( warp_id==0 ) printf("tid:%d,col:%d,val:%f\n", threadIdx.x, col, val);
        for( int kk=0; kk<32; kk+=TB )
        {
          #pragma unroll
          for( int ii=0; ii<TB; ii++ )
          {
            col_all[ii] = __shfl(col, ii+kk);
            val_all[ii] = __shfl(val, ii+kk);
            //sum        += val_all[ii]*__ldg(B_offset+col_all[ii]);
            vals[   ii] = val_all[ii]*__ldg(B_offset+col_all[ii]);
            //vals[   ii] = __ldg(B_offset+col_all[ii]);
          }

          //if( warp_id==0 && blockIdx.y==0 )
          //  printf("row:%d,tid:%d,col_all:%d,ii:%d,load_id:%d,val:%f\n",row,thread_id,col_all>>6, ii, col_all+lane_id+((blockIdx.y&1)<<5), vals[ii]);

          #pragma unroll
          for( int ii=0; ii<TB; ii++ )
          {
            //val_all[ii] = __shfl(val, ii+kk);
            //sum += val_all[ii]*vals[ii];
            sum += vals[ii];
          //  if( threadIdx.x==1 && warp_id==0 && blockIdx.y==0 ) printf("tid:%d,ii:%d,val:%f\n", threadIdx.x, ii, vals[ii]);
          }

          //if( warp_id==0 && blockIdx.y==0 ) printf("tid:%d,val:%f\n", threadIdx.x, vals[0]);
        }
      }

      C_denseVal[C_offset] = sum;
    }
  } // spmmRowKernel2


  // Baseline implementation (col major) based on Bell/Garland 2008
  template<typename c, int TB>
  __global__ void spmmColKernel( const Index A_nrows, 
      const Index B_ncols, const Index A_ncols, const Index A_nvals,
      const Index* A_csrRowPtr, const Index* A_csrColInd, const c* A_csrVal, 
      const c* B_denseVal, c* C_denseVal )
  {
    float vals[TB];

    int thread_id = blockDim.x*blockIdx.x+threadIdx.x; // global thrd idx
    int warp_id   = thread_id>>5;                      // global warp idx
    int lane      = thread_id & (32 - 1);

    // one warp per row
    int row, slab;
    //if( threadIdx.x==0 )
    //  printf("row:%d\n", row);

    for( slab=0; slab<B_ncols; slab+=TB ) {
      row = warp_id;

      if( row < A_nrows ) {
        int row_start = __ldg(A_csrRowPtr+row);
        int row_end   = __ldg(A_csrRowPtr+row+1);

        // compute running sum per thread
        #pragma unroll
        for( int ii=0; ii<TB; ii++ )
          vals[ii] = 0.0;

        for( int jj=row_start+lane; jj<row_end; jj+=32 ) {
          //printf("row:%d,tid:%d,jj:%d,row_start:%d,row_end:%d,slab:%d\n", row, threadIdx.x, jj, row_start, row_end, slab);
          int   col = A_csrColInd[jj];
          float val = A_csrVal[jj];
          #pragma unroll
          for( int ii=0; ii<TB; ii++ )
            vals[ii] += val*__ldg(B_denseVal+col+A_nrows*(ii+slab));
        }
      }

      // parallel reduction in shared memory
      #pragma unroll
      for( int ii=0; ii<TB; ii++ ) {
        vals[ii] += __shfl_xor(vals[ii], 16);
        vals[ii] += __shfl_xor(vals[ii], 8 );
        vals[ii] += __shfl_xor(vals[ii], 4 );
        vals[ii] += __shfl_xor(vals[ii], 2 );
        vals[ii] += __shfl_xor(vals[ii], 1 );
      }

      // first thread writes the result
      if( lane==0 )
        #pragma unroll
        for( int ii=0; ii<TB; ii++ )
          C_denseVal[row+A_nrows*(ii+slab)] = vals[ii];
    }

    // Incomplete slab
    /*row = warp_id;

    if( row < A_nrows ) {
      int row_start = __ldg(A_csrRowPtr+row);
      int row_end   = __ldg(A_csrRowPtr+row+1);

      // compute running sum per thread
      #pragma unroll
      for( int ii=0; ii<TB; ii++ )
        vals[ii] = 0.0;
      for( int jj=row_start+lane; jj<row_end; jj+=32 ) { 
        //printf("row:%d,tid:%d,jj:%d,row_start:%d,row_end:%d,slab:%d\n", row, threadIdx.x, jj, row_start, row_end, slab);
        int   col = A_csrColInd[jj];
        float val = A_csrVal[jj];
        #pragma unroll
        for( int ii=0; ii<TB; ii++ )
          if( ii+slab<B_ncols )
            vals[ii] += val*__ldg(B_denseVal+col+A_nrows*(ii+slab));
      }
    }

    // parallel reduction in shared memory
    for( int offset = 16; offset > 0; offset /= 2 )
      #pragma unroll
      for( int ii=0; ii<TB; ii++ )
        vals[ii] += __shfl_down(vals[ii], offset);

    // first thread writes the result
    if( lane==0 )
      #pragma unroll
      for( int ii=0; ii<TB; ii++ )
        if( ii+slab<B_ncols )
          C_denseVal[row+A_nrows*(ii+slab)] = vals[ii];*/
  }

}  // backend
}  // graphblas

#endif  // GRB_BACKEND_APSPIE_KERNELS_SPMM_HPP
