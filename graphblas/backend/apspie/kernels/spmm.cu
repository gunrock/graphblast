#ifndef GRB_BACKEND_APSPIE_KERNELS_SPMM_CU
#define GRB_BACKEND_APSPIE_KERNELS_SPMM_CU

#include <cuda.h>
#include <cstdio>
//#include <helper_math.h>

//#define TA     32
//#define TB     32
//#define NT     64

namespace graphblas
{
namespace backend
{
    typedef magma_index_t Index;

  // In paper "Design Principles for Sparse Matrix Multiplication"
  /*template<typename c, int TB>
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
    int C_offset  = (row<<6)+lane_id+((blockIdx.y&1)<<5);

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
  } // spmmRowKernel2*/

  // Varies by B_ncols
  template<typename c, int TB>
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
    //int C_offset  = (row*B_ncols)+lane_id+(blockIdx.y<<5);
    int C_offset  = (lane_id+(blockIdx.y<<5))*A_nrows+row;

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

}  // backend
}  // graphblas

#endif  // GRB_BACKEND_APSPIE_KERNELS_SPMM_CU
