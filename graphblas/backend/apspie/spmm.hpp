#ifndef GRB_BACKEND_APSPIE_SPMM_HPP
#define GRB_BACKEND_APSPIE_SPMM_HPP

#include <iostream>

#include <cuda.h>
#include <cusparse.h>
#include <helper_math.h>

#include <moderngpu.cuh>

#include "graphblas/backend/apspie/SparseMatrix.hpp"
#include "graphblas/backend/apspie/DenseMatrix.hpp"
#include "graphblas/types.hpp"
#include "graphblas/util.hpp"

//#define TA     32
//#define TB     32
//#define NT     64

namespace graphblas
{
namespace backend
{
  template<typename c, int TB>
  __global__ void spmmRowKernel( const Index A_nrows, 
      const Index B_ncols, const Index A_ncols, const Index A_nvals,
      const Index* A_csrRowPtr, const Index* A_csrColInd, const c* A_csrVal, 
      const c* B_denseVal, c* C_denseVal );

  template<typename c, int TB>
  __global__ void spmmRowKernel2( const Index A_nrows, 
      const Index B_ncols, const Index A_ncols, const Index A_nvals,
      const Index* A_csrRowPtr, const Index* A_csrColInd, const c* A_csrVal, 
      const c* B_denseVal, c* C_denseVal );

  template<typename c, int TB>
  __global__ void spmmRowKernel3( const Index A_nrows, 
      const Index B_ncols, const Index A_ncols, const Index A_nvals,
      const Index* A_csrRowPtr, const Index* A_csrColInd, const c* A_csrVal, 
      const c* B_denseVal, c* C_denseVal );

  template<typename c, int TB>
  __global__ void spmmColKernel( const Index A_nrows, 
      const Index B_ncols, const Index A_ncols, const Index A_nvals,
      const Index* A_csrRowPtr, const Index* A_csrColInd, const c* A_csrVal, 
      const c* B_denseVal, c* C_denseVal );

  template<typename c, typename m, typename a, typename b>
  Info spmm( DenseMatrix<c>&        C,
             const SparseMatrix<m>& mask,
             const BinaryOp&        accum,
             const Semiring&        op,
             const SparseMatrix<a>& A,
             const DenseMatrix<b>&  B,
             const Descriptor&      desc )
  {
    Index A_nrows, A_ncols, A_nvals;
    Index B_nrows, B_ncols;
    Index C_nrows, C_ncols;

    A.nrows( A_nrows );
    A.ncols( A_ncols );
    A.nvals( A_nvals );
    B.nrows( B_nrows );
    B.ncols( B_ncols );
    C.nrows( C_nrows );
    C.ncols( C_ncols );

    // Dimension compatibility check
    if( (A_ncols != B_nrows) || (C_ncols != B_ncols) || (C_nrows != A_nrows ) )
    {
      std::cout << "Dim mismatch" << std::endl;
      std::cout << A_ncols << " " << B_nrows << std::endl;
      std::cout << C_ncols << " " << B_ncols << std::endl;
      std::cout << C_nrows << " " << A_nrows << std::endl;
      return GrB_DIMENSION_MISMATCH;
    }

    // Domain compatibility check
    // TODO: add domain compatibility check

    // Read descriptor
    Desc_value mode, ta, tb, nt;
    desc.get( GrB_MODE, mode );
    desc.get( GrB_TA  , ta );
    desc.get( GrB_TB  , tb );
    desc.get( GrB_NT  , nt );

    // Computation
    const int T        = static_cast<int>(ta);
    const int TB       = static_cast<int>(tb);
    const int NTHREADS = static_cast<int>(nt);
    const int NBLOCKS  = (T*A_nrows+NTHREADS-1)/NTHREADS;

    dim3 NT;
    dim3 NB;
    NT.x = NTHREADS;
    NT.y = 1;
    NT.z = 1;
    NB.x = NBLOCKS;
    NB.y = (B_ncols+31)/32;
    NB.z = 1;

    //CUDA( cudaDeviceSetCacheConfig( cudaFuncCachePreferL1 ) );
    if( mode == GrB_FIXEDROW )
      switch( TB ) {
        case 1:
          spmmRowKernel3<c,1><<<NB,NT>>>( A_nrows, B_ncols, A_ncols, 
              A_nvals, A.d_csrRowPtr_, A.d_csrColInd_, A.d_csrVal_,
              B.d_denseVal_, C.d_denseVal_ );
          break;
        case 2:
          spmmRowKernel3<c,2><<<NB,NT>>>( A_nrows, B_ncols, A_ncols, 
              A_nvals, A.d_csrRowPtr_, A.d_csrColInd_, A.d_csrVal_,
              B.d_denseVal_, C.d_denseVal_ );
          break;
        case 4:
          spmmRowKernel3<c,4><<<NB,NT>>>( A_nrows, B_ncols, A_ncols, 
              A_nvals, A.d_csrRowPtr_, A.d_csrColInd_, A.d_csrVal_,
              B.d_denseVal_, C.d_denseVal_ );
          break;
        case 8:
          spmmRowKernel3<c,8><<<NB,NT>>>( A_nrows, B_ncols, A_ncols, 
              A_nvals, A.d_csrRowPtr_, A.d_csrColInd_, A.d_csrVal_,
              B.d_denseVal_, C.d_denseVal_ );
          break;
        case 16:
          spmmRowKernel3<c,16><<<NB,NT>>>( A_nrows, B_ncols, A_ncols, 
              A_nvals, A.d_csrRowPtr_, A.d_csrColInd_, A.d_csrVal_,
              B.d_denseVal_, C.d_denseVal_ );
          break;
        case 32:
          spmmRowKernel3<c,32><<<NB,NT>>>( A_nrows, B_ncols, A_ncols, 
              A_nvals, A.d_csrRowPtr_, A.d_csrColInd_, A.d_csrVal_,
              B.d_denseVal_, C.d_denseVal_ );
          break;
        break;
      } else switch( TB ) {
        /*case 1:
          spmmColKernel<c,1><<<NBLOCKS,NTHREADS>>>( A_nrows, 
            B_ncols, A_ncols, A_nvals, A.d_csrRowPtr_, A.d_csrColInd_, A.d_csrVal_,
            B.d_denseVal_, C.d_denseVal_ );
          break;
        case 2:
          spmmColKernel<c,2><<<NBLOCKS,NTHREADS>>>( A_nrows, 
            B_ncols, A_ncols, A_nvals, A.d_csrRowPtr_, A.d_csrColInd_, A.d_csrVal_,
            B.d_denseVal_, C.d_denseVal_ );
          break;*/
        case 4:
          spmmColKernel<c,4><<<NBLOCKS,NTHREADS>>>( A_nrows, 
            B_ncols, A_ncols, A_nvals, A.d_csrRowPtr_, A.d_csrColInd_, A.d_csrVal_,
            B.d_denseVal_, C.d_denseVal_ );
          break;
        case 8:
          spmmColKernel<c,8><<<NBLOCKS,NTHREADS>>>( A_nrows, 
            B_ncols, A_ncols, A_nvals, A.d_csrRowPtr_, A.d_csrColInd_, A.d_csrVal_,
            B.d_denseVal_, C.d_denseVal_ );
          break;
        case 16:
          spmmColKernel<c,16><<<NBLOCKS,NTHREADS>>>( A_nrows, 
            B_ncols, A_ncols, A_nvals, A.d_csrRowPtr_, A.d_csrColInd_, A.d_csrVal_,
            B.d_denseVal_, C.d_denseVal_ );
          break;
        case 32:
          spmmColKernel<c,32><<<NBLOCKS,NTHREADS>>>( A_nrows, 
            B_ncols, A_ncols, A_nvals, A.d_csrRowPtr_, A.d_csrColInd_, A.d_csrVal_,
            B.d_denseVal_, C.d_denseVal_ );
          break;
      }

    //spmmColKernel<<<NBLOCKS,NTHREADS>>>( A_nrows, B_ncols, A_ncols, A_nvals,
    //  A.d_csrRowPtr_, A.d_csrColInd_, A.d_csrVal_, B.d_denseVal_, C.d_denseVal_ );

    C.need_update_ = true;
    return GrB_SUCCESS;
  }

  // Baseline implementation (row major) based on Bell/Garland 2008
  //
  template<typename c, int TB>
  __global__ void spmmRowKernel( const Index A_nrows, 
      const Index B_ncols, const Index A_ncols, const Index A_nvals,
      const Index* A_csrRowPtr, const Index* A_csrColInd, const c* A_csrVal, 
      const c* B_denseVal, c* C_denseVal )
      //const c* B_denseVal, float4* C_denseVal )
  {
    //float  vals[TB];
    float4 raws[TB>>2];

    int thread_id = blockDim.x*blockIdx.x+threadIdx.x; // global thrd idx
    int warp_id   = thread_id>>5;                      // global warp idx
    int lane      = thread_id & (32 - 1);
    int row;

    //for( slab=0; slab<B_ncols; slab+=TB ) {

    // one warp per row
    // Note: Must reset this value every slab
      row = warp_id;
      //if( threadIdx.x==0 )
      //  printf("row:%d,slab:%d\n", row, slab);

      if( row < A_nrows ) {
        int row_start = __ldg(A_csrRowPtr+row);
        int row_end   = __ldg(A_csrRowPtr+row+1);

        // compute running sum per thread
        #pragma unroll
        for( int ii=0; ii<TB>>2; ii++ )
        {
          raws[ii].x = 0.f;
          raws[ii].y = 0.f;
          raws[ii].z = 0.f;
          raws[ii].w = 0.f;
        }
        //#pragma unroll
        //for( int ii=0; ii<TB; ii++ )
        //  vals[ii] = 0.0;

        for( int jj=row_start+lane; jj<row_end; jj+=32 ) {
          int   col = A_csrColInd[jj];
          float val = A_csrVal[jj];

          #pragma unroll
          for( int ii=0; ii<64>>2; ii++ ) {
            raws[ii] += val*__ldg((float4*)(B_denseVal+(col<<6)+(ii<<2)));
            //raws[ii] = __ldg((float4*)(B_denseVal+col*B_ncols+(ii<<2)+slab));
            //printf("row:%d,tid:%d,vals_idx:%d\n",row,thread_id,(ii<<2)+slab);
            //printf("row:%d,col:%d,tid:%d,0:%.0f,1:%.0f,2:%.0f,3:%.0f,idx:%d\n",row,col,thread_id,raws[ii].x,raws[ii].y,raws[ii].z,raws[ii].w, col*B_ncols+(ii<<2)+slab);
            /*vals[(ii<<2)  ] += val*raws[ii].x;
            vals[(ii<<2)+1] += val*raws[ii].y;
            vals[(ii<<2)+2] += val*raws[ii].z;
            vals[(ii<<2)+3] += val*raws[ii].w;*/
          }
        }

        // parallel reduction in register memory
        //for( int offset = 16; offset > 0; offset /= 2 )
          #pragma unroll
          for( int ii=0; ii<TB>>2; ii++ ) {
            raws[ii].x += __shfl_xor(raws[ii].x, 16);
            raws[ii].x += __shfl_xor(raws[ii].x, 8 );
            raws[ii].x += __shfl_xor(raws[ii].x, 4 );
            raws[ii].x += __shfl_xor(raws[ii].x, 2 );
            raws[ii].x += __shfl_xor(raws[ii].x, 1 );
            raws[ii].y += __shfl_xor(raws[ii].y, 16);
            raws[ii].y += __shfl_xor(raws[ii].y, 8 );
            raws[ii].y += __shfl_xor(raws[ii].y, 4 );
            raws[ii].y += __shfl_xor(raws[ii].y, 2 );
            raws[ii].y += __shfl_xor(raws[ii].y, 1 );
            raws[ii].z += __shfl_xor(raws[ii].z, 16);
            raws[ii].z += __shfl_xor(raws[ii].z, 8 );
            raws[ii].z += __shfl_xor(raws[ii].z, 4 );
            raws[ii].z += __shfl_xor(raws[ii].z, 2 );
            raws[ii].z += __shfl_xor(raws[ii].z, 1 );
            raws[ii].w += __shfl_xor(raws[ii].w, 16);
            raws[ii].w += __shfl_xor(raws[ii].w, 8 );
            raws[ii].w += __shfl_xor(raws[ii].w, 4 );
            raws[ii].w += __shfl_xor(raws[ii].w, 2 );
            raws[ii].w += __shfl_xor(raws[ii].w, 1 );
          }
          /*#pragma unroll
          for( int ii=0; ii<TB; ii++ ) {
            vals[ii] += __shfl_xor(vals[ii], 16);
            vals[ii] += __shfl_xor(vals[ii], 8 );
            vals[ii] += __shfl_xor(vals[ii], 4 );
            vals[ii] += __shfl_xor(vals[ii], 2 );
            vals[ii] += __shfl_xor(vals[ii], 1 );
          }*/

        // first thread writes the result
        if( lane==0 ) {
          #pragma unroll
          //for( int ii=0; ii<TB; ii++ )
          for( int ii=0; ii<(TB>>2); ii++ )
            reinterpret_cast<float4*>(C_denseVal)[(row<<4)+ii] = raws[ii];
            //C_denseVal[(row<<6)+ii] = vals[ii];
            //C_denseVal[row*B_ncols+ii+slab] = vals[ii];
            //printf("ii:%d,ind:%d\n",ii,threadIdx.x-lane+ii);
        }
      }
    //} // slab
  } // spmmColKernel

  // In paper "Design Principles for Sparse Matrix Multiplication"
  template<typename c, int TB>
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
  } // spmmRowKernel2

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
    int C_offset  = (row*B_ncols)+lane_id+(blockIdx.y<<5);

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
          //if( warp_id==0 ) printf("tid:%d,col:%d,val:%f\n", threadIdx.x, col, val);
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
        if( lane_id<leftover )
          C_denseVal[C_offset] = sum;
      }
    }
  } // spmmColKernel
  // Baseline implementation (col major) based on Bell/Garland 2008
  //
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

  template<typename c, typename a, typename b>
  Info cusparse_spmm( DenseMatrix<c>&        C,
                      const Semiring&        op,
                      const SparseMatrix<a>& A,
                      const DenseMatrix<b>&  B )
  {
    Index A_nrows, A_ncols, A_nvals;
    Index B_nrows, B_ncols;
    Index C_nrows, C_ncols;

    A.nrows( A_nrows );
    A.ncols( A_ncols );
    A.nvals( A_nvals );
    B.nrows( B_nrows );
    B.ncols( B_ncols );
    C.nrows( C_nrows );
    C.ncols( C_ncols );

    // Dimension compatibility check
    if( (A_ncols != B_nrows) || (C_ncols != B_ncols) || (C_nrows != A_nrows ) )
    {
      std::cout << "Dim mismatch" << std::endl;
      std::cout << A_ncols << " " << B_nrows << std::endl;
      std::cout << C_ncols << " " << B_ncols << std::endl;
      std::cout << C_nrows << " " << A_nrows << std::endl;
      return GrB_DIMENSION_MISMATCH;
    }

    // Domain compatibility check
    // TODO: add domain compatibility check

    // Computation
    cusparseHandle_t handle;
    cusparseCreate( &handle );
    cusparseSetPointerMode( handle, CUSPARSE_POINTER_MODE_HOST );

    cusparseMatDescr_t descr;
    cusparseCreateMatDescr( &descr );

    cusparseSetMatType( descr, CUSPARSE_MATRIX_TYPE_GENERAL );
    cusparseSetMatIndexBase( descr, CUSPARSE_INDEX_BASE_ZERO );
    cusparseStatus_t status;
    float alpha = 1.0;
    float beta  = 0.0;
    status = cusparseScsrmm( handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE, A_nrows, B_ncols, A_ncols, A_nvals,
        &alpha, descr, A.d_csrVal_, A.d_csrRowPtr_, A.d_csrColInd_, B.d_denseVal_,
        A_ncols,      // ldb = max(1,k) since op(A) = A
        &beta, C.d_denseVal_,
        A_nrows );    // ldc = max(1,m) since op(A) = A

    switch( status ) {
        case CUSPARSE_STATUS_SUCCESS:
            //std::cout << "SpMM successful!\n";
            break;
        case CUSPARSE_STATUS_NOT_INITIALIZED:
            std::cout << "Error: Library not initialized.\n";
            break;
        case CUSPARSE_STATUS_INVALID_VALUE:
            std::cout << "Error: Invalid parameters m, n, or nnz.\n";
            break;
        case CUSPARSE_STATUS_EXECUTION_FAILED:
            std::cout << "Error: Failed to launch GPU.\n";
            break;
        case CUSPARSE_STATUS_ALLOC_FAILED:
            std::cout << "Error: Resources could not be allocated.\n";
            break;
        case CUSPARSE_STATUS_ARCH_MISMATCH:
            std::cout << "Error: Device architecture does not support.\n";
            break;
        case CUSPARSE_STATUS_INTERNAL_ERROR:
            std::cout << "Error: An internal operation failed.\n";
            break;
        case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            std::cout << "Error: Matrix type not supported.\n";
    }

    C.need_update_ = true;  // Set flag that we need to copy data from GPU
    return GrB_SUCCESS;
  }

  template<typename c, typename a, typename b>
  Info cusparse_spmm2( DenseMatrix<c>&        C,
                       const Semiring&        op,
                       const SparseMatrix<a>& A,
                       const DenseMatrix<b>&  B )
  {
    Index A_nrows, A_ncols, A_nvals;
    Index B_nrows, B_ncols;
    Index C_nrows, C_ncols;

    A.nrows( A_nrows );
    A.ncols( A_ncols );
    A.nvals( A_nvals );
    B.nrows( B_nrows );
    B.ncols( B_ncols );
    C.nrows( C_nrows );
    C.ncols( C_ncols );

    // Dimension compatibility check
    if( (A_ncols != B_nrows) || (C_ncols != B_ncols) || (C_nrows != A_nrows ) )
    {
      std::cout << "Dim mismatch" << std::endl;
      std::cout << A_ncols << " " << B_nrows << std::endl;
      std::cout << C_ncols << " " << B_ncols << std::endl;
      std::cout << C_nrows << " " << A_nrows << std::endl;
      return GrB_DIMENSION_MISMATCH;
    }

    // Domain compatibility check
    // TODO: add domain compatibility check

    // Computation
    cusparseHandle_t handle;
    cusparseCreate( &handle );
    cusparseSetPointerMode( handle, CUSPARSE_POINTER_MODE_HOST );

    cusparseMatDescr_t descr;
    cusparseCreateMatDescr( &descr );

    cusparseSetMatType( descr, CUSPARSE_MATRIX_TYPE_GENERAL );
    cusparseSetMatIndexBase( descr, CUSPARSE_INDEX_BASE_ZERO );
    cusparseStatus_t status;
    float alpha = 1.0;
    float beta  = 0.0;
    status = cusparseScsrmm2( handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
        A_nrows, B_ncols, A_ncols, A_nvals,
        &alpha, descr, 
        A.d_csrVal_, A.d_csrRowPtr_, A.d_csrColInd_, B.d_denseVal_,
        B_ncols,      // ldb = max(1,k) since op(A) = A
        &beta, C.d_denseVal_,
        A_nrows );    // ldc = max(1,m) since op(A) = A

    switch( status ) {
        case CUSPARSE_STATUS_SUCCESS:
            //std::cout << "SpMM successful!\n";
            break;
        case CUSPARSE_STATUS_NOT_INITIALIZED:
            std::cout << "Error: Library not initialized.\n";
            break;
        case CUSPARSE_STATUS_INVALID_VALUE:
            std::cout << "Error: Invalid parameters m, n, or nnz.\n";
            break;
        case CUSPARSE_STATUS_EXECUTION_FAILED:
            std::cout << "Error: Failed to launch GPU.\n";
            break;
        case CUSPARSE_STATUS_ALLOC_FAILED:
            std::cout << "Error: Resources could not be allocated.\n";
            break;
        case CUSPARSE_STATUS_ARCH_MISMATCH:
            std::cout << "Error: Device architecture does not support.\n";
            break;
        case CUSPARSE_STATUS_INTERNAL_ERROR:
            std::cout << "Error: An internal operation failed.\n";
            break;
        case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            std::cout << "Error: Matrix type not supported.\n";
    }

    C.need_update_ = true;  // Set flag that we need to copy data from GPU
    return GrB_SUCCESS;
  }

  template<typename c, typename a, typename b>
  Info mergepath_spmm( DenseMatrix<c>&        C,
                       const Semiring&        op,
                       const SparseMatrix<a>& A,
                       const DenseMatrix<b>&  B,
                       Descriptor&            desc )
  {
    Index A_nrows, A_ncols, A_nvals;
    Index B_nrows, B_ncols;
    Index C_nrows, C_ncols;

    A.nrows( A_nrows );
    A.ncols( A_ncols );
    A.nvals( A_nvals );
    B.nrows( B_nrows );
    B.ncols( B_ncols );
    C.nrows( C_nrows );
    C.ncols( C_ncols );

    //std::cout << "A: " << A_nrows << " " << A_ncols << std::endl;
    //std::cout << "B: " << B_nrows << " " << B_ncols << std::endl;
    //std::cout << "C: " << C_nrows << " " << C_ncols << std::endl;

    // Dimension compatibility check
    if( (A_ncols != B_nrows) || (C_ncols != B_ncols) || (C_nrows != A_nrows ) )
    {
      std::cout << "Dim mismatch" << std::endl;
      std::cout << A_ncols << " " << B_nrows << std::endl;
      std::cout << C_ncols << " " << B_ncols << std::endl;
      std::cout << C_nrows << " " << A_nrows << std::endl;
      return GrB_DIMENSION_MISMATCH;
    }

    // Domain compatibility check
    // TODO: add domain compatibility check

    // Temporarily for testing purposes
    //C.allocate();
    Desc_value tb, nt;
    desc.get( GrB_TB  , tb );
    desc.get( GrB_NT  , nt );

    // Computation
    const int TB       = static_cast<int>(tb);
    const int NT       = static_cast<int>(nt);

    // Computation
    //std::cout << "Success creating mgpu context\n";
    mgpu::SpmmCsrBinary( A.d_csrVal_, A.d_csrColInd_, A_nvals, A.d_csrRowPtr_, 
        A_nrows, B.d_denseVal_, false, C.d_denseVal_, (c) 0, 
        mgpu::multiplies<c>(), mgpu::plus<c>(), B_ncols, desc.d_limits_, 
        desc.d_carryin_, desc.d_carryout_, TB, NT, *desc.d_context_ );
    //std::cout << "Finished SpmmCsrBinary\n";

    C.need_update_ = true;  // Set flag that we need to copy data from GPU
    return GrB_SUCCESS;
  }

}  // backend
}  // graphblas

#endif  // GRB_BACKEND_APSPIE_SPMM_HPP
