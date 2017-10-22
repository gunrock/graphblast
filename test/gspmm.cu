#define GRB_USE_APSPIE
#define private public

#include <iostream>
#include <algorithm>
#include <string>

#include <cstdio>
#include <cstdlib>
#include <cuda_profiler_api.h>

#include "graphblas/mmio.hpp"
#include "graphblas/util.hpp"
#include "graphblas/graphblas.hpp"

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE spmm_suite

#include <boost/test/included/unit_test.hpp>
#include <boost/program_options.hpp>
#include <test/test.hpp>

struct TestSPMM {
  TestSPMM() :
    TA(32),
    TB(8),
    NT(256),
    ROW_MAJOR(true),
    DEBUG(true) {}

  int TA, TB, NT;
  bool ROW_MAJOR, DEBUG;
};

BOOST_AUTO_TEST_SUITE(spmm_suite)

BOOST_FIXTURE_TEST_CASE( spmm3, TestSPMM )
{
  std::vector<graphblas::Index> row_indices;
  std::vector<graphblas::Index> col_indices;
  std::vector<float> values;
  graphblas::Index nrows, ncols, nvals;

  if( DEBUG ) {
    std::cout << "ta:    " << TA        << "\n";
    std::cout << "tb:    " << TB        << "\n";
    std::cout << "nt:    " << NT        << "\n";
    std::cout << "row:   " << ROW_MAJOR << "\n";
    std::cout << "debug: " << DEBUG     << "\n";
  }

  //char const *argv = "dataset/small/test_bc.mtx";
  //char const *argv = "/home/ctcyang/GraphBLAS/dataset/small/chesapeake.mtx";
  //char const *argv = "/data-2/gunrock_dataset/large/delaunay_n10/delaunay_n10.mtx";
  //char const *argv = "/data-2/gunrock_dataset/large/benchmark2/12month1/12month1.mtx";
  char const *argv = "/data-2/gunrock_dataset/large/benchmark/ASIC_320k/ASIC_320k.mtx";
  //char const *argv = "/home/ctcyang/GraphBLAS/dataset/large/ASIC_320k/ASIC_320k.mtx";
  readMtx( argv, row_indices, col_indices, values, nrows, ncols, nvals, DEBUG );

  // Matrix A
  graphblas::Matrix<float> a(nrows, ncols);
  a.build( row_indices, col_indices, values, nvals );
  a.nrows( nrows );
  a.ncols( ncols );
  a.nvals( nvals );
  if( DEBUG ) a.print();

  // Matrix B
  graphblas::Index MEM_SIZE = 1000000000;  // 2x4=8GB GPU memory for dense
  graphblas::Index max_ncols = 64;//std::min( MEM_SIZE/nrows/32*32, ncols );
  if( ncols%32!=0 && max_ncols%32!=0 ) max_ncols = (ncols+31)/32*32;
  if( DEBUG && max_ncols!=ncols ) std::cout << "Restricting col to: "
      << max_ncols << std::endl;

  graphblas::Matrix<float> b(ncols, max_ncols);
  std::vector<float> denseVal;

  // default values of TA, TB, NT will be used
  graphblas::Descriptor desc;
  desc.set( graphblas::GrB_MODE, graphblas::GrB_MERGEPATH );
  //desc.set( graphblas::GrB_MODE, graphblas::GrB_FIXEDROW );
  //desc.set( graphblas::GrB_MODE, graphblas::GrB_CUSPARSE2 );
  desc.set( graphblas::GrB_NT, NT );
  desc.set( graphblas::GrB_TA, TA );
  desc.set( graphblas::GrB_TB, TB );

  graphblas::Index a_nvals;
  a.nvals( a_nvals );
  int num_blocks = (a_nvals+NT-1)/NT;
  int num_segreduce = (num_blocks + NT - 1)/NT;
  CUDA( cudaMalloc( &desc.descriptor_.d_limits_,
      (num_blocks+1)*sizeof(graphblas::Index) ));
  CUDA( cudaMalloc( &desc.descriptor_.d_carryin_,
      num_blocks*max_ncols*sizeof(float) ));
  CUDA( cudaMalloc( &desc.descriptor_.d_carryout_,
      num_segreduce*sizeof(float)      ));

  // Row major order
  if( ROW_MAJOR )
    for( int i=0; i<nrows; i++ )
      for( int j=0; j<max_ncols; j++ ) {
        if( i==j ) denseVal.push_back(1.0);
        else denseVal.push_back(0.0);
      }
  else
  // Column major order
    for( int i=0; i<max_ncols; i++ )
      for( int j=0; j<nrows; j++ ) {
        //denseVal.push_back(1.0);
        if( i==j ) denseVal.push_back(1.0);
        else denseVal.push_back(0.0);
      }
  b.build( denseVal );
  graphblas::Matrix<float> c(nrows, max_ncols);
  graphblas::Semiring op;

  graphblas::GpuTimer gpu_mxm;
  cudaProfilerStart();
  gpu_mxm.Start();
  graphblas::mxm<float, float, float>( c, graphblas::GrB_NULL, graphblas::GrB_NULL, op, a, b, desc );
  gpu_mxm.Stop();
  cudaProfilerStop();
  float elapsed_mxm = gpu_mxm.ElapsedMillis();
  std::cout << "mxm: " << elapsed_mxm << " ms\n";
  //ROW_MAJOR=false;

  std::vector<float> out_denseVal;
  if( DEBUG ) c.print();
  c.extractTuples( out_denseVal );
  int count = 0, correct=0;
  for( int i=0; i<nvals; i++ ) {
    graphblas::Index row = row_indices[i];
    graphblas::Index col = col_indices[i];
    float            val = values[i];
    if( col<max_ncols ) {
      count++;
      // Row major order
      if( ROW_MAJOR ) {
        if( val!=out_denseVal[row*max_ncols+col] )
        {
          std::cout << row << " " << col << " " << val << " " << out_denseVal[row*max_ncols+col] << std::endl;
          correct++;
        }
        //BOOST_ASSERT( val==out_denseVal[row*max_ncols+col] );
      } else
      // Column major order
      //std::cout << row << " " << col << " " << val << " " << out_denseVal[col*nrows+row] << std::endl;
        BOOST_ASSERT( val==out_denseVal[col*nrows+row] );
    }
  }
  std::cout << "There were " << correct << " errors out of " << count << ".\n";
}

BOOST_AUTO_TEST_SUITE_END()
