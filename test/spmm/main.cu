#define GRB_USE_APSPIE
#define COL_MAJOR
//#define private public

#include <iostream>
#include <algorithm>

#include <cstdio>
#include <cstdlib>
#include <cuda_profiler_api.h>

#include "graphblas/mmio.hpp"
#include "graphblas/util.hpp"
#include "graphblas/graphblas.hpp"

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE spmm_suite

#include <boost/test/included/unit_test.hpp>
#include <test/test.hpp>

struct TestSPMM {
  TestSPMM() :
    argc( boost::unit_test::framework::master_test_suite().argc ),
    argv( boost::unit_test::framework::master_test_suite().argv ) {}

  int argc;
  char **argv;
};

BOOST_AUTO_TEST_SUITE(spmm_suite)

BOOST_FIXTURE_TEST_CASE( spmm1, TestSPMM )
{
  std::vector<graphblas::Index> row_indices;
  std::vector<graphblas::Index> col_indices;
  std::vector<float> values;
	graphblas::Index nrows, ncols, nvals;

	// Read in sparse matrix
  if (argc < 2) {
    fprintf(stderr, "Usage: %s [matrix-market-filename]\n", argv[0]);
    exit(1);
  } else { 
	  readMtx( argv[1], row_indices, col_indices, values, nrows, ncols, nvals );
  }
  //printArray( "row_indices", row_indices );
  //printArray( "col_indices", col_indices );

  graphblas::Matrix<float> a(nrows, ncols);

  graphblas::Index MEM_SIZE = 1000000000;  // 2x4=8GB GPU memory for dense
  graphblas::Index max_ncols = std::min( MEM_SIZE/nrows, ncols );
  if( max_ncols<ncols ) std::cout << "Restricting col to: " << max_ncols <<
      std::endl;

  graphblas::Matrix<float> b(nrows, max_ncols);
  a.build( row_indices, col_indices, values, nvals );
  a.nrows( nrows );
  a.ncols( ncols );
  a.nvals( nvals );
  a.print();
  std::vector<float> denseVal;

  // Row major order
  #ifdef ROW_MAJOR
  for( int i=0; i<nrows; i++ ) {
    for( int j=0; j<max_ncols; j++ ) {
      if( i==j ) denseVal.push_back(1.0);
      else denseVal.push_back(0.0);
    }
  }
  #endif
  // Column major order
  #ifdef COL_MAJOR
  for( int i=0; i<max_ncols; i++ ) {
    for( int j=0; j<nrows; j++ ) {
      denseVal.push_back(1.0);
      //if( i==j ) denseVal.push_back(1.0);
      //else denseVal.push_back(0.0);
    }
  }
  #endif
  b.build( denseVal );
  graphblas::Matrix<float> c(nrows, max_ncols);
  graphblas::Semiring op;

  cudaProfilerStart();
  graphblas::mxm<float, float, float>( c, op, a, b );
  cudaProfilerStop();

  std::vector<float> out_denseVal;
  c.print();
  c.extractTuples( out_denseVal );
  for( int i=0; i<nvals; i++ ) {
    graphblas::Index row = row_indices[i];
    graphblas::Index col = col_indices[i];
    float            val = values[i];
    if( col<max_ncols ) {
      // Row major order
      #ifdef ROW_MAJOR
      //std::cout << row << " " << col << " " << val << " " << out_denseVal[row*max_ncols+col] << std::endl;
      BOOST_ASSERT( val==out_denseVal[row*max_ncols+col] );
      #endif
      // Column major order
      #ifdef COL_MAJOR
      //std::cout << row << " " << col << " " << val << " " << out_denseVal[col*nrows+row] << std::endl;
      BOOST_ASSERT( val==out_denseVal[col*nrows+row] );
      #endif
    }
}}

BOOST_AUTO_TEST_SUITE_END()
