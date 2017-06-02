#define GRB_USE_APSPIE
//#define private public

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
    TB(32),
    NT(64),
    ROW_MAJOR(true),
    DEBUG(true) {}

  int TA, TB, NT;
  bool ROW_MAJOR, DEBUG;
};

BOOST_AUTO_TEST_SUITE(spmm_suite)

// SpMM unit test (test_cc)
/*BOOST_FIXTURE_TEST_CASE( spmm1, TestSPMM )
{
  if( DEBUG ) {
    std::cout << "ta:    " << TA        << "\n";
    std::cout << "tb:    " << TB        << "\n";
    std::cout << "nt:    " << NT        << "\n";
    std::cout << "row:   " << ROW_MAJOR << "\n";
    std::cout << "debug: " << DEBUG     << "\n";
  }
  std::vector<graphblas::Index> row_indices = {0, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 5, 6, 6, 6, 8, 9, 9, 10, 10};
  std::vector<graphblas::Index> col_indices = {1, 0, 0, 1, 4, 0, 2, 1, 2, 2, 3, 4, 3, 4, 5, 7, 7, 8, 7, 8};
  std::vector<float> values (20, 1.0);
  graphblas::Matrix<float> a(11, 11);
  graphblas::Index nvals = 20;
  graphblas::Index nrows, ncols;
  a.build( row_indices, col_indices, values, 20 );
  a.nrows( nrows );
  a.ncols( ncols );
  a.nvals( nvals );
  BOOST_ASSERT( nrows==11 );
  BOOST_ASSERT( ncols==11 );
  BOOST_ASSERT( nvals==20 );
  a.print();
  
  graphblas::Index MEM_SIZE = 1000000000;  // 2x4=8GB GPU memory for dense
  graphblas::Index max_ncols = std::min( MEM_SIZE/nrows/32*32, ncols );
  if( ncols%32!=0 && max_ncols%32!=0 ) max_ncols = (ncols+31)/32*32;
  if( DEBUG && max_ncols!=ncols ) std::cout << "Restricting col to: " 
      << max_ncols << std::endl;
  graphblas::Matrix<float> b(11, max_ncols);

  std::vector<float> denseVal;
  if( ROW_MAJOR )
    for( int i=0; i<11; i++ ) {
      for( int j=0; j<max_ncols; j++ ) {
        if( i==j ) denseVal.push_back(1.0);
        else denseVal.push_back(0.0);
      }
    }
  else
    for( int i=0; i<max_ncols; i++ ) {
      for( int j=0; j<11; j++ ) {
        if( i==j ) denseVal.push_back(1.0);
        else denseVal.push_back(0.0);
      }
    }
    
  b.build( denseVal );
  graphblas::Matrix<float> c(11, max_ncols);
  graphblas::Semiring op;

  GpuTimer gpu_mxm;
  gpu_mxm.Start();
  graphblas::mxm<float, float, float>( c, op, a, b, TA, TB, NT, ROW_MAJOR );
  gpu_mxm.Stop();
  float elapsed_mxm = gpu_mxm.ElapsedMillis();
  std::cout << "mxm: " << elapsed_mxm << " ms\n";

  c.print();
  std::vector<float> out_denseVal;
  c.extractTuples( out_denseVal );
  for( int i=0; i<20; i++ ) {
    graphblas::Index row = row_indices[i];
    graphblas::Index col = col_indices[i];
    float            val = values[i];
    // Row Major layout
    if( ROW_MAJOR ) {
      //std::cout << row << " " << col << " " << val << " " << out_denseVal[row*max_ncols+col] << std::endl;
      BOOST_ASSERT( val==out_denseVal[row*max_ncols+col] );
    } else {
    // Column Major layout
    //std::cout << row << " " << col << " " << val << " " << out_denseVal[col*nrows+row] << std::endl;
      BOOST_ASSERT( val==out_denseVal[col*11+row] );
    }
  }
}

// SpMM unit test (chesapeake)
BOOST_FIXTURE_TEST_CASE( spmm2, TestSPMM )
{
  if( DEBUG ) {
    std::cout << "ta:    " << TA        << "\n";
    std::cout << "tb:    " << TB        << "\n";
    std::cout << "nt:    " << NT        << "\n";
    std::cout << "row:   " << ROW_MAJOR << "\n";
    std::cout << "debug: " << DEBUG     << "\n";
  }
  std::vector<graphblas::Index> row_indices;
  std::vector<graphblas::Index> col_indices;
  std::vector<float> values;
  graphblas::Index nrows, ncols, nvals;

  // Read in chesapeake.mtx
  char const *argv = "../dataset/small/chesapeake.mtx";
  readMtx( argv, row_indices, col_indices, values, nrows, ncols, nvals, DEBUG );

  graphblas::Matrix<float> a(nrows, ncols);
  a.build( row_indices, col_indices, values, nvals );
  a.nrows( nrows );
  a.ncols( ncols );
  a.nvals( nvals );
  a.print();
  BOOST_ASSERT( nrows==39 );
  BOOST_ASSERT( ncols==39 );
  BOOST_ASSERT( nvals==340 );
  std::vector<float> denseVal;

  graphblas::Index MEM_SIZE = 1000000000;  // 2x4=8GB GPU memory for dense
  graphblas::Index max_ncols = std::min( MEM_SIZE/nrows/32*32, ncols );
  if( ncols%32!=0 && max_ncols%32!=0 ) max_ncols = (ncols+31)/32*32;
  if( DEBUG && max_ncols!=ncols ) std::cout << "Restricting col to: " 
      << max_ncols << std::endl;

  graphblas::Matrix<float> b(nrows, max_ncols);
  for( int i=0; i<nrows; i++ ) {
    for( int j=0; j<max_ncols; j++ ) {
      if( i==j ) denseVal.push_back(1.0);
      else denseVal.push_back(0.0);
    }
  }
  b.build( denseVal );
  graphblas::Matrix<float> c(nrows, max_ncols);
  graphblas::Semiring op;

  GpuTimer gpu_mxm;
  gpu_mxm.Start();
  graphblas::mxm<float, float, float>( c, op, a, b, TA, TB, NT, ROW_MAJOR );
  gpu_mxm.Stop();
  float elapsed_mxm = gpu_mxm.ElapsedMillis();
  std::cout << "mxm: " << elapsed_mxm << " ms\n";

  std::vector<float> out_denseVal;
  c.print();
  c.extractTuples( out_denseVal );
  for( int i=0; i<nvals; i++ ) {
    graphblas::Index row = row_indices[i];
    graphblas::Index col = col_indices[i];
    float            val = values[i];
    // Row Major layout
    if( ROW_MAJOR ) {
      //std::cout << row << " " << col << " " << val << " " << out_denseVal[row*max_ncols+col] << std::endl;
      BOOST_ASSERT( val==out_denseVal[row*max_ncols+col] );
    } else {
    // Column Major layout
    //std::cout << row << " " << col << " " << val << " " << out_denseVal[col*nrows+row] << std::endl;
      BOOST_ASSERT( val==out_denseVal[col*nrows+row] );
    }
  }
}*/

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

  char const *argv = "/data-2/gunrock_dataset/large/ak2010/ak2010.mtx";
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
  graphblas::Index max_ncols = std::min( MEM_SIZE/nrows/32*32, ncols );
  if( ncols%32!=0 && max_ncols%32!=0 ) max_ncols = (ncols+31)/32*32;
  if( DEBUG && max_ncols!=ncols ) std::cout << "Restricting col to: " 
      << max_ncols << std::endl;

  graphblas::Matrix<float> b(nrows, max_ncols);
  std::vector<float> denseVal;

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

  GpuTimer gpu_mxm;
  gpu_mxm.Start();
  graphblas::mxm<float, float, float>( c, op, a, b, TA, TB, NT, ROW_MAJOR );
  gpu_mxm.Stop();
  float elapsed_mxm = gpu_mxm.ElapsedMillis();
  std::cout << "mxm: " << elapsed_mxm << " ms\n";

  std::vector<float> out_denseVal;
  /*if( DEBUG ) c.print();
  c.extractTuples( out_denseVal );
  for( int i=0; i<nvals; i++ ) {
    graphblas::Index row = row_indices[i];
    graphblas::Index col = col_indices[i];
    float            val = values[i];
    if( col<max_ncols ) {
      // Row major order
      if( ROW_MAJOR ) {
        //std::cout << row << " " << col << " " << val << " " << out_denseVal[row*max_ncols+col] << std::endl;
        BOOST_ASSERT( val==out_denseVal[row*max_ncols+col] );
      } else
      // Column major order
      //std::cout << row << " " << col << " " << val << " " << out_denseVal[col*nrows+row] << std::endl;
        BOOST_ASSERT( val==out_denseVal[col*nrows+row] );
    }
  }*/
}

BOOST_AUTO_TEST_SUITE_END()
