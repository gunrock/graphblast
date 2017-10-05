#define GRB_USE_APSPIE
//#define private public

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE spgemm_suite

#include <iostream>
#include <string>

#include <cstdio>
#include <cstdlib>

#include <cuda_profiler_api.h>

#include "graphblas/mmio.hpp"
#include "graphblas/util.hpp"
#include "graphblas/graphblas.hpp"

#include <boost/test/included/unit_test.hpp>
#include <boost/program_options.hpp>
#include <test/test.hpp>

struct TestSPGEMM {
  TestSPGEMM() :
    TA(32),
    TB(32),
    NT(64),
    ROW_MAJOR(true),
    DEBUG(true) {}

  int TA, TB, NT;
  bool ROW_MAJOR, DEBUG;
};

BOOST_FIXTURE_TEST_SUITE(spgemm_suite, TestSPGEMM)

// SpGEMM unit test (C=test_cc*test_cc)
BOOST_AUTO_TEST_CASE( spgemm1 )
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
  graphblas::Matrix<float> b(11, 11);
  graphblas::Index nvals = 20;
  graphblas::Index nrows, ncols;
  a.build( row_indices, col_indices, values, 20 );
  a.nrows( nrows );
  a.ncols( ncols );
  a.nvals( nvals );
  BOOST_ASSERT( nrows==11 );
  BOOST_ASSERT( ncols==11 );
  BOOST_ASSERT( nvals==20 );
  if( DEBUG ) a.print();
  b.build( row_indices, col_indices, values, 20 );
  b.nrows( nrows );
  b.ncols( ncols );
  b.nvals( nvals );
  BOOST_ASSERT( nrows==11 );
  BOOST_ASSERT( ncols==11 );
  BOOST_ASSERT( nvals==20 );
  graphblas::Matrix<float> c(11, 11);
  graphblas::Semiring op;

  graphblas::GpuTimer gpu_mxm, gpu_mxm2;
  gpu_mxm.Start();
  graphblas::mxm<float, float, float>( c, op, a, b, TA, TB, NT, ROW_MAJOR );
  gpu_mxm.Stop();
  gpu_mxm2.Start();
  graphblas::mxm<float, float, float>( c, op, a, b, TA, TB, NT, ROW_MAJOR );
  gpu_mxm2.Stop();
  float elapsed_mxm = gpu_mxm.ElapsedMillis();
  if( DEBUG ) std::cout << "mxm: " << elapsed_mxm << " ms\n";
  elapsed_mxm = gpu_mxm2.ElapsedMillis();
  if( DEBUG ) std::cout << "mxm: " << elapsed_mxm << " ms\n";

  if( DEBUG ) c.print();
  std::vector<float> out_val;
  c.extractTuples( out_val );
  for( int i=0; i<20; i++ ) {
    graphblas::Index row = row_indices[i];
    graphblas::Index col = col_indices[i];
    float            val = values[i];
    // Row Major layout
    /*if( ROW_MAJOR )
    //std::cout << row << " " << col << " " << val << " " << out_val[row*11+col] << std::endl;
    BOOST_ASSERT( val==out_val[row*11+col] );
    else
    // Column Major layout
    //std::cout << row << " " << col << " " << val << " " << out_val[col*11+row] << std::endl;
    BOOST_ASSERT( val==out_val[col*11+row] );*/
  }
}

// SpGEMM unit test (chesapeake)
BOOST_AUTO_TEST_CASE( spgemm2 )
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
  graphblas::Matrix<float> b(nrows, ncols);
  a.build( row_indices, col_indices, values, nvals );
  a.nrows( nrows );
  a.ncols( ncols );
  a.nvals( nvals );
  if( DEBUG ) a.print();
  BOOST_ASSERT( nrows==39 );
  BOOST_ASSERT( ncols==39 );
  BOOST_ASSERT( nvals==340 );

  b.build( row_indices, col_indices, values, nvals );
  b.nrows( nrows );
  b.ncols( ncols );
  b.nvals( nvals );
  if( DEBUG ) b.print();
  BOOST_ASSERT( nrows==39 );
  BOOST_ASSERT( ncols==39 );
  BOOST_ASSERT( nvals==340 );

  graphblas::Matrix<float> c(nrows, ncols);
  graphblas::Semiring op;

  graphblas::GpuTimer gpu_mxm;
  gpu_mxm.Start();
  graphblas::mxm<float, float, float>( c, op, a, b, TA, TB, NT, ROW_MAJOR );
  gpu_mxm.Stop();
  float elapsed_mxm = gpu_mxm.ElapsedMillis();
  if( DEBUG ) std::cout << "mxm: " << elapsed_mxm << " ms\n";

  std::vector<float> out_val;
  if( DEBUG ) c.print();
  c.extractTuples( out_val );
  for( int i=0; i<nvals; i++ ) {
    graphblas::Index row = row_indices[i];
    graphblas::Index col = col_indices[i];
    float            val = values[i];
    // Row Major layout
    /*if( ROW_MAJOR )
    //std::cout << row << " " << col << " " << val << " " << out_val[row*ncols+col] << std::endl;
    BOOST_ASSERT( val==out_val[row*ncols+col] );
    else
    // Column Major layout
    //std::cout << row << " " << col << " " << val << " " << out_val[col*nrows+row] << std::endl;
    BOOST_ASSERT( val==out_val[col*nrows+row] );*/
  }
}

BOOST_AUTO_TEST_CASE( spgemm3 )
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
  if( DEBUG ) a.print();

  // Matrix B
  graphblas::Matrix<float> b(nrows, ncols);
  b.build( row_indices, col_indices, values, nvals );

  graphblas::Matrix<float> c(nrows, ncols);
  graphblas::Semiring op;

  graphblas::GpuTimer gpu_mxm;
  gpu_mxm.Start();
  graphblas::mxm<float, float, float>( c, op, a, b, TA, TB, NT, ROW_MAJOR );
  gpu_mxm.Stop();
  float elapsed_mxm = gpu_mxm.ElapsedMillis();
  if( DEBUG ) std::cout << "mxm: " << elapsed_mxm << " ms\n";

  std::vector<float> out_val;
  if( DEBUG ) c.print();
  c.extractTuples( out_val );
  for( int i=0; i<nvals; i++ ) {
    graphblas::Index row = row_indices[i];
    graphblas::Index col = col_indices[i];
    float            val = values[i];
    /*if( col<max_ncols ) {
      // Row major order
      if( ROW_MAJOR ) {
        //std::cout << row << " " << col << " " << val << " " << out_val[row*max_ncols+col] << std::endl;
        BOOST_ASSERT( val==out_val[row*max_ncols+col] );
      } else
      // Column major order
      //std::cout << row << " " << col << " " << val << " " << out_val[col*nrows+row] << std::endl;
        BOOST_ASSERT( val==out_val[col*nrows+row] );
    }*/
  }
}

BOOST_AUTO_TEST_SUITE_END()
