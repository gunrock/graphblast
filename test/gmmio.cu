#define GRB_USE_APSPIE
//#define GRB_USE_SEQUENTIAL

#include <vector>
#include <iostream>
#include <cstdio>
#include <cstdlib>

#include <graphblas/graphblas.hpp>

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE mmio_suite

#include <boost/test/included/unit_test.hpp>
#include <test/test.hpp>

BOOST_AUTO_TEST_SUITE( mmio_suite )

// test_cc.mtx
BOOST_AUTO_TEST_CASE( mmio1 )
{
  std::vector<graphblas::Index> row_indices;
  std::vector<graphblas::Index> col_indices;
  std::vector<float> values;
  graphblas::Index nrows;
  graphblas::Index ncols;
  graphblas::Index nvals;

  // Read in test_cc.mtx
  char const *argv = "dataset/small/test_cc.mtx";
  readMtx( argv, row_indices, col_indices, values, nrows, ncols, nvals, true );

  int rhs[12] = {0, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 5};
  //for( int i=0; i<12; i++ ) std::cout << row_indices[i] << "\n";
  BOOST_ASSERT_LIST( row_indices, rhs, 12 );
}

// chesapeake.mtx
BOOST_AUTO_TEST_CASE( mmio2 )
{
  std::vector<graphblas::Index> row_indices;
  std::vector<graphblas::Index> col_indices;
  std::vector<float> values;
  graphblas::Index nrows;
  graphblas::Index ncols;
  graphblas::Index nvals;

  // Read in chesapeake.mtx
  char const *argv = "dataset/small/chesapeake.mtx";
  readMtx( argv, row_indices, col_indices, values, nrows, ncols, nvals, true );

  int rhs[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1};
  //for( int i=0; i<12; i++ ) std::cout << row_indices[i] << "\n";
  BOOST_ASSERT_LIST( row_indices, rhs, 12 );
}

BOOST_AUTO_TEST_SUITE_END()
