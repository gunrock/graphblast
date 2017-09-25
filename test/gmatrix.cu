//#define GRB_USE_SEQUENTIAL
#define GRB_USE_APSPIE
//#define private public

#include <iostream>
#include <random>
#include <algorithm>

#include <cstdio>
#include <cstdlib>
#include <cuda_profiler_api.h>

#include "graphblas/graphblas.hpp"

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE matrix_suite

#include <boost/test/included/unit_test.hpp>
#include "test/test.hpp"

BOOST_AUTO_TEST_SUITE(matrix_suite)

BOOST_AUTO_TEST_CASE( matrix1 )
{
  std::vector<graphblas::Index> row_indices = {0, 1, 2};
  std::vector<graphblas::Index> col_indices = {1, 1, 1};
  std::vector<float> values = {1.0, 2.0, 3.0};
  graphblas::Matrix<float> a(3, 3);
  a.build( row_indices, col_indices, values, 3 );
  std::vector<graphblas::Index> row;
  std::vector<graphblas::Index> col;
  std::vector<float> val;
  a.extractTuples( row, col, val );
  BOOST_ASSERT_LIST( row_indices, row );
  BOOST_ASSERT_LIST( col_indices, col );
  BOOST_ASSERT_LIST( values, val );
}

// SpGEMM unit test
// Assert: error: out of dimension tuple passed into build
BOOST_AUTO_TEST_CASE( matrix2 )
{
  std::vector<graphblas::Index> row_indices = {2, 3, 4, 1, 3, 5, 4, 5, 6, 6, 7, 3, 6, 7, 7, 9, 10, 11, 10, 11};
  std::vector<graphblas::Index> col_indices = {1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 6, 8, 8, 8, 9, 9};
  std::vector<float> values (20, 1.0);
  graphblas::Matrix<float> a(11, 11);
  graphblas::Matrix<float> b(11, 11);
  graphblas::Info err = a.build( row_indices, col_indices, values, 20 );
  BOOST_ASSERT( err == graphblas::GrB_INDEX_OUT_OF_BOUNDS );
}

// SpMM unit test
// Assert: error out of dimension tuple passed into build
BOOST_AUTO_TEST_CASE( matrix3 )
{
  std::vector<graphblas::Index> row_indices = {2, 3, 4, 1, 3, 5, 4, 5, 6, 6, 7, 3, 6, 7, 7, 9, 10, 11, 10, 11};
  std::vector<graphblas::Index> col_indices = {1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 6, 8, 8, 8, 9, 9};
  std::vector<float> values (20, 1.0);
  graphblas::Matrix<float> a(11, 11);
  graphblas::Matrix<float> b(11, 11);
  graphblas::Info err = a.build( row_indices, col_indices, values, 20 );
  BOOST_ASSERT( err == graphblas::GrB_INDEX_OUT_OF_BOUNDS );
}

BOOST_AUTO_TEST_SUITE_END() 
