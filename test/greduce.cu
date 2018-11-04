#define GRB_USE_APSPIE
#define private public

#include <vector>
#include <iostream>
#include <string>
#include <numeric>
#include <algorithm>
#include <map>

#include <cstdio>
#include <cstdlib>

#include "graphblas/graphblas.hpp"
#include "test/test.hpp"

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE dup_suite

#include <boost/test/included/unit_test.hpp>
#include <boost/program_options.hpp>

void testReduce( char const* mtx,
                 const std::vector<float>& correct )
{
  std::vector<graphblas::Index> row_indices;
  std::vector<graphblas::Index> col_indices;
  std::vector<float> values;
  graphblas::Index nrows, ncols, nvals;
  graphblas::Info err;
  graphblas::Descriptor desc;
  char* dat_name;

  // Read in sparse matrix
  readMtx(mtx, row_indices, col_indices, values, nrows, ncols, nvals, 0, false,
      &dat_name);

  graphblas::Matrix<float> adj(nrows, ncols);
  err = adj.build(&row_indices, &col_indices, &values, nvals, GrB_NULL,
      dat_name);
  adj.print();

  std::cout << nrows << " " << ncols << " " << nvals << std::endl;
  graphblas::Vector<float> vec(nrows);

  err = graphblas::reduce<float,float,float>( &vec, GrB_NULL, GrB_NULL, graphblas::PlusMonoid<float>(), &adj, &desc );

  graphblas::Index nrows_t = nrows;
  err = vec.print();
  err = vec.extractTuples( &values, &nrows_t );
  BOOST_ASSERT( nrows == nrows_t );
  BOOST_ASSERT_LIST( values, correct, nrows );
}

struct TestMatrix
{
  TestMatrix() :
    DEBUG(true) {}

  bool DEBUG;
};

BOOST_AUTO_TEST_SUITE(dup_suite)

BOOST_FIXTURE_TEST_CASE( dup1, TestMatrix )
{
  std::vector<float> correct{   1., 1., 3., 2., 2., 3., 3., 0., 1., 2., 2. };
  std::vector<float> correct_t{ 3., 3., 3., 2., 3., 1., 0., 3., 2., 0., 0. };
  testReduce( "data/small/test_cc.mtx", correct );
}

BOOST_FIXTURE_TEST_CASE( dup2, TestMatrix )
{
  std::vector<float> correct{   1., 1., 3., 2., 2., 3., 3. };
  std::vector<float> correct_t{ 3., 3., 3., 2., 3., 1., 0. };
  testReduce( "data/small/test_bc.mtx", correct );
}

BOOST_AUTO_TEST_SUITE_END()
