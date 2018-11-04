#define GRB_USE_APSPIE
#define private public

#include <iostream>
#include <algorithm>
#include <string>

#include <cstdio>
#include <cstdlib>

#include "graphblas/graphblas.hpp"
#include "test/test.hpp"

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE dup_suite

#include <boost/test/included/unit_test.hpp>
#include <boost/program_options.hpp>

void testVxm( char const*               mtx,
              const std::vector<float>& vec,
              const std::vector<float>& correct,
              po::variables_map&        vm )
{
  std::vector<graphblas::Index> row_indices;
  std::vector<graphblas::Index> col_indices;
  std::vector<float> values;
  graphblas::Index nrows, ncols, nvals;
  char* dat_name;

  // Read in sparse matrix
  readMtx(mtx, row_indices, col_indices, values, nrows, ncols, 
      nvals, 0, false, &dat_name);

  // Matrix A
  graphblas::Matrix<float> a(nrows, ncols);
  a.build(&row_indices, &col_indices, &values, nvals, GrB_NULL, dat_name);
  a.nrows(&nrows);
  a.ncols(&ncols);
  a.nvals(&nvals);
  a.print();

  // Vector x
  graphblas::Vector<float> x(nrows);
  x.build(&vec, vec.size());
  x.print();

  // Vector y
  graphblas::Vector<float> y(nrows);

  // Descriptor
  graphblas::Descriptor desc;
  desc.loadArgs(vm);
  //desc.set(graphblas::GrB_MXVMODE, graphblas::GrB_PUSHONLY) );

  // Compute
  graphblas::vxm<float, float, float, float>(&y, GrB_NULL, GrB_NULL, 
      graphblas::PlusMultipliesSemiring<float>(), &x, &a, &desc);

  y.print();
  y.vector_.sparse2dense(0.f, &desc.descriptor_);
  y.print();
  y.extractTuples( &values, &nrows );
  BOOST_ASSERT( nrows == correct.size() );
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
  int argc = 3;
  char* argv[] = {"app", "--debug", "1"};
  po::variables_map vm;
  parseArgs( argc, argv, vm );
  std::vector<float> vec(11, 1.f);
  std::vector<float> correct{ 3., 3., 3., 2., 3., 1., 0., 3., 2., 0., 0. };
  testVxm( "data/small/test_cc.mtx", vec, correct, vm );
}

BOOST_FIXTURE_TEST_CASE( dup2, TestMatrix )
{
  int argc = 3;
  char* argv[] = {"app", "--debug", "1"};
  po::variables_map vm;
  parseArgs( argc, argv, vm );
  std::vector<float> vec(11, 2.f);
  std::vector<float> correct{ 6., 6., 6., 4., 6., 2., 0., 6., 4., 0., 0. };
  testVxm( "data/small/test_cc.mtx", vec, correct, vm );
}

BOOST_FIXTURE_TEST_CASE( dup3, TestMatrix )
{
  int argc = 3;
  char* argv[] = {"app", "--debug", "1"};
  po::variables_map vm;
  parseArgs( argc, argv, vm );
  std::vector<float>     vec{0, 13,7, 0, 1, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 4, 4, 5, 4};
  std::vector<float> correct{0, 13,7, 0, 1, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 4, 4, 5, 4};
  testVxm( "data/small/test_sgm.mtx", vec, correct, vm );
}
BOOST_AUTO_TEST_SUITE_END()
