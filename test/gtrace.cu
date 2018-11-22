#define GRB_USE_CUDA
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

void testTrace( const std::vector<graphblas::Index>& row_ind,
                const std::vector<graphblas::Index>& col_ind,
                const std::vector<float>& values, 
                graphblas::Index nrows, float correct )
{
  graphblas::Index ncols = nrows;
  graphblas::Index nvals = row_ind.size();
  graphblas::Info err;
  graphblas::Descriptor desc;
  char* dat_name;

  // Read in sparse matrix
  graphblas::Matrix<float> adj(nrows, ncols);
  err = adj.build(&row_ind, &col_ind, &values, nvals, GrB_NULL, dat_name);
  adj.print();

  std::cout << nrows << " " << ncols << " " << nvals << std::endl;

  float trace_val;
  err = graphblas::traceMxmTranspose<float,float,float>( &trace_val, 
      graphblas::PlusMultipliesSemiring<float>(), &adj, &adj, &desc );
  std::cout << trace_val << " = " << correct << std::endl;

  BOOST_ASSERT( trace_val == correct );
}

void testTraceMtx( char const* mtx, float correct )
{
  std::vector<graphblas::Index> row_indices;
  std::vector<graphblas::Index> col_indices;
  std::vector<float> values;
  graphblas::Index nrows, ncols, nvals;
  graphblas::Info err;
  graphblas::Descriptor desc;

  // Read in sparse matrix
  readMtx(mtx, row_indices, col_indices, values, nrows, ncols, nvals, 0, true);

  graphblas::Matrix<float> adj(nrows, ncols);
  err = adj.build( &row_indices, &col_indices, &values, nvals, GrB_NULL );
  adj.print();

  std::cout << nrows << " " << ncols << " " << nvals << std::endl;

  float trace_val;
  err = graphblas::traceMxmTranspose<float,float,float>( &trace_val,
      graphblas::PlusMultipliesSemiring<float>(), &adj, &adj, &desc );
  std::cout << trace_val << " = " << correct << std::endl;

  BOOST_ASSERT( trace_val == correct );
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
  std::vector<graphblas::Index> row_ind{0, 1, 2, 3, 4, 5, 6, 7 };
  std::vector<graphblas::Index> col_ind{0, 1, 2, 3, 4, 5, 6, 6 };
  std::vector<float>            values {0.,1.,2.,3.,4.,5.,6.,0.};
  graphblas::Index nrows = 8;
  float correct = 91.;
  testTrace( row_ind, col_ind, values, nrows, correct );
}

BOOST_FIXTURE_TEST_CASE( dup2, TestMatrix )
{
  std::vector<graphblas::Index> row_ind{0, 0, 2, 3, 4, 5, 6, 7 };
  std::vector<graphblas::Index> col_ind{0, 6, 2, 3, 4, 5, 6, 6 };
  std::vector<float>            values {0.,1.,2.,3.,4.,5.,6.,0.};
  graphblas::Index nrows = 8;
  float correct = 91.;
  testTrace( row_ind, col_ind, values, nrows, correct );
}

BOOST_FIXTURE_TEST_CASE( dup3, TestMatrix )
{
  float correct = 341.;
  testTraceMtx( "data/small/chesapeake_trace.mtx", correct );
}

BOOST_AUTO_TEST_SUITE_END()
