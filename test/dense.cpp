#define GrB_USE_SEQUENTIAL
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
#include "graphblas/backend/sequential/sequential.hpp"
#include "test/test.hpp"

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE triple_suite

#include <boost/test/included/unit_test.hpp>
#include <boost/program_options.hpp>

void testMxm( char const* tsv, std::vector<int>& correct, 
              std::vector<int>& correct_t )
{
  std::vector<graphblas::Index> row_indices;
  std::vector<graphblas::Index> col_indices;
  std::vector<float> values;
  graphblas::Index nrows, ncols, nvals;
  graphblas::Info err;

  int argc = 3;
  char** argv = (char**) malloc(3*sizeof(char*));
  argv[1] = (char *) "--verbose=false";
  argv[2] = (char *) "--merge.r=0";

  // Read in sparse matrix
  err = readTsv( tsv, row_indices, col_indices, values, nrows, ncols,
    nvals );

  graphblas::BlockState<graphblas::T> state( nrows, ncols );
  err = state.parseArgs( argc, argv );
  err = state.build( row_indices, col_indices, values, nvals );

  std::vector<int> k_out;
  std::vector<int> k_in;
  int temp;

  printArray( "curr_part", state.block_state_.curr_partition_, nrows );
  err = graphblas::mxm( state.block_state_.blk_, 
                                  state.block_state_.curr_partition_, 
                                  state.block_state_.adj_ );
  /*for( int i=0; i<nrows; i++ )
  {
    err = state.block_state_.adj_.getRowLength( temp, i );
    k_out.push_back( temp );
    err = state.block_state_.adj_t_.getRowLength( temp, i );
    k_in.push_back( temp );
  }

  BOOST_ASSERT_LIST( k_out, correct, std::min(nrows,10) );
  BOOST_ASSERT_LIST( k_in, correct_t, std::min(nrows,10) );

  BOOST_ASSERT( err==0 );*/
}

struct TestMatrix
{
  TestMatrix() :
    DEBUG(true) {}

  bool DEBUG;
};

BOOST_AUTO_TEST_SUITE(triple_suite)

BOOST_FIXTURE_TEST_CASE( triple1, TestMatrix )
{
  std::vector<int> adj_deg   = {1, 1, 3, 2, 2, 3, 3, 0, 1, 2};
  std::vector<int> adj_t_deg = {3, 3, 3, 2, 3, 1, 0, 3, 2, 0};
  testMxm( "data/small/test_cc.tsv", adj_deg, adj_t_deg );
}

BOOST_FIXTURE_TEST_CASE( triple2, TestMatrix )
{
  std::vector<int> adj_deg   = {1, 1, 3, 2, 2, 3, 3};
  std::vector<int> adj_t_deg = {3, 3, 3, 2, 3, 1, 0};
  testMxm( "data/small/test_bc.tsv", adj_deg, adj_t_deg );
}

BOOST_FIXTURE_TEST_CASE( triple3, TestMatrix )
{
  std::vector<int> adj_deg   = {4, 3, 3, 1, 1, 2, 4, 4, 1, 3};
  std::vector<int> adj_t_deg = {4, 4, 2, 1, 2, 1, 3, 3, 4, 1};
  testMxm( "data/small/chesapeake.tsv", adj_deg, adj_t_deg );
}

/*BOOST_FIXTURE_TEST_CASE( triple4, TestMatrix )
{
  testDup( "data/small/test_cc.tsv" );
}

BOOST_FIXTURE_TEST_CASE( triple5, TestMatrix )
{
  testDup( "data/small/test_bc.tsv" );
}

BOOST_FIXTURE_TEST_CASE( triple6, TestMatrix )
{
  testDup( "data/small/chesapeake.tsv" );
}

BOOST_FIXTURE_TEST_CASE( triple7, TestMatrix )
{
  testShiftRight( "data/small/test_cc.tsv" ); 
}

BOOST_FIXTURE_TEST_CASE( triple8, TestMatrix )
{
  std::map<graphblas::Index,graphblas::Index> out_edge;
  out_edge.insert({{2, 1}, {3, 0}, {5,1}});
  std::vector<graphblas::Index> correct = {3, 4, -1};
  testEwiseSub( "data/small/test_cc.tsv", out_edge, correct);
}

BOOST_FIXTURE_TEST_CASE( triple9, TestMatrix )
{
  std::map<graphblas::Index,graphblas::Index> out_edge;
  out_edge.insert({{2,0}, {4,1}});
  std::vector<graphblas::Index> correct_ind = {2, 4};
  std::vector<graphblas::T> correct_val = {1, 1};
  testEwiseAdd( "data/small/test_cc.tsv", out_edge, 0, correct_ind, 
      correct_val, 3);
}

BOOST_FIXTURE_TEST_CASE( triple10, TestMatrix )
{
  std::map<graphblas::Index,graphblas::Index> out_edge;
  out_edge.insert({{1,1}, {2,0}, {4,1}, {5,2}, {6,1}});
  std::vector<graphblas::Index> correct_ind = {1, 2, 3, 4, 5, 6};
  std::vector<graphblas::T> correct_val = {1, 1, 1, 2, 2, 1};
  testEwiseAdd( "data/small/test_cc.tsv", out_edge, 4, correct_ind, 
      correct_val, 2);
}

BOOST_FIXTURE_TEST_CASE( triple11, TestMatrix )
{
  std::map<graphblas::Index,graphblas::Index> out_edge;
  out_edge.insert( {{1,1}, {2,0}, {4,1}, {5,2}, {6,1}, {7,1}, {8,10}} );
  std::vector<graphblas::Index> correct_ind = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<graphblas::T> correct_val = {1, 1, 1, 2, 2, 1, 1, 10};
  testEwiseAdd( "data/small/test_cc.tsv", out_edge, 7, correct_ind, 
      correct_val, 1);
}

BOOST_FIXTURE_TEST_CASE( triple12, TestMatrix )
{
  std::map<graphblas::Index,graphblas::Index> out_edge;
  out_edge.insert({{2, 1}, {3, 0}, {4,1}});
  std::vector<graphblas::Index> correct = {1, 2, 3, 0, 2, 4, 3, 4, -1, 5, 6, 2, 6, -1, 6, 8, 9, 10, 9, 10};
  testEwiseSubCol( "data/small/test_cc.tsv", out_edge, correct);
}

BOOST_FIXTURE_TEST_CASE( triple13, TestMatrix )
{
  testResize( "data/small/test_cc.tsv" );
}

BOOST_FIXTURE_TEST_CASE( triple14, TestMatrix )
{
  testResize( "data/small/chesapeake.tsv" );
}

BOOST_FIXTURE_TEST_CASE( triple15, TestMatrix )
{
  testRequest( "data/small/test_cc.tsv", 5 );
}

BOOST_FIXTURE_TEST_CASE( triple16, TestMatrix )
{
  testRequest( "data/small/chesapeake.tsv", 5 );
}

BOOST_FIXTURE_TEST_CASE( triple17, TestMatrix )
{
  testRequest( "data/static/simulated_blockmodel_graph_1000_nodes.tsv", 5 );
}

BOOST_FIXTURE_TEST_CASE( triple18, TestMatrix )
{
  std::map<graphblas::Index,graphblas::Index> out_edge;
  out_edge.insert( {{1,1}, {2,0}, {4,1}, {5,2}, {6,1}, {7,1}, {8,10}} );
  std::vector<graphblas::Index> correct_ind = {1, 2, 3, 0, 2, 4, 5, 3, 4, 5, 5, 6, 2, 5, 6, 5, 6, 5, 5, 8, 9, 10, 5, 9, 10 };
  std::vector<graphblas::T>     correct_val = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 10, 1, 1 };
  testEwiseAddCol( "data/small/test_cc.tsv", out_edge, correct_ind, correct_val);
}*/

BOOST_AUTO_TEST_SUITE_END()
