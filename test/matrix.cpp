#define GRB_USE_SEQUENTIAL
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

void testRowLength( char const* tsv, std::vector<int>& correct, 
           std::vector<int>& correct_t )
{
  std::vector<graphblas::Index> row_indices;
  std::vector<graphblas::Index> col_indices;
  std::vector<float> values;
  graphblas::Index nrows, ncols, nvals;
  graphblas::Info err;

  int argc = 2;
  char** argv = (char**) malloc(2*sizeof(char*));
  argv[1] = (char *) "--verbose=false";

  // Read in sparse matrix
  err = readTsv( tsv, row_indices, col_indices, values, nrows, ncols,
    nvals );

  graphblas::BlockState<graphblas::T> state( nrows, ncols );
  err = state.parseArgs( argc, argv );
  err = state.build( row_indices, col_indices, values, nvals );

  std::vector<int> k_out;
  std::vector<int> k_in;
  int temp;

  for( int i=0; i<nrows; i++ )
  {
    err = state.block_state_.adj_.getRowLength( temp, i );
    k_out.push_back( temp );
    err = state.block_state_.adj_t_.getRowLength( temp, i );
    k_in.push_back( temp );
  }

  BOOST_ASSERT_LIST( k_out, correct, std::min(nrows,10) );
  BOOST_ASSERT_LIST( k_in, correct_t, std::min(nrows,10) );

  BOOST_ASSERT( err==0 );
}

void testDup( char const* tsv )
{
  std::vector<graphblas::Index> row_indices;
  std::vector<graphblas::Index> col_indices;
  std::vector<float> values;
  graphblas::Index nrows, ncols, nvals;
  graphblas::Info err;
  std::vector<graphblas::Index> adj_row, adj_col, blk_row, blk_col;
  std::vector<float> adj_val, blk_val;

  // Read in sparse matrix
  readTsv( tsv, row_indices, col_indices, values, nrows, ncols,
    nvals );

  // Matrix adj (adjacency matrix) and adj_t (adjacency matrix transpose)
  graphblas::Matrix<float> adj(nrows, ncols);
  adj.build( row_indices, col_indices, values, nvals );
  adj.print();
  check( adj.sparse_ );

  graphblas::Matrix<float> adj_t(nrows, ncols);
  adj_t.build( row_indices, col_indices, values, nvals, true );

  // Matrix blk (interblock edge count matrix) and blk_t (interblock transpose)
  graphblas::Matrix<float> blk(nrows, ncols);
  err = blk.dup( adj );
  err = adj.extractTuples( adj_row, adj_col, adj_val );
  err = blk.extractTuples( blk_row, blk_col, blk_val ); 
  err = adj.clear();

  BOOST_ASSERT_LIST( adj_row, blk_row, nvals );
  BOOST_ASSERT_LIST( adj_col, blk_col, nvals );
  BOOST_ASSERT_LIST( adj_val, blk_val, nvals );

  graphblas::Matrix<float> blk_t(nrows, ncols);
  err = blk_t.dup( adj_t );
  err = adj_t.extractTuples( adj_row, adj_col, adj_val );
  err = blk_t.extractTuples( blk_row, blk_col, blk_val ); 
  err = adj_t.clear();

  BOOST_ASSERT_LIST( adj_row, blk_row, nvals );
  BOOST_ASSERT_LIST( adj_col, blk_col, nvals );
  BOOST_ASSERT_LIST( adj_val, blk_val, nvals );

  BOOST_ASSERT( err==0 );
}

void testShiftRight( char const* tsv )
{
  std::vector<graphblas::Index> row_indices;
  std::vector<graphblas::Index> col_indices;
  std::vector<float> values;
  graphblas::Index nrows, ncols, nvals;
  graphblas::Info err;

  int argc = 2;
  char** argv = (char**) malloc(2*sizeof(char*));
  argv[1] = (char *) "--verbose=false";

  // Read in sparse matrix
  err = readTsv( tsv, row_indices, col_indices, values, nrows, ncols,
    nvals );

  graphblas::BlockState<graphblas::T> state( nrows, ncols );
  err = state.parseArgs( argc, argv );
  err = state.build( row_indices, col_indices, values, nvals );
  check( state.block_state_.blk_.sparse_ );

  graphblas::Index col_ind = state.block_state_.blk_.sparse_.h_csrRowPtr_[5];
  state.block_state_.blk_.sparse_.h_csrColInd_[col_ind] = -1;
  state.block_state_.blk_.sparse_.h_rowLength_[5]--;

  state.block_state_.blk_.sparse_.shiftRight( 5 );
  check( state.block_state_.blk_.sparse_ );
  BOOST_ASSERT( err==0 );
}

void testEwiseSub( char const*                          tsv, 
                   const std::map<graphblas::Index,
                       graphblas::Index>&              out_edge,
                   const std::vector<graphblas::Index> correct )
{
  std::vector<graphblas::Index> row_indices;
  std::vector<graphblas::Index> col_indices;
  std::vector<float> values;
  graphblas::Index nrows, ncols, nvals;
  graphblas::Info err;

  int argc = 2;
  char** argv = (char**) malloc(2*sizeof(char*));
  argv[1] = (char *) "--verbose=false";

  // Read in sparse matrix
  err = readTsv( tsv, row_indices, col_indices, values, nrows, ncols,
    nvals );

  graphblas::BlockState<graphblas::T> state( nrows, ncols );
  err = state.parseArgs( argc, argv );
  err = state.build( row_indices, col_indices, values, nvals );
  //printArray("col_ind",state.block_state_.blk_.sparse_.h_csrColInd_);

  err = ewiseSub( state.block_state_.blk_.sparse_, 5, out_edge );
  //printArray("col_ind",state.block_state_.blk_.sparse_.h_csrColInd_);

  state.block_state_.blk_.sparse_.shiftRight( 5 );
  //printArray("col_ind",state.block_state_.blk_.sparse_.h_csrColInd_);
  check( state.block_state_.blk_.sparse_ );

  graphblas::Index col_ind = state.block_state_.blk_.sparse_.h_csrRowPtr_[5];
  BOOST_ASSERT_LIST( state.block_state_.blk_.sparse_.h_csrColInd_+col_ind, correct, 
      (int) correct.size() );
  BOOST_ASSERT( err==0 );
}

void testEwiseAdd( char const*                          tsv, 
                   const std::map<graphblas::Index,
                       graphblas::Index>&              out_edge,
                   const graphblas::Index              to_add,
                   const std::vector<graphblas::Index> correct_ind,
                   const std::vector<graphblas::T>     correct_val,
                   const int                            case_choice )
{
  std::vector<graphblas::Index> row_indices;
  std::vector<graphblas::Index> col_indices;
  std::vector<float> values;
  graphblas::Index nrows, ncols, nvals;
  graphblas::Info err;

  int argc = 2;
  char** argv = (char**) malloc(2*sizeof(char*));
  argv[1] = (char *) "--verbose=false";

  // Read in sparse ma://sacmail.nvidia.com/owa/#path=/mailtrix
  err = readTsv( tsv, row_indices, col_indices, values, nrows, ncols,
    nvals );

  graphblas::BlockState<graphblas::T> state( nrows, ncols );
  err = state.parseArgs( argc, argv );
  err = state.build( row_indices, col_indices, values, nvals );

  graphblas::Index p_rowLength = state.block_state_.blk_.sparse_.h_csrRowPtr_[6]-
      state.block_state_.blk_.sparse_.h_csrRowPtr_[5];
  switch( case_choice )
  {
    case 3:
      state.block_state_.blk_.sparse_.remove( 5, 3 );
      state.block_state_.blk_.sparse_.remove( 5, 4 );
      state.block_state_.blk_.sparse_.shiftRight( 5 );
      BOOST_ASSERT( to_add+state.block_state_.blk_.sparse_.h_rowLength_[5] <= p_rowLength );
      break;
    case 2:
      BOOST_ASSERT( to_add+state.block_state_.blk_.sparse_.h_rowLength_[5] > p_rowLength );
      BOOST_ASSERT( to_add+state.block_state_.blk_.sparse_.h_csrRowPtr_
          [state.block_state_.blk_.sparse_.nrows_] <= state.block_state_.blk_.sparse_.ncapacity_ );
      break;
    case 1:
      BOOST_ASSERT( to_add+state.block_state_.blk_.sparse_.h_rowLength_[5] > p_rowLength );
      BOOST_ASSERT( to_add+state.block_state_.blk_.sparse_.h_csrRowPtr_
          [state.block_state_.blk_.sparse_.nrows_] > state.block_state_.blk_.sparse_.ncapacity_ );
  }
  //printArray("col_ind",state.block_state_.blk_.sparse_.h_csrColInd_);
  //printArray("val",state.block_state_.blk_.sparse_.h_csrVal_);

  err = ewiseAdd( state.block_state_.blk_.sparse_, 5, out_edge );

  state.block_state_.blk_.sparse_.shiftRight( 5 );
  //printArray("col_ind",state.block_state_.blk_.sparse_.h_csrColInd_);
  //printArray("val",state.block_state_.blk_.sparse_.h_csrVal_);
  check( state.block_state_.blk_.sparse_ );

  if( case_choice==1 )
    BOOST_ASSERT( state.block_state_.blk_.sparse_.ncapacity_==1.5*nvals*1.2 );

  graphblas::Index col_ind = state.block_state_.blk_.sparse_.h_csrRowPtr_[5];
  graphblas::Index col_ind_old = state.block_state_.adj_.sparse_.h_csrRowPtr_[6];
  graphblas::Index col_ind_new = state.block_state_.blk_.sparse_.h_csrRowPtr_[6];
  graphblas::Index length = nvals-col_ind_old;
  BOOST_ASSERT_LIST( state.block_state_.blk_.sparse_.h_csrColInd_+col_ind, correct_ind, 
      (int) correct_ind.size() );
  BOOST_ASSERT_LIST( state.block_state_.blk_.sparse_.h_csrVal_+col_ind, correct_val, 
      (int) correct_val.size() );
  BOOST_ASSERT_LIST( state.block_state_.blk_.sparse_.h_csrColInd_+col_ind_new, 
      state.block_state_.adj_.sparse_.h_csrColInd_+col_ind_old, length );
  BOOST_ASSERT_LIST( state.block_state_.blk_.sparse_.h_csrVal_+col_ind_new, 
      state.block_state_.adj_.sparse_.h_csrVal_+col_ind_old, length );
  BOOST_ASSERT( err==0 );
}

void testEwiseSubCol( char const*                          tsv, 
                      const std::map<graphblas::Index,
                          graphblas::Index>&              out_edge,
                      const std::vector<graphblas::Index> correct )
{
  std::vector<graphblas::Index> row_indices;
  std::vector<graphblas::Index> col_indices;
  std::vector<float> values;
  graphblas::Index nrows, ncols, nvals;
  graphblas::Info err;

  int argc = 2;
  char** argv = (char**) malloc(2*sizeof(char*));
  argv[1] = (char *) "--verbose=false";

  // Read in sparse matrix
  err = readTsv( tsv, row_indices, col_indices, values, nrows, ncols,
    nvals );

  graphblas::BlockState<graphblas::T> state( nrows, ncols );
  err = state.parseArgs( argc, argv );
  err = state.build( row_indices, col_indices, values, nvals );
  //printArray("col_ind",state.block_state_.blk_.sparse_.h_csrColInd_);

  err = ewiseSubCol( state.block_state_.blk_t_.sparse_, 5, out_edge );
  //printArray("col_ind",state.block_state_.blk_.sparse_.h_csrColInd_);

  state.block_state_.blk_t_.sparse_.shiftRight( 5 );
  //printArray("col_ind",state.block_state_.blk_.sparse_.h_csrColInd_);
  check( state.block_state_.blk_.sparse_ );

  BOOST_ASSERT_LIST( state.block_state_.blk_t_.sparse_.h_csrColInd_, correct, 
      (int) correct.size() );
  BOOST_ASSERT( err==0 );
}

void testEwiseAddCol( char const*                          tsv, 
                      const std::map<graphblas::Index,
                          graphblas::Index>&              out_edge,
                      const std::vector<graphblas::Index> correct_ind,
                      const std::vector<graphblas::T>     correct_val )
{
  std::vector<graphblas::Index> row_indices;
  std::vector<graphblas::Index> col_indices;
  std::vector<float> values;
  graphblas::Index nrows, ncols, nvals;
  graphblas::Info err;

  int argc = 2;
  char** argv = (char**) malloc(2*sizeof(char*));
  argv[1] = (char *) "--verbose=false";

  // Read in sparse matrix
  err = readTsv( tsv, row_indices, col_indices, values, nrows, ncols,
    nvals );

  graphblas::BlockState<graphblas::T> state( nrows, ncols );
  err = state.parseArgs( argc, argv );
  err = state.build( row_indices, col_indices, values, nvals );
  //printArray("col_ind",state.block_state_.blk_.sparse_.h_csrColInd_);

  err = ewiseAddCol( state.block_state_.blk_t_.sparse_, 5, out_edge );
  //printArray("col_ind",state.block_state_.blk_.sparse_.h_csrColInd_);

  state.block_state_.blk_t_.sparse_.shiftRight( 5 );
  //printArray("col_ind",state.block_state_.blk_.sparse_.h_csrColInd_);
  check( state.block_state_.blk_.sparse_ );

  BOOST_ASSERT_LIST( state.block_state_.blk_t_.sparse_.h_csrColInd_, correct_ind, 
      (int) correct_ind.size() );
  BOOST_ASSERT_LIST( state.block_state_.blk_t_.sparse_.h_csrVal_, correct_val, 
      (int) correct_val.size() );
  BOOST_ASSERT( err==0 );
}

void testRequest( char const* tsv, 
                  const graphblas::Index row )
{
  std::vector<graphblas::Index> row_indices;
  std::vector<graphblas::Index> col_indices;
  std::vector<float> values;
  graphblas::Index nrows, ncols, nvals;
  graphblas::Info err;

  int argc = 2;
  char** argv = (char**) malloc(2*sizeof(char*));
  argv[1] = (char *) "--verbose=false";

  // Read in sparse matrix
  err = readTsv( tsv, row_indices, col_indices, values, nrows, ncols,
    nvals );

  graphblas::BlockState<graphblas::T> state( nrows, ncols );
  err = state.parseArgs( argc, argv );
  err = state.build( row_indices, col_indices, values, nvals );
  //printArray("col_ind",state.block_state_.blk_.sparse_.h_csrColInd_);

  const graphblas::Index request = state.block_state_.blk_.sparse_.ncapacity_-nvals;
  //std::cout << state.block_state_.blk_.sparse_.ncapacity_ << " " << request << std::endl;
  err = state.block_state_.blk_.sparse_.request( row, request );
  graphblas::Index col_ind = state.block_state_.blk_.sparse_.h_csrRowPtr_[row+1];
  BOOST_ASSERT_LIST( state.block_state_.blk_.sparse_.h_csrColInd_+col_ind+request, 
      state.block_state_.adj_.sparse_.h_csrColInd_+col_ind, 
      state.block_state_.blk_.sparse_.h_csrColInd_[nrows]-
          state.block_state_.blk_.sparse_.h_csrColInd_[row+1] );
  //printArray("col_ind",state.block_state_.blk_.sparse_.h_csrColInd_);

  //check( state.block_state_.blk_.sparse_ );

  BOOST_ASSERT( err==0 );
}

void testResize( char const* tsv )
{
  std::vector<graphblas::Index> row_indices;
  std::vector<graphblas::Index> col_indices;
  std::vector<float> values;
  graphblas::Index nrows, ncols, nvals;
  graphblas::Info err;

  int argc = 2;
  char** argv = (char**) malloc(2*sizeof(char*));
  argv[1] = (char *) "--verbose=false";

  // Read in sparse matrix
  err = readTsv( tsv, row_indices, col_indices, values, nrows, ncols,
    nvals );

  graphblas::BlockState<graphblas::T> state( nrows, ncols );
  err = state.parseArgs( argc, argv );
  err = state.build( row_indices, col_indices, values, nvals );
  //printArray("col_ind",state.block_state_.blk_.sparse_.h_csrColInd_);

  //std::cout << state.block_state_.blk_.sparse_.ncapacity_ << " " << request << std::endl;
  err = state.block_state_.blk_.sparse_.resize();
  BOOST_ASSERT_LIST( state.block_state_.blk_.sparse_.h_csrColInd_, 
      state.block_state_.adj_.sparse_.h_csrColInd_, nvals );
  //printArray("col_ind",state.block_state_.blk_.sparse_.h_csrColInd_);

  //check( state.block_state_.blk_.sparse_ );

  BOOST_ASSERT( err==0 );
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
  std::vector<int> adj_deg   = {1, 1, 3, 2, 2, 3, 3, 0, 1, 2};
  std::vector<int> adj_t_deg = {3, 3, 3, 2, 3, 1, 0, 3, 2, 0};
  testRowLength( "data/small/test_cc.tsv", adj_deg, adj_t_deg );
}

BOOST_FIXTURE_TEST_CASE( dup2, TestMatrix )
{
  std::vector<int> adj_deg   = {1, 1, 3, 2, 2, 3, 3};
  std::vector<int> adj_t_deg = {3, 3, 3, 2, 3, 1, 0};
  testRowLength( "data/small/test_bc.tsv", adj_deg, adj_t_deg );
}

BOOST_FIXTURE_TEST_CASE( dup3, TestMatrix )
{
  std::vector<int> adj_deg   = {4, 3, 3, 1, 1, 2, 4, 4, 1, 3};
  std::vector<int> adj_t_deg = {4, 4, 2, 1, 2, 1, 3, 3, 4, 1};
  testRowLength( "data/small/chesapeake.tsv", adj_deg, adj_t_deg );
}

BOOST_FIXTURE_TEST_CASE( dup4, TestMatrix )
{
  testDup( "data/small/test_cc.tsv" );
}

BOOST_FIXTURE_TEST_CASE( dup5, TestMatrix )
{
  testDup( "data/small/test_bc.tsv" );
}

BOOST_FIXTURE_TEST_CASE( dup6, TestMatrix )
{
  testDup( "data/small/chesapeake.tsv" );
}

BOOST_FIXTURE_TEST_CASE( dup7, TestMatrix )
{
  testShiftRight( "data/small/test_cc.tsv" ); 
}

BOOST_FIXTURE_TEST_CASE( dup8, TestMatrix )
{
  std::map<graphblas::Index,graphblas::Index> out_edge;
  out_edge.insert({{2, 1}, {3, 0}, {5,1}});
  std::vector<graphblas::Index> correct = {3, 4, -1};
  testEwiseSub( "data/small/test_cc.tsv", out_edge, correct);
}

BOOST_FIXTURE_TEST_CASE( dup9, TestMatrix )
{
  std::map<graphblas::Index,graphblas::Index> out_edge;
  out_edge.insert({{2,0}, {4,1}});
  std::vector<graphblas::Index> correct_ind = {2, 4};
  std::vector<graphblas::T> correct_val = {1, 1};
  testEwiseAdd( "data/small/test_cc.tsv", out_edge, 0, correct_ind, 
      correct_val, 3);
}

BOOST_FIXTURE_TEST_CASE( dup10, TestMatrix )
{
  std::map<graphblas::Index,graphblas::Index> out_edge;
  out_edge.insert({{1,1}, {2,0}, {4,1}, {5,2}, {6,1}});
  std::vector<graphblas::Index> correct_ind = {1, 2, 3, 4, 5, 6};
  std::vector<graphblas::T> correct_val = {1, 1, 1, 2, 2, 1};
  testEwiseAdd( "data/small/test_cc.tsv", out_edge, 4, correct_ind, 
      correct_val, 2);
}

BOOST_FIXTURE_TEST_CASE( dup11, TestMatrix )
{
  std::map<graphblas::Index,graphblas::Index> out_edge;
  out_edge.insert( {{1,1}, {2,0}, {4,1}, {5,2}, {6,1}, {7,1}, {8,10}} );
  std::vector<graphblas::Index> correct_ind = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<graphblas::T> correct_val = {1, 1, 1, 2, 2, 1, 1, 10};
  testEwiseAdd( "data/small/test_cc.tsv", out_edge, 7, correct_ind, 
      correct_val, 1);
}

BOOST_FIXTURE_TEST_CASE( dup12, TestMatrix )
{
  std::map<graphblas::Index,graphblas::Index> out_edge;
  out_edge.insert({{2, 1}, {3, 0}, {4,1}});
  std::vector<graphblas::Index> correct = {1, 2, 3, 0, 2, 4, 3, 4, -1, 5, 6, 2, 6, -1, 6, 8, 9, 10, 9, 10};
  testEwiseSubCol( "data/small/test_cc.tsv", out_edge, correct);
}

BOOST_FIXTURE_TEST_CASE( dup13, TestMatrix )
{
  testResize( "data/small/test_cc.tsv" );
}

BOOST_FIXTURE_TEST_CASE( dup14, TestMatrix )
{
  testResize( "data/small/chesapeake.tsv" );
}

BOOST_FIXTURE_TEST_CASE( dup15, TestMatrix )
{
  testRequest( "data/small/test_cc.tsv", 5 );
}

BOOST_FIXTURE_TEST_CASE( dup16, TestMatrix )
{
  testRequest( "data/small/chesapeake.tsv", 5 );
}

BOOST_FIXTURE_TEST_CASE( dup17, TestMatrix )
{
  testRequest( "data/static/simulated_blockmodel_graph_1000_nodes.tsv", 5 );
}

BOOST_FIXTURE_TEST_CASE( dup18, TestMatrix )
{
  std::map<graphblas::Index,graphblas::Index> out_edge;
  out_edge.insert( {{1,1}, {2,0}, {4,1}, {5,2}, {6,1}, {7,1}, {8,10}} );
  std::vector<graphblas::Index> correct_ind = {1, 2, 3, 0, 2, 4, 5, 3, 4, 5, 5, 6, 2, 5, 6, 5, 6, 5, 5, 8, 9, 10, 5, 9, 10 };
  std::vector<graphblas::T>     correct_val = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 10, 1, 1 };
  testEwiseAddCol( "data/small/test_cc.tsv", out_edge, correct_ind, correct_val);
}

BOOST_AUTO_TEST_SUITE_END()
