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

void testDup( char const* tsv );
void testCusparseSpgemm( char const* tsv, const int select=1);
void testFill( const int nrows, const int ncols );
void testTranspose( const int nrows, const int ncols );
void testMxm( char const* tsv, 
              const std::vector<graphblas::Index>& C_row_indices, 
              const std::vector<graphblas::Index>& C_col_indices, 
              const std::vector<float>& C_values,
              const int nblocks,
              const int select=1 );
void testReduce( char const* tsv,
                 const std::vector<float>& correct,
                 const std::vector<float>& correct_t ); 

struct TestMatrix
{
  TestMatrix() :
    DEBUG(true) {}

  bool DEBUG;
};

BOOST_AUTO_TEST_SUITE(dup_suite)

BOOST_FIXTURE_TEST_CASE( dup1, TestMatrix )
{
  testDup( "data/small/test_cc.tsv" );
}

BOOST_FIXTURE_TEST_CASE( dup2, TestMatrix )
{
  testDup( "data/small/test_bc.tsv" );
}

BOOST_FIXTURE_TEST_CASE( dup3, TestMatrix )
{
  testDup( "data/small/chesapeake.tsv" );
}

BOOST_FIXTURE_TEST_CASE( dup4, TestMatrix )
{
  testCusparseSpgemm( "data/small/test_cc.tsv" );
}

BOOST_FIXTURE_TEST_CASE( dup5, TestMatrix )
{
  testCusparseSpgemm( "data/small/test_bc.tsv" );
}

BOOST_FIXTURE_TEST_CASE( dup6, TestMatrix )
{
  testCusparseSpgemm( "data/small/chesapeake.tsv" );
}

BOOST_FIXTURE_TEST_CASE( dup7, TestMatrix )
{
  testCusparseSpgemm( "data/small/test_cc.tsv", 2 );
}

BOOST_FIXTURE_TEST_CASE( dup8, TestMatrix )
{
  testCusparseSpgemm( "data/small/test_bc.tsv", 2 );
}

BOOST_FIXTURE_TEST_CASE( dup9, TestMatrix )
{
  testCusparseSpgemm( "data/small/chesapeake.tsv", 2 );
}

BOOST_FIXTURE_TEST_CASE( dup10, TestMatrix )
{
  testFill( 10, 10 );
}

BOOST_FIXTURE_TEST_CASE( dup11, TestMatrix )
{
  testFill( 40, 40 );
}

BOOST_FIXTURE_TEST_CASE( dup12, TestMatrix )
{
  testFill( 100, 100 );
}

BOOST_FIXTURE_TEST_CASE( dup13, TestMatrix )
{
  testTranspose( 10, 10 );
}

BOOST_FIXTURE_TEST_CASE( dup14, TestMatrix )
{
  testTranspose( 40, 40 );
}

BOOST_FIXTURE_TEST_CASE( dup15, TestMatrix )
{
  testTranspose( 100, 100 );
}

BOOST_FIXTURE_TEST_CASE( dup16, TestMatrix )
{
  std::vector<int> row_indices(1, 1);
  std::vector<int> col_indices(1, 1);
  std::vector<float> values(   1, 20.);
  testMxm( "data/small/test_cc.tsv", row_indices, col_indices, values, 5 );
}

BOOST_FIXTURE_TEST_CASE( dup17, TestMatrix )
{
  std::vector<int> row_indices(1, 1);
  std::vector<int> col_indices(1, 1);
  std::vector<float> values(   1, 15.);
  testMxm( "data/small/test_bc.tsv", row_indices, col_indices, values, 5 );
}

BOOST_FIXTURE_TEST_CASE( dup18, TestMatrix )
{
  std::vector<int> row_indices(1, 1);
  std::vector<int> col_indices(1, 1);
  std::vector<float> values(   1, 170.);
  testMxm( "data/small/chesapeake.tsv", row_indices, col_indices, values, 5 );
}

BOOST_FIXTURE_TEST_CASE( dup19, TestMatrix )
{
  std::vector<float> correct{   1., 1., 3., 2., 2., 3., 3., 0., 1., 2., 2. };
  std::vector<float> correct_t{ 3., 3., 3., 2., 3., 1., 0., 3., 2., 0., 0. };
  testReduce( "data/small/test_cc.tsv", correct, correct_t );
}

BOOST_FIXTURE_TEST_CASE( dup20, TestMatrix )
{
  std::vector<float> correct{   1., 1., 3., 2., 2., 3., 3. };
  std::vector<float> correct_t{ 3., 3., 3., 2., 3., 1., 0. };
  testReduce( "data/small/test_bc.tsv", correct, correct_t );
}

BOOST_AUTO_TEST_SUITE_END()

// Tests dup(), build(), extractTuples(), clear()
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

// Tests dup(), build(), extractTuples(), clear(), cusparse_spgemm()
void testCusparseSpgemm( char const* tsv, const int select )
{
  std::vector<graphblas::Index> row_indices;
  std::vector<graphblas::Index> col_indices;
  std::vector<float> values;
  graphblas::Index nrows, ncols, nvals;
  graphblas::Info err;
  std::vector<graphblas::Index> C_row_indices;
  std::vector<graphblas::Index> C_col_indices;
  std::vector<float> C_values;

  // Read in sparse matrix
  readTsv( tsv, row_indices, col_indices, values, nrows, ncols,
    nvals );

  // Set identity matrix to be as big as sparse matrix
  std::vector<graphblas::Index> I_row_indices( nrows+1, 0 );
  std::vector<float> I_values( nrows, 1. );
  std::iota( std::begin(I_row_indices), std::end(I_row_indices), 0 );
  BOOST_ASSERT( I_row_indices[nrows]==nrows );

  // Matrix adj (adjacency matrix) and adj_t (adjacency matrix transpose)
  graphblas::SparseMatrix<float> adj(nrows, ncols);
  graphblas::SparseMatrix<float> I(  nrows, ncols, nrows );
  err = adj.build( row_indices, col_indices, values, nvals );
  err = adj.gpuToCpu();
  //check( adj );
  //adj.print();
  err = I.build( I_row_indices, I_row_indices, I_values, nrows);
  err = I.gpuToCpu();
  //check( I );

  graphblas::SparseMatrix<float> C(  nrows, ncols);
  if( select==1 )
    err = graphblas::cusparse_spgemm( C, adj, I );
  else if( select==2 )
    err = graphblas::cusparse_spgemm2( C, adj, I );
  err = C.gpuToCpu();
  //check( C );
  //C.print();
  err = C.extractTuples( C_row_indices, C_col_indices, C_values );

  BOOST_ASSERT_LIST( C_row_indices, row_indices, nvals );
  BOOST_ASSERT_LIST( C_col_indices, col_indices, nvals );
  BOOST_ASSERT_LIST( C_values,      values,      nvals );

  BOOST_ASSERT( err==0 );
}

// Tests fillAscending(), fill()
void testFill( const int nrows, const int ncols )
{
  graphblas::Info err;

  // Matrix adj (adjacency matrix) and adj_t (adjacency matrix transpose)
  graphblas::SparseMatrix<float> adj;

  // initialize curr_partitions nx1
  err = adj.nnew( nrows, ncols, nrows );
  err = adj.fillAscending( 0, nrows+1, 0  );
  err = adj.fill(          1, nrows,   1  );
  err = adj.fill(          2, nrows,   1. );

  std::vector<graphblas::Index> row_ptr(nrows+1, 0);
  std::iota( std::begin(row_ptr), std::end(row_ptr), 0);
  std::vector<graphblas::Index> col_ind(nrows, 1);
  std::vector<float> vals(nrows, 1.);

  BOOST_ASSERT_LIST( adj.h_csrRowPtr_, row_ptr, nrows+1 );
  BOOST_ASSERT_LIST( adj.h_csrColInd_, col_ind, nrows );
  BOOST_ASSERT_LIST( adj.h_csrVal_,    vals,    nrows );

  BOOST_ASSERT( err==0 );
}

// Tests transpose() and gpuToCpu()
void testTranspose( const int nrows, const int ncols )
{
  graphblas::Info err;
  std::vector<graphblas::Index> row_indices;
  std::vector<graphblas::Index> col_indices;
  std::vector<float>             values;

  // Matrix adj (adjacency matrix) and adj_t (adjacency matrix transpose)
  graphblas::Matrix<float> adj;
  graphblas::Matrix<float> adj_t;

  // initialize curr_partitions nx1
  err = adj.nnew( nrows, ncols, nrows );
  err = adj_t.nnew( ncols, nrows, nrows );
  err = adj.fillAscending( 0, nrows+1, 0  );
  err = adj.fill(          1, nrows,   1  );
  err = adj.fill(          2, nrows,   1. );
  err = adj_t.fill(        2, nrows,   1. );

  std::vector<graphblas::Index> row_ptr(nrows+1, 0);
  std::iota( std::begin(row_ptr), std::end(row_ptr), 0);
  std::vector<graphblas::Index> col_ind(nrows, 1);
  std::vector<float> vals(nrows, 1.);

  err = graphblas::transpose( adj_t, adj );

  std::vector<graphblas::Index> row_ptr_t(nrows+1, 1);
  std::vector<graphblas::Index> col_ind_t(nrows, 0);
  std::iota( std::begin(col_ind_t), std::end(col_ind_t), 0 );
  std::vector<float> vals_t(nrows, 1.);
  err = adj.extractTuples( row_indices, col_indices, values );
  err = adj_t.extractTuples( row_indices, col_indices, values );

  BOOST_ASSERT_LIST( row_indices, row_ptr_t, nrows+1 );
  BOOST_ASSERT_LIST( col_indices, col_ind_t, nrows );
  BOOST_ASSERT_LIST( values,      vals_t,    nrows );

  BOOST_ASSERT( err==0 );
}

// Tests build(), extractTuples(), mxm(), transpose()
void testMxm( char const* tsv, 
              const std::vector<graphblas::Index>& C_row_indices, 
              const std::vector<graphblas::Index>& C_col_indices, 
              const std::vector<float>& C_values,
              const int nblocks,
              const int select )
{
  std::vector<graphblas::Index> row_indices;
  std::vector<graphblas::Index> col_indices;
  std::vector<float> values;
  graphblas::Index nrows, ncols, nvals;
  graphblas::Info err;

  // Read in sparse matrix
  readTsv( tsv, row_indices, col_indices, values, nrows, ncols,
    nvals );

  // Set partition matrix
  graphblas::Matrix<float> curr_partition;
  graphblas::Matrix<float> curr_partition_t;
  err = curr_partition.nnew( nrows, nblocks, nrows );
  err = curr_partition.fillAscending( 0, nrows+1, 0 );
  err = curr_partition.fill(          1, nrows,   1  );
  err = curr_partition.fill(          2, nrows,   1. );
  err = curr_partition_t.nnew( nblocks, nrows, nrows );
  err = curr_partition_t.fill(        2, nrows,   1. );
  err = graphblas::transpose( curr_partition_t, curr_partition );
  //curr_partition.print();
  //curr_partition_t.print();
  //check( curr_partition.sparse_ );
  //check( curr_partition_t.sparse_ );

  // Matrix adj (adjacency matrix) and adj_t (adjacency matrix transpose)
  graphblas::Matrix<float> adj(nrows, ncols);
  err = adj.build( row_indices, col_indices, values, nvals );
  //adj.print();
  //check( adj.sparse_ );

  graphblas::Matrix<float> C(  nblocks, nblocks);
  graphblas::Matrix<float> D(  nrows,   nblocks);
  if( select==1 )
    err = graphblas::mxm( C, D, curr_partition_t, adj, 
        curr_partition );
  else if( select==2 )
    err = graphblas::mxm( C, D, curr_partition_t, adj, 
        curr_partition );
  //C.print();
  //check( C.sparse_ );

  err = C.extractTuples( row_indices, col_indices, values );

  BOOST_ASSERT_LIST( C_row_indices, row_indices, nvals );
  BOOST_ASSERT_LIST( C_col_indices, col_indices, nvals );
  BOOST_ASSERT_LIST( C_values,      values,      nvals );

  BOOST_ASSERT( err==0 );
}

void testReduce( char const* tsv,
                 const std::vector<float>& correct, 
                 const std::vector<float>& correct_t ) 
{
  std::vector<graphblas::Index> row_indices;
  std::vector<graphblas::Index> col_indices;
  std::vector<float> values;
  graphblas::Index nrows, ncols, nvals;
  graphblas::Info err;

  // Read in sparse matrix
  readTsv( tsv, row_indices, col_indices, values, nrows, ncols,
    nvals );

  graphblas::Matrix<float> adj(nrows, ncols);
  err = adj.build( row_indices, col_indices, values, nvals );

  graphblas::Matrix<float> adj_t(nrows, ncols);
  err = adj_t.build( row_indices, col_indices, values, nvals, true );

  graphblas::Vector<float> vec(nrows);
  graphblas::Vector<float> vec_t(ncols);

  err = graphblas::reduce( vec, adj );
  err = graphblas::reduce( vec_t, adj_t );

  err = vec.extract( values );
  BOOST_ASSERT_LIST( values, correct, nrows );

  err = vec_t.extract( values );
  BOOST_ASSERT_LIST( values, correct_t, nrows );
}
