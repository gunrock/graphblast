#define GRB_USE_APSPIE
//#define private public

#include <iostream>
#include <random>
#include <algorithm>

#include <cstdio>
#include <cstdlib>

#include <graphblas/graphblas.hpp>

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE matrix_suite

#include <boost/test/included/unit_test.hpp>
#include <test/test.hpp>

struct TestMatrix {
  TestMatrix() :
    argc( boost::unit_test::framework::master_test_suite().argc ),
    argv( boost::unit_test::framework::master_test_suite().argv ) {}

  int argc;
  char **argv;
};

BOOST_AUTO_TEST_SUITE(matrix_suite)

// SpMM unit test (chesapeake)
BOOST_AUTO_TEST_CASE( matrix5 )
{
  std::vector<graphblas::Index> row_indices = {
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
	//a.print();
  std::vector<float> denseVal;
	for( int i=0; i<11; i++ ) {
    for( int j=0; j<11; j++ ) {
      if( i==j ) denseVal.push_back(1.0);
			else denseVal.push_back(0.0);
		}
	}
  b.build( denseVal );
  graphblas::Matrix<float> c(11, 11);
  graphblas::Semiring op;
  graphblas::mxm<float, float, float>( c, op, a, b );
  //c.print();
	std::vector<float> out_denseVal;
	c.extractTuples( out_denseVal );
	for( int i=0; i<20; i++ ) {
		graphblas::Index row = row_indices[i];
		graphblas::Index col = col_indices[i];
    float            val = values[i];
		//std::cout << row << " " << col << " " << val << " " << out_denseVal[col*11+row] << std::endl;
		BOOST_ASSERT( val==out_denseVal[col*11+row] );
	}
}

BOOST_AUTO_TEST_SUITE_END() 


BOOST_AUTO_TEST_CASE( matrix4 )
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

// SpMM unit test
// Assert: error out of dimension tuple passed into build
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
// Assert: error: out of dimension tuple passed into build
BOOST_AUTO_TEST_CASE( matrix1 )
{
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
	//a.print();
  std::vector<float> denseVal;
	for( int i=0; i<11; i++ ) {
    for( int j=0; j<11; j++ ) {
      if( i==j ) denseVal.push_back(1.0);
			else denseVal.push_back(0.0);
		}
	}
  b.build( denseVal );
  graphblas::Matrix<float> c(11, 11);
  graphblas::Semiring op;
  graphblas::mxm<float, float, float>( c, op, a, b );
  //c.print();
	std::vector<float> out_denseVal;
	c.extractTuples( out_denseVal );
	for( int i=0; i<20; i++ ) {
		graphblas::Index row = row_indices[i];
		graphblas::Index col = col_indices[i];
    float            val = values[i];
		//std::cout << row << " " << col << " " << val << " " << out_denseVal[col*11+row] << std::endl;
		BOOST_ASSERT( val==out_denseVal[col*11+row] );
	}
}

BOOST_AUTO_TEST_SUITE_END() 
