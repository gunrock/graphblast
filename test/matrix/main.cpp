#define GRB_USE_APSPIE
//#define private public

#include <iostream>
#include <random>
#include <algorithm>

#include <cstdio>
#include <cstdlib>

#include <graphblas/mmio.hpp>
#include <graphblas/util.hpp>
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



BOOST_AUTO_TEST_CASE( matrix1 )
{
  std::vector<graphblas::Index> row_indices = {0, 1, 2};
  std::vector<graphblas::Index> col_indices = {1, 1, 1};
  std::vector<float> values = {1.0, 2.0, 3.0};
  graphblas::Matrix<float> a(3, 3);
  graphblas::Index nvals = 3;

  a.build( row_indices, col_indices, values, 3 );
  a.print();
}

BOOST_FIXTURE_TEST_CASE( matrix2, TestMatrix )
{
  std::vector<graphblas::Index> row_indices;
  std::vector<graphblas::Index> col_indices;
  std::vector<float> values;
	graphblas::Index nrows, ncols, nvals;

	// Read in sparse matrix
  if (argc < 2) {
    fprintf(stderr, "Usage: %s [matrix-market-filename]\n", argv[0]);
    exit(1);
  } else { 
	  readMtx( argv[1], row_indices, col_indices, values, nrows, ncols, nvals );
  }

  printArray( "row_indices", row_indices );
  printArray( "col_indices", col_indices );

	graphblas::Matrix<float> a( nrows,ncols );
	std::cout << nrows << " " << ncols << " " << nvals << std::endl;
	std::cout << row_indices.size() << " " << col_indices.size() << " " << 
			values.size() << std::endl;
  a.build( row_indices, col_indices, values, nvals );
  a.print();

  // Assume 8GB GPU RAM, 4B per float
  graphblas::Index MEM_SIZE = 10000000;//1000000000; 
  graphblas::Index max_ncols = std::min( 4, ncols );//MEM_SIZE/nrows, ncols );
  std::cout << "Restrict ncols to: " << max_ncols << std::endl;
  std::vector<float> dense(nrows*max_ncols, 1.0);
  printArray( "random", dense );
  graphblas::Matrix<float> b( nrows, max_ncols );

	int rhs[7] = {6, 7, 10, 11, 12, 21, 22};
	//BOOST_ASSERT_LIST( a.matrix.h_csrColInd, rhs, 7 );
}

BOOST_FIXTURE_TEST_CASE( matrix3, TestMatrix )
{
  std::vector<graphblas::Index> row_indices;
  std::vector<graphblas::Index> col_indices;
  std::vector<float> values;
	graphblas::Index nrows, ncols, nvals;

	// Read in sparse matrix
  if (argc < 2) {
    fprintf(stderr, "Usage: %s [matrix-market-filename]\n", argv[0]);
    exit(1);
  } else { 
	  readMtx( argv[1], row_indices, col_indices, values, nrows, ncols, nvals );
  }

  printArray( "row_indices", row_indices );
  printArray( "col_indices", col_indices );

	graphblas::Matrix<float> a( nrows,ncols );
	std::cout << nrows << " " << ncols << " " << nvals << std::endl;
	std::cout << row_indices.size() << " " << col_indices.size() << " " << 
			values.size() << std::endl;
  a.build( row_indices, col_indices, values, nvals );

  /*for( graphblas::Index i=0; i<nrows; i++ ) {
	  if( a.matrix.h_csrRowPtr[i]>a.matrix.h_csrRowPtr[i+1] )
			std::cout << i << " " << a.matrix.h_csrRowPtr[i] << " " << 
					a.matrix.h_csrRowPtr[i+1] << std::endl;
		BOOST_ASSERT( a.matrix.h_csrRowPtr[i]<=a.matrix.h_csrRowPtr[i+1] );
  }*/
}

BOOST_FIXTURE_TEST_CASE( matrix4, TestMatrix )
{
  std::vector<graphblas::Index> row_indices;
  std::vector<graphblas::Index> col_indices;
  std::vector<float> values;
	graphblas::Index nrows, ncols, nvals;

	// Read in sparse matrix
  if (argc < 2) {
    fprintf(stderr, "Usage: %s [matrix-market-filename]\n", argv[0]);
    exit(1);
  } else { 
	  readMtx( argv[1], row_indices, col_indices, values, nrows, ncols, nvals );
  }

  printArray( "row_indices", row_indices );
  printArray( "col_indices", col_indices );

	graphblas::Matrix<float> a( nrows,ncols );
	std::cout << nrows << " " << ncols << " " << nvals << std::endl;
	std::cout << row_indices.size() << " " << col_indices.size() << " " << 
			values.size() << std::endl;
  a.build( row_indices, col_indices, values, nvals );

  //BOOST_ASSERT( a.matrix.h_csrRowPtr[nrows]==nvals );
}

BOOST_AUTO_TEST_SUITE_END()
