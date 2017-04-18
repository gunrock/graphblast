#define GRB_USE_APSPIE
#define private public

#include <iostream>
#include <algorithm>

#include <cstdio>
#include <cstdlib>

#include <graphblas/mmio.hpp>
#include <graphblas/util.hpp>
#include <graphblas/graphblas.hpp>

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE spmm_suite

#include <boost/test/included/unit_test.hpp>
#include <test/test.hpp>

struct TestSPMM {
  TestSPMM() :
    argc( boost::unit_test::framework::master_test_suite().argc ),
    argv( boost::unit_test::framework::master_test_suite().argv ) {}

  int argc;
  char **argv;
};

BOOST_AUTO_TEST_SUITE(spmm_suite)

BOOST_FIXTURE_TEST_CASE( spmm2, TestSPMM )
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
	graphblas::Index MEM_SIZE = 1000000000;//1000000000; 
	//graphblas::Index max_ncols = std::min( 10, ncols );
	graphblas::Index max_ncols = std::min( MEM_SIZE/nrows, ncols );
	std::cout << "Restrict ncols to: " << max_ncols << std::endl;
	std::vector<float> dense(nrows*max_ncols, 1.0);
  std::cout << "Size: " << dense.size() << std::endl;
	printArray( "B matrix", dense );
  graphblas::Matrix<float> b( nrows, max_ncols );
  b.build( dense );

	graphblas::Matrix<float> c( nrows, max_ncols );
	// This statement is required if mxm() is used to build matrix rather than build()
	//c.storage( graphblas::Dense ); 
	graphblas::Semiring op;

	CpuTimer cpu_mxm;
	cpu_mxm.Start();
	graphblas::mxm<float, float, float>( c, op, a, b );
  cpu_mxm.Stop();

	c.print();
	float elapsed_mxm = cpu_mxm.ElapsedMillis();
	std::cout << "mxm: " << elapsed_mxm << " ms" << std::endl;
	//int rhs[7] = {0, 0, 0, 0, 0, 0, 1};
	//BOOST_ASSERT_LIST( a.matrix.h_csrRowPtr, rhs, 7 );
}

BOOST_AUTO_TEST_SUITE_END()
