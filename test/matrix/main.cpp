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

BOOST_AUTO_TEST_CASE( matrix2 )
{
  std::vector<graphblas::Index> row_indices = {2, 3, 4, 1, 3, 5, 4, 5, 6, 6, 7, 3, 6, 7, 7, 9, 10, 11, 10, 11};
	std::vector<graphblas::Index> col_indices = {1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 6, 8, 8, 8, 9, 9};
	std::vector<float> values (20, 1.0);
	graphblas::Matrix<float> a(11, 11);
  graphblas::Matrix<float> b(11, 11);
	graphblas::Index nvals = 20;
	a.build( row_indices, col_indices, values, 3 );
	b.build( row_indices, col_indices, values, 3 );
  graphblas::Matrix<float> c(11, 11);
  graphblas::Semiring op;
  graphblas::mxm<float, float, float>( c, op, a, b );

  /*std::vector<float> dense(nrows*max_ncols, 1.0);
  std::cout << "Size: " << dense.size() << std::endl;
  //printArray( "B matrix", dense );

  CpuTimer cpu_mxm;
  cpu_mxm.Start();
  graphblas::mxm<float, float, float>( c, op, a, b );
  cpu_mxm.Stop();

  c.print();
  float elapsed_mxm = cpu_mxm.ElapsedMillis();
  std::cout << "mxm: " << elapsed_mxm << " ms" << std::endl;
  //int rhs[7] = {0, 0, 0, 0, 0, 0, 1};
  //BOOST_ASSERT_LIST( a.matrix.h_csrRowPtr, rhs, 7 );*/
}

BOOST_AUTO_TEST_SUITE_END() 
