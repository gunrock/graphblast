//#define GRB_USE_APSPIE
#define GRB_USE_SEQUENTIAL

#include <vector>
#include <cstdio>
#include <cstdlib>

#include <graphblas/graphblas.hpp>

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE mmio_suite

#include <boost/test/included/unit_test.hpp>
#include <test/test.hpp>

struct TestMMIO {
  TestMMIO() :
    argc( boost::unit_test::framework::master_test_suite().argc ),
    argv( boost::unit_test::framework::master_test_suite().argv ) {}

  int argc;
  char **argv;
};

BOOST_AUTO_TEST_SUITE( mmio_suite )

BOOST_FIXTURE_TEST_CASE( mmio1, TestMMIO )
{
  std::vector<graphblas::Index> row_indices;
  std::vector<graphblas::Index> col_indices;
  std::vector<float> values;
	graphblas::Index nrows;
	graphblas::Index ncols;
	graphblas::Index nvals;

  if (argc < 2) {
    fprintf(stderr, "Usage: %s [matrix-market-filename]\n", argv[0]);
    exit(1);
  } else
	readMtx( argv[1], row_indices, col_indices, values, nrows, ncols, nvals );

	int rhs[6] = {5, 6, 6, 6, 7, 7};
	BOOST_ASSERT_LIST( row_indices, rhs, 6 );
}

BOOST_AUTO_TEST_SUITE_END()
