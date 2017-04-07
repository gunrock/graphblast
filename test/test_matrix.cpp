#define GRB_USE_APSPIE

#include <graphblas/graphblas.hpp>

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE matrix_suite

#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(matrix_suite)

BOOST_AUTO_TEST_CASE(matrix_build)
{
    graphblas::Matrix<double> matrix(3, 4);

        
}

BOOST_AUTO_TEST_SUITE_END()
