#define GRB_USE_SEQUENTIAL

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

void testCountUnique( std::vector<int>& array, const int correct )
{
  int count = countUnique( array );
  //std::cout << count << "=" << correct << std::endl;
  BOOST_ASSERT( count==correct );
}

BOOST_AUTO_TEST_SUITE( util_suite )

BOOST_AUTO_TEST_CASE( util1 )
{
  std::vector<int> adj_deg   = {1, 1, 3, 2, 2, 3, 3};
  testCountUnique( adj_deg, 3 );
}

BOOST_AUTO_TEST_CASE( util2 )
{
  std::vector<int> adj_t_deg = {3, 3, 3, 2, 3, 1, 0};
  testCountUnique( adj_t_deg, 4 );
}

BOOST_AUTO_TEST_CASE( util3 )
{
  std::vector<int> adj_deg   = {4, 3, 3, 1, 1, 2, 4, 4, 1, 3};
  testCountUnique( adj_deg, 4 );
}

BOOST_AUTO_TEST_CASE( util4 )
{
  std::vector<int> adj_t_deg = {4, 4, 2, 1, 2, 1, 3, 3, 4, 1};
  testCountUnique( adj_t_deg, 4 );
}

BOOST_AUTO_TEST_CASE( util5 )
{
  std::vector<int> correct_ind = {1, 2, 3, 4, 5, 6};
  testCountUnique( correct_ind, 6 );
}

BOOST_AUTO_TEST_CASE( util6 )
{
  std::vector<int> correct_val = {1, 1, 1, 2, 2, 1};
  testCountUnique( correct_val, 2 );
}

BOOST_AUTO_TEST_CASE( util7 )
{
  std::vector<int> correct_ind = {1, 2, 3, 4, 5, 6, 7, 8};
  testCountUnique( correct_ind, 8 );
}

BOOST_AUTO_TEST_CASE( util8 )
{
  std::vector<int> correct_val = {1, 1, 1, 2, 2, 1, 1, 10};
  testCountUnique( correct_val, 3 );
}

/*BOOST_FIXTURE_TEST_CASE( dup12, TestMatrix )
{
  std::map<graphblas::Index,graphblas::Index> out_edge;
  out_edge.insert({{2, 1}, {3, 0}, {4,1}});
  std::vector<graphblas::Index> correct = {1, 2, 3, 0, 2, 4, 3, 4, -1, 5, 6, 2, 6, -1, 6, 8, 9, 10, 9, 10};
  testEwiseSubCol( "data/small/test_cc.tsv", out_edge, correct);
}

BOOST_FIXTURE_TEST_CASE( dup18, TestMatrix )
{
  std::map<graphblas::Index,graphblas::Index> out_edge;
  out_edge.insert( {{1,1}, {2,0}, {4,1}, {5,2}, {6,1}, {7,1}, {8,10}} );
  std::vector<graphblas::Index> correct_ind = {1, 2, 3, 0, 2, 4, 5, 3, 4, 5, 5, 6, 2, 5, 6, 5, 6, 5, 5, 8, 9, 10, 5, 9, 10 };
  std::vector<graphblas::T>     correct_val = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 10, 1, 1 };
  testEwiseAddCol( "data/small/test_cc.tsv", out_edge, correct_ind, correct_val);
}*/

BOOST_AUTO_TEST_SUITE_END()
