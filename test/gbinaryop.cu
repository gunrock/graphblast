#define GRB_USE_CUDA
#define private public

#include <vector>
#include <iostream>

#include "graphblas/graphblas.hpp"
#include "test/test.hpp"

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE vec_suite

#include <boost/test/included/unit_test.hpp>
#include <boost/program_options.hpp>

// Tests 1: dup, 2: build, 3: extract, 4: gpuToCpu, 5: constructor
void testDup( const std::vector<int>& rhs )
{
  graphblas::Vector<int> vec1(rhs.size());
  graphblas::Index size = rhs.size();
  std::vector<int> lhs;
  CHECKVOID( vec1.build(&rhs, rhs.size()) );
  CHECKVOID( vec1.extractTuples(&lhs, &size) );
  BOOST_ASSERT_LIST( lhs, rhs, rhs.size() );

  graphblas::Vector<int> vec2(rhs.size());
  CHECKVOID( vec2.dup(&vec1) );

  size = rhs.size();
  CHECKVOID( vec2.extractTuples(&lhs, &size) );
  BOOST_ASSERT_LIST( lhs, rhs, rhs.size() );
}

void testReduce( const std::vector<int>& rhs )
{
  graphblas::Vector<int> vec1(rhs.size());
  graphblas::Index size = rhs.size();
  std::vector<int> lhs;
  vec1.build( &rhs, rhs.size() );

  graphblas::Descriptor desc;

  int val     = 0;
  int cpu_val = 0;
  graphblas::reduce<int,int>( &val, GrB_NULL, 
      graphblas::MultipliesMonoid<int>(), &vec1, &desc );
  for( int i=0; i<rhs.size(); i++ )
    cpu_val *= rhs[i];

  BOOST_ASSERT( val==cpu_val );
}

struct TestVector
{
  TestVector() :
    DEBUG(true) {}

  bool DEBUG;
};

BOOST_AUTO_TEST_SUITE( vec_suite )

BOOST_FIXTURE_TEST_CASE( vec1, TestVector )
{
  std::vector<int> vec = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  testDup( vec );
}

BOOST_FIXTURE_TEST_CASE( vec2, TestVector )
{
  std::vector<int> vec = {1, 2, 3, 4, 5, 6, 7, 8, 0, 2};
  testDup( vec );
}

BOOST_FIXTURE_TEST_CASE( vec3, TestVector )
{
  std::vector<int> vec = {1, 2, 3, 4, 5, 6, 7, 8, 0, 2};
  testReduce( vec );
}

BOOST_AUTO_TEST_SUITE_END()
