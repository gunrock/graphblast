#define GRB_USE_APSPIE
//#define GrB_PLUS graphblas::BinaryOp<float,float,float>( std::plus<float>() )

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
  graphblas::Vector<int> vec1;
  graphblas::Index size = rhs.size();
  std::vector<int> lhs;
  vec1.build( &rhs, rhs.size() );
  vec1.extractTuples( &lhs, &size );
  BOOST_ASSERT_LIST( lhs, rhs, rhs.size() );

  graphblas::BinaryOp<float,float,float> GrB_PLUS_FP32;
  float a = GrB_PLUS_FP32(3.f,2.f);
  std::cout << a << std::endl;

  GrB_PLUS_FP32.nnew( std::multiplies<float>() );
  a = GrB_PLUS_FP32(3.f,2.f);
  std::cout << a << std::endl;

  a = graphblas::BinaryOp<float,float,float>( std::minus<float>() )(3.f,2.f);
  std::cout << a << std::endl;

  //a = GrB_PLUS(3.f,2.f);
  //std::cout << a << std::endl;

  typedef graphblas::BinaryOp<float,float,float>( std::minus<float>() )::operator() GrB_PLUS_OP;
  a = GrB_PLUS_OP(3.f,2.f);
  std::cout << a << std::endl;

  graphblas::Vector<int> vec2;
  vec2.dup( &vec1 );

  size = rhs.size();
  vec2.extractTuples( &lhs, &size );
  BOOST_ASSERT_LIST( lhs, rhs, rhs.size() );
}

void testResize( const std::vector<int>& rhs, const int nvals )
{
  graphblas::Vector<int> vec1;
  graphblas::Index size = rhs.size();
  std::vector<int> lhs;
  vec1.build( &rhs, rhs.size() );
  vec1.extractTuples( &lhs, &size );
  BOOST_ASSERT_LIST( lhs, rhs, rhs.size() );

  vec1.resize( nvals );
  size = rhs.size();
  vec1.extractTuples( &lhs, &size );
  BOOST_ASSERT_LIST( lhs, rhs, rhs.size() );
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
  testResize( vec, 15 );
}

BOOST_AUTO_TEST_SUITE_END()
