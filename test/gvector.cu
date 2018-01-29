#define GRB_USE_APSPIE
#define private public

#include <vector>
#include <iostream>

#include "graphblas/graphblas.hpp"
#include "test/test.hpp"

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE vec_suite

#include <boost/test/included/unit_test.hpp>
#include <boost/program_options.hpp>

void testConvert( const std::vector<int>& rhs_ind, 
                  const std::vector<int>& rhs_val, 
                  int                     rhs_size )
{
  graphblas::Vector<int> vec1(rhs_size);
  std::vector<int> lhs_ind;
  std::vector<int> lhs_val;
  CHECKVOID( vec1.fill(0) );
  CHECKVOID( vec1.setElement(1, 1) );
  vec1.print();
  vec1.vector_.dense_.nnz_ = 1;
  graphblas::Descriptor desc;
  desc.descriptor_.debug_ = true;
  desc.descriptor_.switchpoint_ = 0.9;
  vec1.vector_.ratio_ = 1.f;
  vec1.vector_.convert( 0, 1, &desc.descriptor_ );
  CHECKVOID( vec1.print() );

  graphblas::Index nvals = 1;
  CHECKVOID( vec1.extractTuples(&lhs_ind, &lhs_val, &nvals) );
  BOOST_ASSERT_LIST( lhs_ind, rhs_ind, nvals );
  BOOST_ASSERT_LIST( lhs_val, rhs_val, nvals );
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
  std::vector<int> ind = {1};
  std::vector<int> vec = {1};
  testConvert( ind, vec, 10 );
}

BOOST_AUTO_TEST_SUITE_END()
