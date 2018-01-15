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

void testVector( const std::vector<int>& rhs_ind, 
                 const std::vector<int>& rhs_val, 
                 int                     rhs_size )
{
  graphblas::Vector<int> vec1(rhs_size);
  graphblas::Index size = rhs_size;
  graphblas::Index nvals= rhs_ind.size();
  std::vector<int> lhs_ind;
  std::vector<int> lhs_val;
  CHECKVOID( vec1.build(&rhs_ind, &rhs_val, nvals, GrB_NULL) );
  CHECKVOID( vec1.extractTuples(&lhs_ind, &lhs_val, &nvals) );
  BOOST_ASSERT_LIST( lhs_ind, rhs_ind, nvals );
  BOOST_ASSERT_LIST( lhs_val, rhs_val, nvals );
}

void testNnew( const std::vector<int>& rhs_ind,
               const std::vector<int>& rhs_val, 
               int                     rhs_size )
{
  graphblas::Vector<int> vec1;
  CHECKVOID( vec1.nnew(rhs_size) );
  graphblas::Index size = rhs_size;
  graphblas::Index nvals= rhs_ind.size();
  std::vector<int> lhs_ind;
  std::vector<int> lhs_val;
  CHECKVOID( vec1.build(&rhs_ind, &rhs_val, nvals, GrB_NULL) );
  CHECKVOID( vec1.extractTuples(&lhs_ind, &lhs_val, &nvals) );
  BOOST_ASSERT_LIST( lhs_ind, rhs_ind, rhs_ind.size() );
  BOOST_ASSERT_LIST( lhs_val, rhs_val, rhs_val.size() );
}

void testDup( const std::vector<int>& rhs_ind,
              const std::vector<int>& rhs_val, 
              int                     rhs_size )
{
  graphblas::Vector<int> vec1(rhs_size);
  graphblas::Index size = rhs_size;
  graphblas::Index nvals= rhs_ind.size();
  std::vector<int> lhs_ind;
  std::vector<int> lhs_val;
  CHECKVOID( vec1.build(&rhs_ind, &rhs_val, nvals, GrB_NULL) );
  CHECKVOID( vec1.extractTuples(&lhs_ind, &lhs_val, &nvals) );
  BOOST_ASSERT_LIST( lhs_ind, rhs_ind, rhs_ind.size() );
  BOOST_ASSERT_LIST( lhs_val, rhs_val, rhs_ind.size() );

  graphblas::Vector<int> vec2(rhs_size);
  vec2.dup( &vec1 );

  nvals = rhs_ind.size();
  CHECKVOID( vec2.extractTuples(&lhs_ind, &lhs_val, &nvals) );
  BOOST_ASSERT_LIST( lhs_ind, rhs_ind, rhs_ind.size() );
  BOOST_ASSERT_LIST( lhs_val, rhs_val, rhs_ind.size() );
}

// Test properties:
// 1) vec_type_ is GrB_UNKNOWN
// 2) nvals_ is 0
// 3) nsize_ remains the same
void testClear( const std::vector<int>& rhs_ind,
                const std::vector<int>& rhs_val, 
                int                     rhs_size )
{
  graphblas::Vector<int> vec1(rhs_size);
  graphblas::Index size = rhs_size;
  graphblas::Index nvals= rhs_ind.size();
  std::vector<int> lhs_ind;
  std::vector<int> lhs_val;
  CHECKVOID( vec1.build(&rhs_ind, &rhs_val, nvals, GrB_NULL) );
  CHECKVOID( vec1.extractTuples(&lhs_ind, &lhs_val, &nvals) );
  BOOST_ASSERT_LIST( lhs_ind, rhs_ind, rhs_ind.size() );
  BOOST_ASSERT_LIST( lhs_val, rhs_val, rhs_ind.size() );
  
  CHECKVOID( vec1.clear() );
  lhs_ind.clear();
  lhs_val.clear();
  CHECKVOID( vec1.size(&size) );
  BOOST_ASSERT( size==rhs_size );

  CHECKVOID( vec1.nvals(&nvals) );
  BOOST_ASSERT( nvals==0 );

  graphblas::Storage storage;
  CHECKVOID( vec1.getStorage(&storage) );
  BOOST_ASSERT( storage==graphblas::GrB_UNKNOWN );
}

void testSize( const std::vector<int>& rhs_ind,
               const std::vector<int>& rhs_val, 
               int                     rhs_size )
{
  graphblas::Vector<int> vec1(rhs_size);
  graphblas::Index size = rhs_size;
  graphblas::Index nvals= rhs_ind.size();
  std::vector<int> lhs_ind;
  std::vector<int> lhs_val;
  CHECKVOID( vec1.build(&rhs_ind, &rhs_val, nvals, GrB_NULL) );
  CHECKVOID( vec1.extractTuples(&lhs_ind, &lhs_val, &nvals) );
  BOOST_ASSERT_LIST( lhs_ind, rhs_ind, rhs_ind.size() );
  BOOST_ASSERT_LIST( lhs_val, rhs_val, rhs_ind.size() );
  
  CHECKVOID( vec1.size(&size) );
  BOOST_ASSERT( size==rhs_size );
}

void testNvals( const std::vector<int>& rhs_ind,
                const std::vector<int>& rhs_val, 
                int                     rhs_size )
{
  graphblas::Vector<int> vec1(rhs_size);
  graphblas::Index size = rhs_size;
  graphblas::Index nvals= rhs_ind.size();
  std::vector<int> lhs_ind;
  std::vector<int> lhs_val;
  CHECKVOID( vec1.build(&rhs_ind, &rhs_val, nvals, GrB_NULL) );
  CHECKVOID( vec1.extractTuples(&lhs_ind, &lhs_val, &nvals) );
  BOOST_ASSERT_LIST( lhs_ind, rhs_ind, rhs_ind.size() );
  BOOST_ASSERT_LIST( lhs_val, rhs_val, rhs_ind.size() );

  CHECKVOID( vec1.nvals(&nvals) );
  BOOST_ASSERT( nvals==rhs_ind.size() );
}

void testBuild( const std::vector<int>& rhs_ind,
                const std::vector<int>& rhs_val, 
                int                     rhs_size )
{
  graphblas::Vector<int> vec1(rhs_size);
  graphblas::Index size = rhs_size;
  graphblas::Index nvals= rhs_ind.size();
  std::vector<int> lhs_ind;
  std::vector<int> lhs_val;
  CHECKVOID( vec1.build(&rhs_ind, &rhs_val, nvals, GrB_NULL) );
  CHECKVOID( vec1.extractTuples(&lhs_ind, &lhs_val, &nvals) );
  BOOST_ASSERT_LIST( lhs_ind, rhs_ind, rhs_ind.size() );
  BOOST_ASSERT_LIST( lhs_val, rhs_val, rhs_ind.size() );
}

void testSetElement( const std::vector<int>& rhs, int val, int ind )
{
}

void testExtractElement( const std::vector<int>& rhs )
{
}

void testExtractTuples( const std::vector<int>& rhs_ind,
                        const std::vector<int>& rhs_val, 
                        int                     rhs_size )
{
  graphblas::Vector<int> vec1(rhs_size);
  graphblas::Index size = rhs_size;
  graphblas::Index nvals= rhs_ind.size();
  std::vector<int> lhs_ind;
  std::vector<int> lhs_val;
  CHECKVOID( vec1.build(&rhs_ind, &rhs_val, nvals, GrB_NULL) );
  CHECKVOID( vec1.extractTuples(&lhs_ind, &lhs_val, &nvals) );
  BOOST_ASSERT_LIST( lhs_ind, rhs_ind, rhs_ind.size() );
  BOOST_ASSERT_LIST( lhs_val, rhs_val, rhs_ind.size() );
}

void testOperatorGetElement( const std::vector<int>& rhs )
{
}

void testFill( int size )
{
}

void testFillAscending( int size )
{
}

void testResize( const std::vector<int>& rhs_ind,
                 const std::vector<int>& rhs_val,
                 const int               rhs_size, 
                 const int               target_size )
{
  graphblas::Vector<int> vec1(rhs_size);
  graphblas::Index size = rhs_size;
  graphblas::Index nvals= rhs_ind.size();
  std::vector<int> lhs_ind;
  std::vector<int> lhs_val;
  CHECKVOID( vec1.build(&rhs_ind, &rhs_val, nvals, GrB_NULL) );
  CHECKVOID( vec1.extractTuples(&lhs_ind, &lhs_val, &nvals) );
  BOOST_ASSERT_LIST( lhs_ind, rhs_ind, rhs_ind.size() );
  BOOST_ASSERT_LIST( lhs_val, rhs_val, rhs_ind.size() );

  CHECKVOID( vec1.resize(target_size) );
  CHECKVOID( vec1.extractTuples(&lhs_ind, &lhs_val, &nvals) );
  BOOST_ASSERT_LIST( lhs_ind, rhs_ind, rhs_ind.size() );
  BOOST_ASSERT_LIST( lhs_val, rhs_val, rhs_ind.size() );

  CHECKVOID( vec1.size(&size) );
  BOOST_ASSERT( size==target_size );
}

void testSwap( const std::vector<int>& lhs_ind,
               const std::vector<int>& lhs_val,
               int                     lhs_size,
               const std::vector<int>& rhs_ind,
               const std::vector<int>& rhs_val, 
               int                     rhs_size )
{
  graphblas::Vector<int> vec1(lhs_size);
  graphblas::Index vec1_size = lhs_size;
  graphblas::Index vec1_nvals= lhs_ind.size();
  std::vector<int> vec1_ind;
  std::vector<int> vec1_val;
  CHECKVOID( vec1.build(&lhs_ind, &lhs_val, vec1_nvals, GrB_NULL) );

  graphblas::Vector<int> vec2(rhs_size);
  graphblas::Index vec2_size = rhs_size;
  graphblas::Index vec2_nvals= rhs_ind.size();
  std::vector<int> vec2_ind;
  std::vector<int> vec2_val;
  CHECKVOID( vec2.build(&rhs_ind, &rhs_val, vec2_nvals, GrB_NULL) );

  CHECKVOID( vec1.swap(&vec2) );
  CHECKVOID( vec1.nvals(&vec1_nvals) );
  CHECKVOID( vec2.nvals(&vec2_nvals) );
  CHECKVOID( vec1.extractTuples(&vec1_ind, &vec1_val, &vec1_nvals) );
  CHECKVOID( vec2.extractTuples(&vec2_ind, &vec2_val, &vec2_nvals) );

  CHECKVOID( vec1.print() );
  CHECKVOID( vec2.print() );
  BOOST_ASSERT_LIST( vec1_ind, rhs_ind, vec1_nvals );
  BOOST_ASSERT_LIST( vec1_val, rhs_val, vec1_nvals );
  BOOST_ASSERT_LIST( vec2_ind, lhs_ind, vec2_nvals );
  BOOST_ASSERT_LIST( vec2_val, lhs_val, vec2_nvals );
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
  std::vector<int> ind = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  std::vector<int> vec = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  testVector( ind, vec, 12 );
  testNnew(   ind, vec, 12 );
  testDup(    ind, vec, 12 );
  testClear(  ind, vec, 12 );
  testSize(   ind, vec, 12 );
  testNvals(  ind, vec, 12 );
  testBuild(  ind, vec, 12 );
  testExtractTuples( ind, vec, 12 );
}

BOOST_FIXTURE_TEST_CASE( vec2, TestVector )
{
  std::vector<int> ind = {1, 2, 3, 4, 5, 6, 7, 8, 0, 2};
  std::vector<int> vec = {1, 2, 3, 4, 5, 6, 7, 8, 0, 2};
  testResize( ind, vec, 12, 15 );
}

BOOST_FIXTURE_TEST_CASE( vec3, TestVector )
{
  std::vector<int> ind1 = {1, 2, 3, 4, 5, 6, 7, 8, 0, 2};
  std::vector<int> vec1 = {1, 2, 3, 4, 5, 6, 7, 8, 0, 2};
  std::vector<int> ind2 = {1, 2, 10, 4, 5, 6, 7, 8};
  std::vector<int> vec2 = {1, 2, 10, 4, 5, 6, 7, 8};
  testSwap( ind1, vec1, 10, ind2, vec2, 12 );
}

BOOST_AUTO_TEST_SUITE_END()
