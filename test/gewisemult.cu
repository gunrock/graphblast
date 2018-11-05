#define GRB_USE_APSPIE
#define private public

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

void testeWiseMultVectorNomaskDenseDense( const std::vector<float>& u_val,
                                          const std::vector<float>& v_val,
                                          po::variables_map&        vm )
{
  std::vector<float> values;
  graphblas::Index nvals = u_val.size();
  graphblas::Info err;
  graphblas::Descriptor desc;
  desc.loadArgs(vm);

  std::vector<float> correct(nvals);
  for (int i = 0; i < nvals; ++i)
    correct[i] = u_val[i] * v_val[i];

  graphblas::Vector<float> u(nvals);
  err = u.build(&u_val, nvals);

  graphblas::Vector<float> v(nvals);
  err = v.build(&v_val, nvals);

  graphblas::Vector<float> vec(nvals);

  err = graphblas::eWiseMult<float, float, float, float>( &vec, GrB_NULL, 
      GrB_NULL, graphblas::PlusMultipliesSemiring<float>(), &u, &v, &desc );

  graphblas::Index nvals_t = nvals;
  err = vec.print();
  err = vec.extractTuples( &values, &nvals_t );
  BOOST_ASSERT( nvals == nvals_t );
  BOOST_ASSERT_LIST( values, correct, nvals );
}

void testeWiseMultVectorSparsemaskDenseDense( 
    const std::vector<graphblas::Index>& mask_ind,
    const std::vector<float>&            mask_val,
    const std::vector<float>&            u_val,
    const std::vector<float>&            v_val,
    po::variables_map&                   vm )
{
  std::vector<graphblas::Index> indices;
  std::vector<float> values;
  graphblas::Index nvals      = v_val.size();
  graphblas::Index mask_nvals = mask_ind.size();
  graphblas::Info err;
  graphblas::Descriptor desc;
  desc.loadArgs(vm);

  std::vector<graphblas::Index> correct_ind(mask_nvals);
  std::vector<float> correct_val(mask_nvals);
  for (int i = 0; i < mask_nvals; ++i)
  {
    graphblas::Index ind = mask_ind[i];
    if (mask_val[i] != 0)
      correct_val[i] = u_val[ind] * v_val[ind];
    else
      correct_val[i] = 0.f;
    correct_ind[i] = ind;
  }

  graphblas::Vector<float> u(nvals);
  err = u.build(&u_val, nvals);

  graphblas::Vector<float> v(nvals);
  err = v.build(&v_val, nvals);

  graphblas::Vector<float> mask(nvals);
  err = mask.build(&mask_ind, &mask_val, mask_nvals, GrB_NULL);

  graphblas::Vector<float> vec(nvals);

  err = graphblas::eWiseMult<float, float, float, float>( &vec, &mask, GrB_NULL,
      graphblas::PlusMultipliesSemiring<float>(), &u, &v, &desc );

  graphblas::Index nvals_t = mask_nvals;
  err = vec.print();
  err = vec.extractTuples( &indices, &values, &nvals_t );
  BOOST_ASSERT( mask_nvals == nvals_t );
  BOOST_ASSERT_LIST( indices, correct_ind, mask_nvals );
  BOOST_ASSERT_LIST( values, correct_val, mask_nvals );
}

void testeWiseMultVectorNomaskSparseDense( 
    const std::vector<graphblas::Index>& u_ind,
    const std::vector<float>&            u_val,
    const std::vector<float>&            v_val,
    po::variables_map&                   vm )
{
  std::vector<graphblas::Index> indices;
  std::vector<float> values;
  graphblas::Index nvals      = v_val.size();
  graphblas::Index u_nvals    = u_ind.size();
  graphblas::Info err;
  graphblas::Descriptor desc;
  desc.loadArgs(vm);

  std::vector<graphblas::Index> correct_ind(u_nvals);
  std::vector<float> correct_val(u_nvals);
  for (int i = 0; i < u_nvals; ++i)
  {
    graphblas::Index ind = u_ind[i];
    correct_ind[i] = ind;
    correct_val[i] = u_val[i] * v_val[ind];
  }

  graphblas::Vector<float> u(nvals);
  err = u.build(&u_ind, &u_val, u_nvals, GrB_NULL);

  graphblas::Vector<float> v(nvals);
  err = v.build(&v_val, nvals);

  graphblas::Vector<float> vec(nvals);

  err = graphblas::eWiseMult<float, float, float, float>( &vec, GrB_NULL,
      GrB_NULL, graphblas::PlusMultipliesSemiring<float>(), &u, &v, &desc );

  graphblas::Index nvals_t = u_nvals;
  err = vec.print();
  err = vec.extractTuples( &indices, &values, &nvals_t );
  BOOST_ASSERT( u_nvals == nvals_t );
  BOOST_ASSERT_LIST( indices, correct_ind, u_nvals );
  BOOST_ASSERT_LIST( values, correct_val, u_nvals );
}

void testeWiseMultVectorSparsemaskSparseDense( 
    const std::vector<graphblas::Index>& mask_ind,
    const std::vector<float>&            mask_val,
    const std::vector<graphblas::Index>& u_ind,
    const std::vector<float>&            u_val,
    const std::vector<float>&            v_val,
    po::variables_map&                   vm )
{
  std::vector<graphblas::Index> indices;
  std::vector<float> values;
  graphblas::Index nvals      = v_val.size();
  graphblas::Index mask_nvals = mask_ind.size();
  graphblas::Index u_nvals    = u_ind.size();
  graphblas::Info err;
  graphblas::Descriptor desc;
  desc.loadArgs(vm);

  std::vector<graphblas::Index> correct_ind(mask_nvals);
  std::vector<float> correct_val(mask_nvals);
  for (int i = 0; i < mask_nvals; ++i)
  {
    graphblas::Index ind = mask_ind[i];
    correct_val[i] = 0.f;
    if (mask_val[i] != 0 && v_val[ind] != 0.f)
    {
      int j;
      for (j = 0; j < u_nvals; ++j)
      {
        if (ind == u_ind[j])
          break;
        if (ind < u_ind[j])
        {
          j = -1;
          break;
        }
      }
      if (j != -1)
      {
        correct_val[i] = u_val[j] * v_val[ind];
      }
    }
    correct_ind[i] = ind;
  }

  graphblas::Vector<float> u(nvals);
  err = u.build(&u_ind, &u_val, u_nvals, GrB_NULL);

  graphblas::Vector<float> v(nvals);
  err = v.build(&v_val, nvals);

  graphblas::Vector<float> mask(nvals);
  err = mask.build(&mask_ind, &mask_val, mask_nvals, GrB_NULL);

  graphblas::Vector<float> vec(nvals);

  err = graphblas::eWiseMult<float, float, float, float>( &vec, &mask, GrB_NULL,
      graphblas::PlusMultipliesSemiring<float>(), &u, &v, &desc );

  graphblas::Index nvals_t = mask_nvals;
  err = vec.print();
  err = vec.extractTuples( &indices, &values, &nvals_t );
  BOOST_ASSERT( mask_nvals == nvals_t );
  BOOST_ASSERT_LIST( indices, correct_ind, mask_nvals );
  BOOST_ASSERT_LIST( values, correct_val, mask_nvals );
}

void testeWiseMultVectorDensemaskSparseDense( 
    const std::vector<float>&            mask_val,
    const std::vector<graphblas::Index>& u_ind,
    const std::vector<float>&            u_val,
    const std::vector<float>&            v_val,
    po::variables_map&                   vm )
{
  std::vector<graphblas::Index> indices;
  std::vector<float> values;
  graphblas::Index nvals   = v_val.size();
  graphblas::Index u_nvals = u_ind.size();
  graphblas::Info err;
  graphblas::Descriptor desc;
  desc.loadArgs(vm);

  std::vector<graphblas::Index> correct_ind(u_nvals);
  std::vector<float> correct_val(u_nvals);
  for (int i = 0; i < u_nvals; ++i)
  {
    graphblas::Index ind = u_ind[i];
    correct_val[i] = 0.f;
    if (mask_val[ind] != 0.f && v_val[ind] != 0.f)
    {
      correct_val[i] = u_val[i] * v_val[ind];
    }
    correct_ind[i] = ind;
  }

  graphblas::Vector<float> u(nvals);
  err = u.build(&u_ind, &u_val, u_nvals, GrB_NULL);

  graphblas::Vector<float> v(nvals);
  err = v.build(&v_val, nvals);

  graphblas::Vector<float> mask(nvals);
  err = mask.build(&mask_val, nvals);

  graphblas::Vector<float> vec(nvals);

  err = graphblas::eWiseMult<float, float, float, float>( &vec, &mask, GrB_NULL,
      graphblas::PlusMultipliesSemiring<float>(), &u, &v, &desc );

  graphblas::Index nvals_t = u_nvals;
  err = vec.print();
  err = vec.extractTuples( &indices, &values, &nvals_t );
  BOOST_ASSERT( u_nvals == nvals_t );
  BOOST_ASSERT_LIST( indices, correct_ind, u_nvals );
  BOOST_ASSERT_LIST( values, correct_val, u_nvals );
}

void testeWiseMultVectorNomaskDenseDenseGreater(
    const std::vector<float>& u_val,
    const std::vector<float>& v_val,
    po::variables_map&        vm )
{
  std::vector<float> values;
  graphblas::Index nvals = u_val.size();
  graphblas::Info err;
  graphblas::Descriptor desc;
  desc.loadArgs(vm);

  std::vector<float> correct(nvals);
  for (int i = 0; i < nvals; ++i) {
    if (u_val[i] == 0 || v_val[i] == 0)
      correct[i] = 0;
    else
      correct[i] = u_val[i] > v_val[i];
  }
  printArray("correct", correct, nvals);

  graphblas::Vector<float> u(nvals);
  err = u.build(&u_val, nvals);

  graphblas::Vector<float> v(nvals);
  err = v.build(&v_val, nvals);

  graphblas::Vector<float> vec(nvals);

  err = graphblas::eWiseMult<float, float, float, float>( &vec, GrB_NULL, 
      GrB_NULL, graphblas::PlusGreaterSemiring<float>(), &u, &v, &desc );

  graphblas::Index nvals_t = nvals;
  err = vec.print();
  err = vec.extractTuples( &values, &nvals_t );
  BOOST_ASSERT( nvals == nvals_t );
  BOOST_ASSERT_LIST( values, correct, nvals );
}

void testeWiseMultVectorNomaskSparseDenseGreater( 
    const std::vector<graphblas::Index>& u_ind,
    const std::vector<float>&            u_val,
    const std::vector<float>&            v_val,
    po::variables_map&                   vm )
{
  std::vector<graphblas::Index> indices;
  std::vector<float> values;
  graphblas::Index nvals      = v_val.size();
  graphblas::Index u_nvals    = u_ind.size();
  graphblas::Info err;
  graphblas::Descriptor desc;
  desc.loadArgs(vm);

  std::vector<graphblas::Index> correct_ind(u_nvals);
  std::vector<float> correct_val(u_nvals);
  for (int i = 0; i < u_nvals; ++i)
  {
    graphblas::Index ind = u_ind[i];
    correct_ind[i] = ind;
    correct_val[i] = u_val[i] > v_val[ind];
  }

  graphblas::Vector<float> u(nvals);
  err = u.build(&u_ind, &u_val, u_nvals, GrB_NULL);

  graphblas::Vector<float> v(nvals);
  err = v.build(&v_val, nvals);

  graphblas::Vector<float> vec(nvals);

  err = graphblas::eWiseMult<float, float, float, float>( &vec, GrB_NULL,
      GrB_NULL, graphblas::PlusGreaterSemiring<float>(), &u, &v, &desc );

  graphblas::Index nvals_t = u_nvals;
  err = vec.print();
  err = vec.extractTuples( &indices, &values, &nvals_t );
  BOOST_ASSERT( u_nvals == nvals_t );
  BOOST_ASSERT_LIST( indices, correct_ind, u_nvals );
  BOOST_ASSERT_LIST( values, correct_val, u_nvals );
}

struct TestMatrix
{
  TestMatrix() :
    DEBUG(true) {}

  bool DEBUG;
};

BOOST_AUTO_TEST_SUITE(dup_suite)

BOOST_FIXTURE_TEST_CASE( dup1, TestMatrix )
{
  int argc = 3;
  char* argv[] = {"app", "--debug", "1"};
  po::variables_map vm;
  parseArgs( argc, argv, vm );
  std::vector<float> u_val{ 1., 1., 3., 2., 2., 0., 3., 0., 1., 0., 2. };
  std::vector<float> v_val{ 1., 0., 3., 2., 2., 3., 0., 0., 1., 2., 2. };
  testeWiseMultVectorNomaskDenseDense(u_val, v_val, vm);
}

BOOST_FIXTURE_TEST_CASE( dup2, TestMatrix )
{
  int argc = 3;
  char* argv[] = {"app", "--debug", "1"};
  po::variables_map vm;
  parseArgs( argc, argv, vm );
  int n = 10000;
  std::vector<float> u_val(n, 0.f);
  std::vector<float> v_val(n, 0.f);
  for (int i = 0; i < n/2; ++i)
  {
    u_val[i] = static_cast<float>(i);
    v_val[i] = (i % 2 == 0) ? 0.f : static_cast<float>(i);
  }
  for (int i = n/2; i < n; ++i)
  {
    u_val[i] = (i % 2 == 0) ? 0.f : static_cast<float>(i);
    v_val[i] = static_cast<float>(i);
  }
  testeWiseMultVectorNomaskDenseDense(u_val, v_val, vm);
}

BOOST_FIXTURE_TEST_CASE( dup3, TestMatrix )
{
  int argc = 3;
  char* argv[] = {"app", "--debug", "1"};
  po::variables_map vm;
  parseArgs( argc, argv, vm );
  std::vector<graphblas::Index> mask_ind{ 0,  2,  4,  6,  7,  8 };
  std::vector<float>            mask_val{ 3., 2., 2., 0., 3., 0.};
  std::vector<float>            u_val   { 3., 2., 2., 0., 3., 1., 2., 3., 5.};
  std::vector<float>            v_val   { 3., 2., 2., 3., 0., 0., 0., 2., 2.};
  testeWiseMultVectorSparsemaskDenseDense(mask_ind, mask_val, u_val, v_val, vm);
}

BOOST_FIXTURE_TEST_CASE( dup5, TestMatrix )
{
  int argc = 3;
  char* argv[] = {"app", "--debug", "1"};
  po::variables_map vm;
  parseArgs( argc, argv, vm );
  std::vector<graphblas::Index> u_ind   { 1,  2,  5,  6,  7,  8 }; 
  std::vector<float>            u_val   { 3., 2., 2., 0., 3., 1.};
  std::vector<float>            v_val   { 3., 2., 2., 3., 0., 0., 3., 2., 2.};
  testeWiseMultVectorNomaskSparseDense(u_ind, u_val, v_val, vm);
}

BOOST_FIXTURE_TEST_CASE( dup7, TestMatrix )
{
  int argc = 3;
  char* argv[] = {"app", "--debug", "1"};
  po::variables_map vm;
  parseArgs( argc, argv, vm );
  std::vector<graphblas::Index> mask_ind{ 0,  2,  4,  6,  7,  8 };
  std::vector<float>            mask_val{ 3., 2., 2., 0., 3., 0.};
  std::vector<graphblas::Index> u_ind   { 1,  2,  5,  6,  7,  8 }; 
  std::vector<float>            u_val   { 3., 2., 2., 0., 3., 1.};
  std::vector<float>            v_val   { 3., 2., 2., 3., 0., 0., 0., 2., 2.};
  testeWiseMultVectorSparsemaskSparseDense(mask_ind, mask_val, u_ind, u_val,
      v_val, vm);
}

BOOST_FIXTURE_TEST_CASE( dup9, TestMatrix )
{
  int argc = 3;
  char* argv[] = {"app", "--debug", "1"};
  po::variables_map vm;
  parseArgs( argc, argv, vm );
  std::vector<float>            mask_val{ 3., 2., 2., 0., 3., 0., 1., 1., 0.};
  std::vector<graphblas::Index> u_ind   { 1,  2,  5,  6,  7,  8 }; 
  std::vector<float>            u_val   { 3., 2., 2., 0., 3., 1.};
  std::vector<float>            v_val   { 3., 2., 2., 3., 0., 0., 0., 2., 2.};
  testeWiseMultVectorDensemaskSparseDense(mask_val, u_ind, u_val, v_val, vm);
}

BOOST_FIXTURE_TEST_CASE( dup11, TestMatrix )
{
  int argc = 3;
  char* argv[] = {"app", "--debug", "1"};
  po::variables_map vm;
  parseArgs( argc, argv, vm );
  std::vector<float> u_val{ 1., 1., 3., 2., 2., 0., 3., 0., 1., 0., 2. };
  std::vector<float> v_val{ 1., 0., 3., 2., 2., 3., 0., 0., 1., 2., 2. };
  testeWiseMultVectorNomaskDenseDenseGreater(u_val, v_val, vm);
}

BOOST_FIXTURE_TEST_CASE( dup12, TestMatrix )
{
  int argc = 3;
  char* argv[] = {"app", "--debug", "1"};
  po::variables_map vm;
  parseArgs( argc, argv, vm );
  int n = 1;
  //int n = 10000;
  std::vector<float> u_val(n, 0.f);
  std::vector<float> v_val(n, 0.f);
  for (int i = 0; i < n/2; ++i)
  {
    u_val[i] = 1.;
    v_val[i] = 0.;
  }
  for (int i = n/2; i < n; ++i)
  {
    u_val[i] = 0.;
    v_val[i] = 1.;
  }
  testeWiseMultVectorNomaskDenseDenseGreater(u_val, v_val, vm);
}

BOOST_FIXTURE_TEST_CASE( dup13, TestMatrix )
{
  int argc = 3;
  char* argv[] = {"app", "--debug", "1"};
  po::variables_map vm;
  parseArgs( argc, argv, vm );
  std::vector<graphblas::Index> u_ind   { 1,  2,  5,  6,  7,  8 }; 
  std::vector<float>            u_val   { 3., 1., 2.,-1., 3., 1.};
  std::vector<float>            v_val   { 3., 2., 2., 3., 0., 0., 0., 2., 2.};
  testeWiseMultVectorNomaskSparseDenseGreater(u_ind, u_val, v_val, vm);
}

BOOST_AUTO_TEST_SUITE_END()
