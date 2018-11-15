#define GRB_USE_APSPIE
#define private public

#include <iostream>
#include <algorithm>
#include <string>

#include <cstdio>
#include <cstdlib>

#include "graphblas/graphblas.hpp"
#include "test/test.hpp"

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE dup_suite

#include <boost/test/included/unit_test.hpp>
#include <boost/program_options.hpp>

void testVxmDenseSparse( char const*               mtx,
                         const std::vector<float>& vec,
                         int                       use_mask,
                         po::variables_map&        vm )
{
  std::vector<graphblas::Index> row_indices;
  std::vector<graphblas::Index> col_indices;
  std::vector<float> values;
  graphblas::Index nrows, ncols, nvals;
  char* dat_name;

  // Read in sparse matrix
  readMtx(mtx, row_indices, col_indices, values, nrows, ncols, 
      nvals, 0, false, &dat_name);

  // Matrix A
  graphblas::Matrix<float> a(nrows, ncols);
  a.build(&row_indices, &col_indices, &values, nvals, GrB_NULL, dat_name);
  a.nrows(&nrows);
  a.ncols(&ncols);
  a.nvals(&nvals);

  std::vector<float> correct(nrows, 0.f);
  for (graphblas::Index row = 0; row < nrows; ++row)
  {
    graphblas::Index row_start = a.matrix_.sparse_.h_csrRowPtr_[row];
    graphblas::Index row_end   = a.matrix_.sparse_.h_csrRowPtr_[row+1];
    float vec_val = vec[row];

    for (; row_start < row_end; ++row_start)
    {
      graphblas::Index col = a.matrix_.sparse_.h_csrColInd_[row_start];
      float val = a.matrix_.sparse_.h_csrVal_[row_start];

      correct[col] += val*vec_val;
    }
  }

  // Vector x
  graphblas::Vector<float> x(nrows);
  x.build(&vec, vec.size());

  // Vector y
  graphblas::Vector<float> y(nrows);

  // Descriptor
  graphblas::Descriptor desc;
  desc.loadArgs(vm);
  //desc.set(graphblas::GrB_MXVMODE, graphblas::GrB_PUSHONLY) );

  // Compute
  graphblas::vxm<float, float, float, float>(&y, GrB_NULL, GrB_NULL, 
      graphblas::PlusMultipliesSemiring<float>(), &x, &a, &desc);

  y.vector_.sparse2dense(0.f, &desc.descriptor_);
  y.print();
  printArray("correct", correct, nrows);
  y.extractTuples( &values, &nrows );
  BOOST_ASSERT( nrows == correct.size() );
  BOOST_ASSERT_LIST( values, correct, nrows );
}

void testVxmSparseSparse( char const*                          mtx,
                          const std::vector<graphblas::Index>& vec_ind,
                          const std::vector<float>&            vec_val,
                          int                                  use_mask,
                          po::variables_map&                   vm )
{
  std::vector<graphblas::Index> row_indices;
  std::vector<graphblas::Index> col_indices;
  std::vector<float> values;
  graphblas::Index nrows, ncols, nvals;
  char* dat_name;

  // Read in sparse matrix
  readMtx(mtx, row_indices, col_indices, values, nrows, ncols, 
      nvals, 0, false, &dat_name);

  // Matrix A
  graphblas::Matrix<float> a(nrows, ncols);
  a.build(&row_indices, &col_indices, &values, nvals, GrB_NULL, dat_name);
  a.nrows(&nrows);
  a.ncols(&ncols);
  a.nvals(&nvals);

  std::vector<float> correct(nrows, 0.f);
  for (graphblas::Index i = 0; i < vec_ind.size(); ++i)
  {
    graphblas::Index row = vec_ind[i];
    graphblas::Index val = vec_val[i];

    graphblas::Index row_start = a.matrix_.sparse_.h_csrRowPtr_[row];
    graphblas::Index row_end   = a.matrix_.sparse_.h_csrRowPtr_[row+1];

    for (; row_start < row_end; ++row_start)
    {
      graphblas::Index col = a.matrix_.sparse_.h_csrColInd_[row_start];
      float dest_val = a.matrix_.sparse_.h_csrVal_[row_start];

      correct[col] += dest_val*val;
    }
  }

  // Vector x
  graphblas::Vector<float> x(nrows);
  x.build(&vec_ind, &vec_val, vec_ind.size(), GrB_NULL);

  // Vector y
  graphblas::Vector<float> y(nrows);

  // Descriptor
  graphblas::Descriptor desc;
  desc.loadArgs(vm);
  //desc.set(graphblas::GrB_MXVMODE, graphblas::GrB_PUSHONLY) );

  // Compute
  graphblas::vxm<float, float, float, float>(&y, GrB_NULL, GrB_NULL, 
      graphblas::PlusMultipliesSemiring<float>(), &x, &a, &desc);

  y.vector_.sparse2dense(0.f, &desc.descriptor_);
  y.print();
  printArray("correct", correct, nrows);
  y.extractTuples( &values, &nrows );
  BOOST_ASSERT( nrows == correct.size() );
  BOOST_ASSERT_LIST( values, correct, nrows );
}

void testVxmSparseSparseDenseMask( 
    char const*                          mtx,
    const std::vector<graphblas::Index>& vec_ind,
    const std::vector<float>&            vec_val,
    const std::vector<float>&            mask_val,
    int                                  use_mask,
    po::variables_map&                   vm )
{
  std::vector<graphblas::Index> row_indices;
  std::vector<graphblas::Index> col_indices;
  std::vector<float> values;
  graphblas::Index nrows, ncols, nvals;
  char* dat_name;

  // Read in sparse matrix
  readMtx(mtx, row_indices, col_indices, values, nrows, ncols, 
      nvals, 0, false, &dat_name);

  // Matrix A
  graphblas::Matrix<float> a(nrows, ncols);
  a.build(&row_indices, &col_indices, &values, nvals, GrB_NULL, dat_name);
  a.nrows(&nrows);
  a.ncols(&ncols);
  a.nvals(&nvals);

  std::vector<float> correct(nrows, 0.f);
  for (graphblas::Index i = 0; i < vec_ind.size(); ++i)
  {
    graphblas::Index row = vec_ind[i];
    graphblas::Index val = vec_val[i];

    graphblas::Index row_start = a.matrix_.sparse_.h_csrRowPtr_[row];
    graphblas::Index row_end   = a.matrix_.sparse_.h_csrRowPtr_[row+1];

    for (; row_start < row_end; ++row_start)
    {
      graphblas::Index col = a.matrix_.sparse_.h_csrColInd_[row_start];
      float dest_val = a.matrix_.sparse_.h_csrVal_[row_start];

      correct[col] += dest_val*val;
    }
  }
  for (graphblas::Index i = 0; i < correct.size(); ++i)
  {
    if (mask_val[i] == 0)
    {
      correct[i] = 0.f;
    }
  }

  // Vector x
  graphblas::Vector<float> x(nrows);
  x.build(&vec_ind, &vec_val, vec_ind.size(), GrB_NULL);

  // Vector mask
  graphblas::Vector<float> mask(nrows);
  mask.build(&mask_val, mask_val.size());

  // Vector y
  graphblas::Vector<float> y(nrows);

  // Descriptor
  graphblas::Descriptor desc;
  desc.loadArgs(vm);
  //desc.set(graphblas::GrB_MXVMODE, graphblas::GrB_PUSHONLY) );

  // Compute
  graphblas::vxm<float, float, float, float>(&y, &mask, GrB_NULL, 
      graphblas::PlusMultipliesSemiring<float>(), &x, &a, &desc);

  y.vector_.sparse2dense(0.f, &desc.descriptor_);
  y.print();
  printArray("correct", correct, nrows);
  y.extractTuples( &values, &nrows );
  BOOST_ASSERT( nrows == correct.size() );
  BOOST_ASSERT_LIST( values, correct, nrows );
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
  std::vector<float> vec(11, 2.f);
  testVxmDenseSparse( "data/small/test_cc.mtx", vec, 0, vm );
}

BOOST_FIXTURE_TEST_CASE( dup2, TestMatrix )
{
  int argc = 3;
  char* argv[] = {"app", "--debug", "1"};
  po::variables_map vm;
  parseArgs( argc, argv, vm );
  std::vector<float> vec{0, 13,7, 0, 1, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 4, 4, 5, 4};
  testVxmDenseSparse( "data/small/test_sgm.mtx", vec, 0, vm );
}

BOOST_FIXTURE_TEST_CASE( dup3, TestMatrix )
{
  int argc = 3;
  char* argv[] = {"app", "--debug", "1"};
  po::variables_map vm;
  parseArgs( argc, argv, vm );
  std::vector<graphblas::Index> vec_ind{0, 1, 4, 6, 8, 10};
  std::vector<float>            vec_val{1.,2.,3.,4.,3.,10.};
  testVxmSparseSparse( "data/small/test_cc.mtx", vec_ind, vec_val, 0, vm );
}

BOOST_FIXTURE_TEST_CASE( dup4, TestMatrix )
{
  int argc = 3;
  char* argv[] = {"app", "--debug", "1"};
  po::variables_map vm;
  parseArgs( argc, argv, vm );
  std::vector<graphblas::Index> vec_ind{1,  2, 4, 16, 17, 18, 19};
  std::vector<float>            vec_val{13.,7.,1.,4., 4., 5., 4.};
  testVxmSparseSparse( "data/small/test_sgm.mtx", vec_ind, vec_val, 0, vm );
}

BOOST_FIXTURE_TEST_CASE( dup5, TestMatrix )
{
  int argc = 3;
  char* argv[] = {"app", "--debug", "1"};
  po::variables_map vm;
  parseArgs( argc, argv, vm );
  std::vector<graphblas::Index> vec_ind{ 0, 1, 4, 6, 8, 10};
  std::vector<float>            vec_val{ 1.,2.,3.,4.,3.,10.};
  std::vector<float>            mask_val{1.,0.,0.,1.,0., 1.,1.,1.,1.,1.};
  testVxmSparseSparseDenseMask( "data/small/test_cc.mtx", vec_ind, vec_val, mask_val, 0, vm );
}

BOOST_FIXTURE_TEST_CASE( dup6, TestMatrix )
{
  int argc = 3;
  char* argv[] = {"app", "--debug", "1"};
  po::variables_map vm;
  parseArgs( argc, argv, vm );
  std::vector<graphblas::Index> vec_ind{ 1,  2, 4, 16, 17, 18, 19};
  std::vector<float>            vec_val{ 13.,7.,1.,4., 4., 5., 4.};
  std::vector<float>            mask_val{1., 1.,0.,1., 0., 0., 1., 0., 0., 1.,
                                         0., 1.,1.,1., 1., 1., 1., 1., 1., 1.};
  testVxmSparseSparseDenseMask( "data/small/test_sgm.mtx", vec_ind, vec_val, mask_val, 0, vm );
}
BOOST_AUTO_TEST_SUITE_END()
