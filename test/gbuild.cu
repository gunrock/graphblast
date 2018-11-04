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

void testMatrixBuild( char const* mtx )
{
  std::vector<graphblas::Index> row_indices;
  std::vector<graphblas::Index> col_indices;
  std::vector<float> values;
  graphblas::Index nrows, ncols, nvals;
  graphblas::Info err;
  graphblas::Descriptor desc;
  char* dat_name;

  // Read in sparse matrix
  readMtx(mtx, row_indices, col_indices, values, nrows, ncols, nvals, 0, false,
      &dat_name);

  // Initialize sparse matrix A
  graphblas::Matrix<float> A(nrows, ncols);
  err = A.build(&row_indices, &col_indices, &values, nvals, GrB_NULL, dat_name);

  // Make a copy of GPU arrays from A
  int* d_csrRowPtr, *d_csrColInd;
  float* d_csrVal;
  cudaMalloc(&d_csrRowPtr, (nrows+1)*sizeof(int));
  cudaMalloc(&d_csrColInd, nvals*sizeof(int));
  cudaMalloc(&d_csrVal,    nvals*sizeof(float));

  cudaMemcpy(d_csrRowPtr, A.matrix_.sparse_.d_csrRowPtr_, (nrows+1)*sizeof(int),
             cudaMemcpyDeviceToDevice);
  cudaMemcpy(d_csrColInd, A.matrix_.sparse_.d_csrColInd_, nvals*sizeof(int),
             cudaMemcpyDeviceToDevice);
  cudaMemcpy(d_csrVal,    A.matrix_.sparse_.d_csrVal_,    nvals*sizeof(int),
             cudaMemcpyDeviceToDevice);

  // Initialize sparse matrix B using A's device arrays
  graphblas::Matrix<float> B(nrows, ncols);
  err = B.build( d_csrRowPtr, d_csrColInd, d_csrVal, nvals );

  std::vector<graphblas::Index> row_ind_t;
  std::vector<graphblas::Index> col_ind_t;
  std::vector<float> val_t;
  graphblas::Index nvals_t = nvals;
  err = B.extractTuples( &row_ind_t, &col_ind_t, &val_t, &nvals );
  BOOST_ASSERT( nvals == nvals_t );
  BOOST_ASSERT_LIST( row_indices, row_ind_t, nrows+1 );
  BOOST_ASSERT_LIST( col_indices, col_ind_t, nvals );
  BOOST_ASSERT_LIST( values, val_t, nvals );
}

void testDenseVectorBuild( std::vector<float>& input, int nvals )
{
  graphblas::Info err;
  graphblas::Descriptor desc;

  // Initialize dense vector v
  graphblas::Vector<float> v(nvals);
  err = v.build( &input, nvals );

  // Make a copy of GPU arrays from v
  float* d_val;
  cudaMalloc(&d_val, nvals*sizeof(float));

  cudaMemcpy(d_val, v.vector_.dense_.d_val_, nvals*sizeof(float),
             cudaMemcpyDeviceToDevice);

  // Initialize dense vector w using v's device arrays
  graphblas::Vector<float> w(nvals);
  err = w.build( d_val, nvals );

  std::vector<float> val_t;
  graphblas::Index nvals_t = nvals;
  err = w.extractTuples( &val_t, &nvals );
  BOOST_ASSERT( nvals == nvals_t );
  BOOST_ASSERT_LIST( input, val_t, nvals );
}

void testSparseVectorBuild( std::vector<graphblas::Index>& indices, 
                            std::vector<float>& values, int nsize, int nvals )
{
  graphblas::Info err;
  graphblas::Descriptor desc;

  // Initialize dense vector v
  graphblas::Vector<float> v(nsize);
  err = v.build( &indices, &values, nvals, GrB_NULL );

  // Make a copy of GPU arrays from v
  int*   d_ind;
  float* d_val;
  cudaMalloc(&d_ind, nvals*sizeof(int));
  cudaMalloc(&d_val, nvals*sizeof(float));

  cudaMemcpy(d_ind, v.vector_.sparse_.d_ind_, nvals*sizeof(int),
             cudaMemcpyDeviceToDevice);
  cudaMemcpy(d_val, v.vector_.sparse_.d_val_, nvals*sizeof(float),
             cudaMemcpyDeviceToDevice);

  // Initialize dense vector w using v's device arrays
  graphblas::Vector<float> w(nsize);
  err = w.build( d_ind, d_val, nvals );

  std::vector<graphblas::Index> ind_t;
  std::vector<float>            val_t;
  graphblas::Index nvals_t = nvals;
  err = w.extractTuples( &ind_t, &val_t, &nvals );
  BOOST_ASSERT( nvals == nvals_t );
  BOOST_ASSERT_LIST( indices, ind_t, nvals );
  BOOST_ASSERT_LIST( values,  val_t, nvals );
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
  testMatrixBuild( "data/small/test_cc.mtx" );
}

BOOST_FIXTURE_TEST_CASE( dup2, TestMatrix )
{
  testMatrixBuild( "data/small/test_bc.mtx" );
}

BOOST_FIXTURE_TEST_CASE( dup3, TestMatrix )
{
  std::vector<float> my_vec{1., 1., 3., 2., 2., 3., 3., 0., 1., 2., 2.};
  testDenseVectorBuild( my_vec, my_vec.size() );
}

BOOST_FIXTURE_TEST_CASE( dup4, TestMatrix )
{
  int nsize = 28;
  std::vector<graphblas::Index> ind{0 , 2 , 6 , 7 , 8 , 10, 13, 14, 17, 20, 27};
  std::vector<float>            val{1., 1., 3., 2., 2., 3., 3., 0., 1., 2., 2.};
  testSparseVectorBuild( ind, val, nsize, ind.size() );
}
BOOST_AUTO_TEST_SUITE_END()
