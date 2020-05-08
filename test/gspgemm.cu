#define GRB_USE_CUDA
#define private public
#include <iostream>
#include <algorithm>
#include <string>

#include <cstdio>
#include <cstdlib>

#include <boost/program_options.hpp>

#include "graphblas/graphblas.hpp"
#include "test/test.hpp"

int main( int argc, char** argv )
{
  bool DEBUG = true;

  std::vector<graphblas::Index> a_row_indices, b_row_indices;
  std::vector<graphblas::Index> a_col_indices, b_col_indices;
  std::vector<float> a_values, b_values;
  graphblas::Index a_num_rows, a_num_cols, a_num_edges;
  graphblas::Index b_num_rows, b_num_cols, b_num_edges;
  char* dat_name;

  // Load A
  std::cout << "loading A" << std::endl;
  readMtx("../data/small/chesapeake.mtx", &a_row_indices, &a_col_indices,
      &a_values, &a_num_rows, &a_num_cols, &a_num_edges, 0, false, &dat_name);
  graphblas::Matrix<float> a(a_num_rows, a_num_cols);
  a.build(&a_row_indices, &a_col_indices, &a_values, a_num_edges, GrB_NULL,
     dat_name);
  if(DEBUG) a.print();

  // Load B
  std::cout << "loading B" << std::endl;
  readMtx("../data/small/chesapeake.mtx", &b_row_indices, &b_col_indices,
      &b_values, &b_num_rows, &b_num_cols, &b_num_edges, 0, false, &dat_name);
  graphblas::Matrix<float> b(b_num_rows, b_num_cols);
  b.build(&b_row_indices, &b_col_indices, &b_values, b_num_edges, GrB_NULL,
      dat_name);
  if(DEBUG) b.print();

  //
  graphblas::Matrix<float> c(a_num_rows, b_num_cols);
  graphblas::Descriptor desc;
  desc.descriptor_.debug_ = true;
  graphblas::mxm<float,float,float,float>(
      &c,
      GrB_NULL,
      GrB_NULL,
      graphblas::PlusMultipliesSemiring<float>(),
      &a,
      &b,
      &desc
  );
  if(DEBUG) c.print();

  // Multiply using GPU array initialization.
  graphblas::Matrix<float> A(a_num_rows, a_num_cols);
  graphblas::Matrix<float> B(b_num_rows, b_num_cols);
  graphblas::Matrix<float> C(a_num_rows, b_num_cols);

  A.build(a.matrix_.sparse_.d_csrRowPtr_, a.matrix_.sparse_.d_csrColInd_, a.matrix_.sparse_.d_csrVal_, a.matrix_.sparse_.nvals_);
  B.build(b.matrix_.sparse_.d_csrRowPtr_, b.matrix_.sparse_.d_csrColInd_, b.matrix_.sparse_.d_csrVal_, b.matrix_.sparse_.nvals_);

  desc.descriptor_.debug_ = true;

  graphblas::mxm<T, T, T, T>(&C, GrB_NULL, GrB_NULL, graphblas::PlusMultipliesSemiring<float>(),
                             &A, &B, &desc);

  // Multiply using CPU array initialization.
  // TODO(ctcyang): Add EXPECT_FAIL, because require pointers to be GPU.
  /*graphblas::Matrix<float> a_(a_num_rows, a_num_cols);
  graphblas::Matrix<float> b_(b_num_rows, b_num_cols);
  graphblas::Matrix<float> c_(a_num_rows, b_num_cols);

  a_.build(a.matrix_.sparse_.h_csrRowPtr_, a.matrix_.sparse_.h_csrColInd_, a.matrix_.sparse_.h_csrVal_, a.matrix_.sparse_.nvals_);
  b_.build(b.matrix_.sparse_.h_csrRowPtr_, b.matrix_.sparse_.h_csrColInd_, b.matrix_.sparse_.h_csrVal_, b.matrix_.sparse_.nvals_);

  desc.descriptor_.debug_ = true;

  graphblas::mxm<T, T, T, T>(&c_, GrB_NULL, GrB_NULL, graphblas::PlusMultipliesSemiring<float>(),
                             &a_, &b_, &desc);*/
}
