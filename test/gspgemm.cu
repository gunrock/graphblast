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

  // // Multiply
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


  std::cout << "done" << std::endl;
}
