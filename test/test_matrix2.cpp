#define GRB_USE_APSPIE

#include <iostream>

#include <cstdio>
#include <cstdlib>

#include <graphblas/mmio.hpp>
#include <graphblas/util.hpp>
#include <graphblas/graphblas.hpp>

int main(int argc, char **argv)
{
  std::vector<graphblas::Index> row_indices;
  std::vector<graphblas::Index> col_indices;
  std::vector<float> values;
	graphblas::Index nrows, ncols, nvals;

  if (argc < 2) {
    fprintf(stderr, "Usage: %s [matrix-market-filename]\n", argv[0]);
    exit(1);
  } else { 
	  readMtx( argv[1], row_indices, col_indices, values, nrows, ncols, nvals );
  }

	graphblas::Matrix<float> a(nrows,ncols);
	std::cout << row_indices.size();
  a.build( row_indices, col_indices, values, nvals );

  return 0;
}
