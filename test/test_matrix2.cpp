#define GRB_USE_APSPIE

#include <iostream>
#include <random>
#include <algorithm>

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

	// Read in sparse matrix
  if (argc < 2) {
    fprintf(stderr, "Usage: %s [matrix-market-filename]\n", argv[0]);
    exit(1);
  } else { 
	  readMtx( argv[1], row_indices, col_indices, values, nrows, ncols, nvals );
  }

  printArray( "row_indices", row_indices );
  printArray( "col_indices", col_indices );

	graphblas::Matrix<float> a( nrows,ncols );
	std::cout << nrows << " " << ncols << " " << nvals << std::endl;
	std::cout << row_indices.size() << " " << col_indices.size() << " " << 
			values.size() << std::endl;
  a.build( row_indices, col_indices, values, nvals );

  // Generate random dense matrix
  std::random_device rnd_device;
  // Specify the engine and distribution.
  std::mt19937 mersenne_engine(rnd_device());
  std::uniform_real_distribution<float> dist(0.0, 1.0);

  auto gen = std::bind(dist, mersenne_engine);
  std::vector<float> dense(nrows*ncols);
  std::generate(begin(dense), end(dense), gen);
	printArray( "random", dense );
  graphblas::Matrix<float, graphblas::Dense> b( nrows, ncols );
  b.build( dense );

  return 0;
}
