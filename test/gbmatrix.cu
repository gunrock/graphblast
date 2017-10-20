#define GRB_USE_APSPIE
//#define private public

#include <iostream>
#include <algorithm>
#include <string>

#include <cstdio>
#include <cstdlib>
#include <cuda_profiler_api.h>

#include "graphblas/graphblas.hpp"

#include <boost/program_options.hpp>
#include <test/test.hpp>

int main( int argc, char** argv )
{
  std::vector<graphblas::Index> row_indices;
  std::vector<graphblas::Index> col_indices;
  std::vector<float> values;
  graphblas::Index nrows, ncols, nvals;

  // Parse arguments
  namespace po = boost::program_options;
  po::variables_map vm;
  parseArgs( argc, argv, vm );
  bool DEBUG = false;

  // Read in sparse matrix
  if (argc < 2) {
    fprintf(stderr, "Usage: %s [matrix-market-filename]\n", argv[0]);
    exit(1);
  } else { 
    readMtx( argv[argc-1], row_indices, col_indices, values, nrows, ncols, 
    nvals, DEBUG );
  }

  // Matrix A
  graphblas::Matrix<float> a(nrows, ncols);
  a.build( row_indices, col_indices, values, nvals );
  a.nrows( nrows );
  a.ncols( ncols );
  a.nvals( nvals );
  if( DEBUG ) a.print();

  a.extractTuples( row_indices, col_indices, values );

  for( int i=0; i<nvals; i++ )
    std::cout << i << " " << row_indices[i] << " " << col_indices[i] << " " <<
        values[i] << std::endl;
}
