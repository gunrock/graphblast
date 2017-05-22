#define GRB_USE_APSPIE
//#define private public

#include <iostream>
#include <algorithm>
#include <string>

#include <cstdio>
#include <cstdlib>
#include <cuda_profiler_api.h>

#include "graphblas/mmio.hpp"
#include "graphblas/util.hpp"
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
  int TA, TB, NT;
  bool ROW_MAJOR, DEBUG;
  if( vm.count("ta") )
    TA = vm["ta"].as<int>();
  if( vm.count("tb") )
    TB = vm["tb"].as<int>();
  if( vm.count("nt") )
    NT = vm["nt"].as<int>();
  if( vm.count("debug") )
    DEBUG = vm["debug"].as<bool>();
  // ROW_MAJOR == 1: means row major
  // ROW_MAJOR == 0: means col major
  if( vm.count("major") )
    ROW_MAJOR = (vm["major"].as<std::string>()=="row");
  if( DEBUG ) {
    std::cout << "ta:    " << TA        << "\n";
    std::cout << "tb:    " << TB        << "\n";
    std::cout << "nt:    " << NT        << "\n";
    std::cout << "row:   " << ROW_MAJOR << "\n";
    std::cout << "debug: " << DEBUG     << "\n";
  }

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

  // Matrix B
  graphblas::Index MEM_SIZE = 1000000000;  // 2x4=8GB GPU memory for dense
  graphblas::Index max_ncols = std::min( MEM_SIZE/nrows, ncols );
  if( DEBUG && max_ncols<ncols ) std::cout << "Restricting col to: " 
      << max_ncols << std::endl;

  graphblas::Matrix<float> b(nrows, max_ncols);
  std::vector<float> denseVal;

  // Row major order
  if( ROW_MAJOR )
    for( int i=0; i<nrows; i++ )
      for( int j=0; j<max_ncols; j++ ) {
        if( i==j ) denseVal.push_back(1.0);
        else denseVal.push_back(0.0);
      }
  else
  // Column major order
    for( int i=0; i<max_ncols; i++ )
      for( int j=0; j<nrows; j++ ) {
        //denseVal.push_back(1.0);
        if( i==j ) denseVal.push_back(1.0);
        else denseVal.push_back(0.0);
      }
  b.build( denseVal );
  graphblas::Matrix<float> c(nrows, max_ncols);
  graphblas::Semiring op;

  cudaProfilerStart();
  graphblas::mxm<float, float, float>( c, op, a, b, TA, TB, NT, ROW_MAJOR );
  cudaProfilerStop();
  /*
  std::vector<float> out_denseVal;
  if( DEBUG ) c.print();
  c.extractTuples( out_denseVal );
  for( int i=0; i<nvals; i++ ) {
    graphblas::Index row = row_indices[i];
    graphblas::Index col = col_indices[i];
    float            val = values[i];
    if( col<max_ncols ) {
      // Row major order
      if( ROW_MAJOR )
      //std::cout << row << " " << col << " " << val << " " << out_denseVal[row*max_ncols+col] << std::endl;
        BOOST_ASSERT( val==out_denseVal[row*max_ncols+col] );
      else
      // Column major order
      //std::cout << row << " " << col << " " << val << " " << out_denseVal[col*nrows+row] << std::endl;
        BOOST_ASSERT( val==out_denseVal[col*nrows+row] );
    }
  }*/
  return 0;
}
