#define GRB_USE_SEQUENTIAL
//#define private public

#include <iostream>
#include <algorithm>
#include <string>

#include <cstdio>
#include <cstdlib>

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
  int TA, TB, NT, NUM_ITER;
  bool ROW_MAJOR, DEBUG, SPLIT;
  if( vm.count("ta") )
    TA       = vm["ta"].as<int>(); // default values of TA, TB, NT will be used
  if( vm.count("tb") )
    TB       = vm["tb"].as<int>();
  if( vm.count("nt") )
    NT       = vm["nt"].as<int>();
  if( vm.count("debug") )
    DEBUG    = vm["debug"].as<bool>();
  if( vm.count("split") )
    SPLIT    = vm["split"].as<bool>();
  if( vm.count("iter") )
    NUM_ITER = vm["iter"].as<int>();
  // ROW_MAJOR == 1: means row major
  // ROW_MAJOR == 0: means col major
  // TA == 0 && TB == 0 && NT == 0: means cusparse
  if( vm.count("major") ) {
    std::string major = vm["major"].as<std::string>();
    ROW_MAJOR = (major=="row");
    if( major=="cusparse" ) {
      TA = 0; TB = 0; NT = 0;
    }
  }

  if( DEBUG ) {
    std::cout << "ta:    " << TA        << "\n";
    std::cout << "tb:    " << TB        << "\n";
    std::cout << "nt:    " << NT        << "\n";
    std::cout << "row:   " << ROW_MAJOR << "\n";
    std::cout << "debug: " << DEBUG     << "\n";
    std::cout << "split: " << SPLIT     << "\n";
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
  graphblas::Matrix<float> b(nrows, ncols);
  b.build( row_indices, col_indices, values, nvals );
  b.nrows( nrows );
  b.ncols( ncols );
  b.nvals( nvals );

  graphblas::Matrix<float> c(nrows, ncols);
  graphblas::Semiring op;

  // Warmup
  graphblas::CpuTimer warmup;
  warmup.Start();
  graphblas::mxm<float, float, float>( c, op, a, b, TA, TB, NT, ROW_MAJOR );
  warmup.Stop();
 
  graphblas::CpuTimer gpu_mxm;
  //cudaProfilerStart();
  gpu_mxm.Start();
  for( int i=0; i<NUM_ITER; i++ ) {
    if( SPLIT )
      graphblas::mxmCompute<float, float, float>( c, op, a, b, TA, TB, NT, 
          ROW_MAJOR );
    else
    graphblas::mxm<float, float, float>( c, op, a, b, TA, TB, NT, ROW_MAJOR );
  }
  //cudaProfilerStop();
  gpu_mxm.Stop();

  float flop = 0;
  if( DEBUG ) std::cout << "warmup, " << warmup.ElapsedMillis() << ", " <<
    flop/warmup.ElapsedMillis()/1000000.0 << "\n";
  float elapsed_mxm = gpu_mxm.ElapsedMillis();
  std::cout << "spgemm, " << elapsed_mxm/NUM_ITER << "\n"; 

  if( DEBUG ) c.print();
  /*c.extractTuples( out_denseVal );
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
