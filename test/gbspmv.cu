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
  int TA, TB, NT, NUM_ITER;
  bool ROW_MAJOR, DEBUG;
  std::string mode;
  if( vm.count("ta") )
    TA       = vm["ta"].as<int>();
  if( vm.count("tb") )
    TB       = vm["tb"].as<int>();
  if( vm.count("nt") )
    NT       = vm["nt"].as<int>();

  // default values of TA, TB, NT will be used
  graphblas::Descriptor desc;
  desc.set( graphblas::GrB_MODE, graphblas::GrB_FIXEDROW );
  desc.set( graphblas::GrB_NT, NT );
  desc.set( graphblas::GrB_TA, TA );
  desc.set( graphblas::GrB_TB, 1 );

  if( vm.count("debug") )
    DEBUG    = vm["debug"].as<bool>();
  if( vm.count("iter") )
    NUM_ITER = vm["iter"].as<int>();
  if( vm.count("mode") ) {
    mode = vm["mode"].as<std::string>();
  }

  // cuSPARSE (column major)
  if( mode=="cusparse" ) {
    ROW_MAJOR = false;
    desc.set( graphblas::GrB_MODE, graphblas::GrB_CUSPARSE );
  // fixed # of threads per row (row major)
  } else if( mode=="fixedrow" ) {
    ROW_MAJOR = true;
    desc.set( graphblas::GrB_MODE, graphblas::GrB_FIXEDROW );
  // fixed # of threads per column (col major)
  } else if( mode=="fixedcol" ) {
    ROW_MAJOR = false;
    desc.set( graphblas::GrB_MODE, graphblas::GrB_FIXEDCOL );
  // variable # of threads per row (row major)
  } else if( mode=="mergepath" ) {
    ROW_MAJOR = true;
    desc.set( graphblas::GrB_MODE, graphblas::GrB_MERGEPATH );
  }

  if( DEBUG ) {
    std::cout << "mode:  " << mode     << "\n";
    std::cout << "ta:    " << TA       << "\n";
    std::cout << "tb:    " << TB       << "\n";
    std::cout << "nt:    " << NT       << "\n";
    std::cout << "iter:  " << NUM_ITER << "\n";
    std::cout << "debug: " << DEBUG    << "\n";
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
  graphblas::Index max_ncols = 1;//std::min( MEM_SIZE/nrows/32*32, ncols );
  if( DEBUG && max_ncols!=ncols ) std::cout << "Restricting col to: " 
      << max_ncols << std::endl;

  graphblas::Matrix<float> b(ncols, max_ncols);
  std::vector<float> denseVal;

  // Row major order
  if( ROW_MAJOR )
    for( int i=0; i<ncols; i++ )
      for( int j=0; j<max_ncols; j++ ) {
        denseVal.push_back(1.0);
        //else denseVal.push_back(0.0);
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
  if( DEBUG ) b.print();
  graphblas::Matrix<float> c(nrows, max_ncols);
  graphblas::Semiring op;

  // Warmup
  cudaProfilerStart();
  graphblas::GpuTimer warmup;
  warmup.Start();
  graphblas::mxv<float, float, float>( c, graphblas::GrB_NULL, graphblas::GrB_NULL, op, a, b, desc );
  CUDA( cudaDeviceSynchronize() );
  warmup.Stop();
  cudaProfilerStop();
  
  // Benchmark
  graphblas::GpuTimer gpu_mxm;
  gpu_mxm.Start();
  for( int i=0; i<NUM_ITER; i++ )
    graphblas::mxv<float, float, float>( c, graphblas::GrB_NULL, graphblas::GrB_NULL, op, a, b, desc );
  CUDA( cudaDeviceSynchronize() );
  gpu_mxm.Stop();
  
  float flop = 2.0*nvals*max_ncols;
  std::cout << "warmup, " << warmup.ElapsedMillis() << ", " <<
    flop/warmup.ElapsedMillis()/1000000.0 << "\n";
  std::cout << "spmm, " << gpu_mxm.ElapsedMillis()/NUM_ITER << ", " <<
    flop/gpu_mxm.ElapsedMillis()*NUM_ITER/1000000.0 << "\n";  
  
  std::vector<float> out_denseVal;
  if( DEBUG ) c.print();
  c.extractTuples( out_denseVal );
  for( int i=0; i<nvals; i++ ) {
    graphblas::Index row = row_indices[i];
    graphblas::Index col = col_indices[i];
    float            val = values[i];
    /*if( col<max_ncols ) {
      // Row major order
      if( ROW_MAJOR )
      //std::cout << row << " " << col << " " << val << " " << out_denseVal[row*max_ncols+col] << std::endl;
        BOOST_ASSERT( val==out_denseVal[row*max_ncols+col] );
      else
      // Column major order
      //std::cout << row << " " << col << " " << val << " " << out_denseVal[col*nrows+row] << std::endl;
        BOOST_ASSERT( val==out_denseVal[col*nrows+row] );
    }*/
  }
  return 0;
}
