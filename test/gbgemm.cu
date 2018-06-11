#define GRB_USE_APSPIE
#define private public

#include <iostream>
#include <algorithm>
#include <string>

#include <cstdio>
#include <cstdlib>
#include <cuda_profiler_api.h>
#include <cublas_v2.h>

#include "graphblas/graphblas.hpp"
#include "../ext/moderngpu/include/constants.h"

#include <boost/program_options.hpp>
#include <test/test.hpp>

template <typename T>
void runTest( const std::string& str, graphblas::Matrix<T>& c, graphblas::Matrix<T>& a, graphblas::Matrix<T>& b, graphblas::Semiring& op, graphblas::Descriptor& desc, graphblas::Index max_ncols, graphblas::Index nrows, graphblas::Index ncols, int NUM_ITER, bool DEBUG, bool ROW_MAJOR )
{
  // Warmup
  graphblas::GpuTimer warmup;
  warmup.Start();
  graphblas::mxm<float, float, float>( c, graphblas::GrB_NULL, graphblas::GrB_NULL, op, a, b, desc );
  warmup.Stop();
 
  // Benchmark
  graphblas::GpuTimer gpu_mxm;
  gpu_mxm.Start();
  for( int i=0; i<NUM_ITER; i++ )
    graphblas::mxm<float, float, float>( c, graphblas::GrB_NULL, graphblas::GrB_NULL, op, a, b, desc );
  CUDA( cudaDeviceSynchronize() );
  gpu_mxm.Stop();
 
  float flop = 2.0*nrows*ncols*max_ncols;
  float byte = nrows*ncols*(sizeof(T))+max_ncols*sizeof(T)*2;
  if( DEBUG )
  {
    std::cout << "warmup, " << warmup.ElapsedMillis() << ", " <<
        flop/warmup.ElapsedMillis()/1000000.0 << ", " << byte/warmup.ElapsedMillis()/ 1000000.0 << "\n";
    std::cout << "spmm, " << gpu_mxm.ElapsedMillis()/NUM_ITER << ", " <<
        flop/gpu_mxm.ElapsedMillis()*NUM_ITER/1000000.0 << ", " << byte/gpu_mxm.ElapsedMillis()*NUM_ITER/1000000.0 << "\n";
  }
  else
  {
    std::cout << str << ", " << gpu_mxm.ElapsedMillis()/NUM_ITER << ", " <<
        flop/gpu_mxm.ElapsedMillis()*NUM_ITER/1000000.0 << ", " << byte/gpu_mxm.ElapsedMillis()*NUM_ITER/1000000.0 << ", ";
  }

  std::vector<float> out_denseVal;
  if( DEBUG ) 
  {
    c.print();
    /*c.extractTuples( out_denseVal );
    for( int i=0; i<nvals; i++ ) {
      float val = values[i];
      if( col<max_ncols ) {
        // Row major order
        if( ROW_MAJOR )
        {
          if( val!=out_denseVal[row*max_ncols+col] )
          {
            std::cout << "FAIL: " << row << " " << col << " " << val << " " << out_denseVal[row*max_ncols+col] << std::endl;
            break;
            //BOOST_ASSERT( val==out_denseVal[row*max_ncols+col] );
          }
        }
        // Column major order
        else
        {
          if( val!=out_denseVal[col*nrows+row] )
          {
            std::cout << "FAIL: " << row << " " << col << " " << val << " " << out_denseVal[col*nrows+row] << std::endl;
            break;
          //BOOST_ASSERT( val==out_denseVal[col*nrows+row] );
          }
        }
      }
    }*/
  }
}

int main( int argc, char** argv )
{
  std::vector<float> A_values;
  graphblas::Index A_nrows, A_ncols, A_nvals;

  // Parse arguments
  namespace po = boost::program_options;
  po::variables_map vm;
  parseArgs( argc, argv, vm );
  int MAX_NCOLS, NUM_ITER;
  bool ROW_MAJOR, DEBUG;
  std::string mode;
  if( vm.count("nrows") )
    A_nrows  = vm["nrows"].as<int>();
  if( vm.count("tb") )
    A_ncols  = vm["ncols"].as<int>();
  if( vm.count("max_ncols") )
    MAX_NCOLS= vm["max_ncols"].as<int>();

  // default values of TA, TB, NT will be used
  graphblas::Descriptor desc;

  if( vm.count("debug") )
    DEBUG    = vm["debug"].as<bool>();
  if( vm.count("iter") )
    NUM_ITER = vm["iter"].as<int>();
  if( vm.count("mode") ) {
    mode = vm["mode"].as<std::string>();
  }

  if( DEBUG ) {
    std::cout << "iter:  " << NUM_ITER << "\n";
    std::cout << "debug: " << DEBUG    << "\n";
  }

  // Generate dense graph
  for( int i=0; i<A_nrows; i++ )
  {
    for( int j=0; j<A_ncols; j++ )
    {
      A_values.push_back(1.f);
    }
  }

  // Matrix A
  graphblas::Matrix<float> a(A_nrows, A_ncols);
  a.build( A_values );
  a.nrows( A_nrows );
  a.ncols( A_ncols );
  a.nvals( A_nvals );
  if( DEBUG ) a.print();
  else
  {
    std::cout << argv[argc-1] << ", " << A_nrows << ", " << A_ncols << ", " << A_nvals << ", ";
    a.printStats();
  }

  // Matrix B
  graphblas::Index MEM_SIZE = 1000000000;  // 2x4=8GB GPU memory for dense
  graphblas::Index max_ncols = MAX_NCOLS;
  //if( ncols%32!=0 && max_ncols%32!=0 ) max_ncols = (ncols+31)/32*32;
  if( DEBUG && max_ncols!=A_ncols ) std::cout << "Restricting col to: " 
      << max_ncols << std::endl;

  graphblas::Matrix<float> b_row(A_ncols, max_ncols);
  graphblas::Matrix<float> b_col(A_ncols, max_ncols);
  std::vector<float> dense_row;
  std::vector<float> dense_col;

  // Row major order
  for( int i=0; i<A_ncols; i++ )
    for( int j=0; j<max_ncols; j++ ) {
      if( i==j ) dense_row.push_back(1.0);
      else dense_row.push_back(0.0);
    }
  // Column major order
  for( int i=0; i<max_ncols; i++ )
    for( int j=0; j<A_ncols; j++ ) {
      if( i==j ) dense_col.push_back(1.0);
      else dense_col.push_back(0.0);
    }
  b_row.build( dense_row );
  b_col.build( dense_col );
  graphblas::Matrix<float> c(A_nrows, max_ncols);
  graphblas::Semiring op;

  // Test cublas
  desc.set( graphblas::GrB_MODE, graphblas::GrB_CUSPARSE );
  ROW_MAJOR = true;
  runTest( "cublas", c, a, b_row, op, desc, max_ncols, A_nrows, A_ncols, NUM_ITER, DEBUG, ROW_MAJOR );

  if( !DEBUG ) std::cout << "\n";

  return 0;
}
