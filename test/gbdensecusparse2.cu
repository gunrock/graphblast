#define GRB_USE_APSPIE
#define private public

#include <iostream>
#include <algorithm>
#include <string>

#include <cstdio>
#include <cstdlib>
#include <cuda_profiler_api.h>

#include "graphblas/graphblas.hpp"
#include "../ext/moderngpu/include/constants.h"

#include <boost/program_options.hpp>
#include <test/test.hpp>

template <typename T>
void runTest( const std::string& str, graphblas::Matrix<T>& c, graphblas::Matrix<T>& a, graphblas::Matrix<T>& b, graphblas::Semiring& op, graphblas::Descriptor& desc, graphblas::Index max_ncols, graphblas::Index nrows, graphblas::Index nvals, int NUM_ITER, bool DEBUG, bool ROW_MAJOR, std::vector<graphblas::Index>& row_indices, std::vector<graphblas::Index>& col_indices, std::vector<float>& values )
{
  if( str=="merge path" )
  {
    graphblas::Index a_nvals;
    a.nvals( a_nvals );
    int num_blocks = (a_nvals+MGPU_NV-1)/MGPU_NV;
    int num_segreduce = (num_blocks + MGPU_NT - 1)/MGPU_NT;
    CUDA( cudaMalloc( &desc.descriptor_.d_limits_,  
        (num_blocks+1)*sizeof(graphblas::Index) ));
    CUDA( cudaMalloc( &desc.descriptor_.d_carryin_, 
        num_blocks*MGPU_BC*sizeof(T) ));
    CUDA( cudaMalloc( &desc.descriptor_.d_carryout_,
        num_segreduce*sizeof(T)      ));
  }

  // Warmup
  graphblas::GpuTimer warmup;
  if( str=="row split" )
  {
    warmup.Start();
    graphblas::mxm<float, float, float>( c, graphblas::GrB_NULL, graphblas::GrB_NULL, op, a, b, desc );
    warmup.Stop();
  }
  else
  { 
    cudaProfilerStart();
    warmup.Start();
    graphblas::mxm<float, float, float>( c, graphblas::GrB_NULL, graphblas::GrB_NULL, op, a, b, desc );
    warmup.Stop();
    cudaProfilerStop();
  }
 
  // Benchmark
  graphblas::GpuTimer gpu_mxm;
  gpu_mxm.Start();
  for( int i=0; i<NUM_ITER; i++ )
    graphblas::mxm<float, float, float>( c, graphblas::GrB_NULL, graphblas::GrB_NULL, op, a, b, desc );
  CUDA( cudaDeviceSynchronize() );
  gpu_mxm.Stop();
 
  float flop = 2.0*nvals*max_ncols;
  float byte = nvals*(sizeof(T) + sizeof(graphblas::Index)) + nrows*(sizeof(graphblas::Index)+max_ncols*sizeof(T)*2);
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
    //c.print();
    c.extractTuples( out_denseVal );
    for( int i=0; i<nvals; i++ ) {
      graphblas::Index row = row_indices[i];
      graphblas::Index col = col_indices[i];
      float            val = values[i];
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
    }
  }
}

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
  int TA, TB, NT, NUM_ITER, MAX_NCOLS, DENSE;
  bool ROW_MAJOR, DEBUG;
  std::string mode;
  if( vm.count("ta") )
    TA       = vm["ta"].as<int>();
  if( vm.count("tb") )
    TB       = vm["tb"].as<int>();
  if( vm.count("nt") )
    NT       = vm["nt"].as<int>();
  if( vm.count("max_ncols") )
    MAX_NCOLS= vm["max_ncols"].as<int>();
  if( vm.count("dense") )
    DENSE    = vm["dense"].as<int>();

  // default values of TA, TB, NT will be used
  graphblas::Descriptor desc;
  desc.set( graphblas::GrB_MODE, graphblas::GrB_FIXEDROW );
  desc.set( graphblas::GrB_NT, NT );
  desc.set( graphblas::GrB_TA, TA );
  desc.set( graphblas::GrB_TB, TB );

  if( vm.count("debug") )
    DEBUG    = vm["debug"].as<bool>();
  if( vm.count("iter") )
    NUM_ITER = vm["iter"].as<int>();
  if( vm.count("mode") ) {
    mode = vm["mode"].as<std::string>();
  }

  if( DEBUG ) {
    std::cout << "mode:  " << mode     << "\n";
    std::cout << "ta:    " << TA       << "\n";
    std::cout << "tb:    " << TB       << "\n";
    std::cout << "nt:    " << NT       << "\n";
    std::cout << "iter:  " << NUM_ITER << "\n";
    std::cout << "debug: " << DEBUG    << "\n";
    std::cout << "dense: " << DENSE    << "\n";
  }

  // Generate dense graph
  nvals = 1<<24;  // 16M nnz
  nrows = DENSE;
  ncols = nvals/nrows;
  for( int i=0; i<nrows; i++ )
  {
    for( int j=0; j<ncols; j++ )
    {
      row_indices.push_back(i);
      col_indices.push_back(j);
      values.push_back(1.f);
    }
  }
  assert( row_indices.size()==nvals );
  assert( col_indices.size()==nvals );
  assert( values.size()==nvals );

  // Matrix A
  graphblas::Matrix<float> a(nrows, ncols);
  a.build( row_indices, col_indices, values, nvals );
  a.nrows( nrows );
  a.ncols( ncols );
  a.nvals( nvals );
  if( DEBUG ) a.print();
  else
  {
    std::cout << argv[argc-1] << ", " << nrows << ", " << ncols << ", " << nvals << ", ";
    a.printStats();
  }

  // Matrix B
  graphblas::Index MEM_SIZE = 1000000000;  // 2x4=8GB GPU memory for dense
  graphblas::Index max_ncols = MAX_NCOLS;
  //if( ncols%32!=0 && max_ncols%32!=0 ) max_ncols = (ncols+31)/32*32;
  if( DEBUG && max_ncols!=ncols ) std::cout << "Restricting col to: " 
      << max_ncols << std::endl;

  graphblas::Matrix<float> b_row(ncols, max_ncols);
  graphblas::Matrix<float> b_col(ncols, max_ncols);
  std::vector<float> dense_row;
  std::vector<float> dense_col;

  // Row major order
  for( int i=0; i<ncols; i++ )
    for( int j=0; j<max_ncols; j++ ) {
      if( i==j ) dense_row.push_back(1.0);
      else dense_row.push_back(0.0);
    }
  // Column major order
  for( int i=0; i<max_ncols; i++ )
    for( int j=0; j<ncols; j++ ) {
      if( i==j ) dense_col.push_back(1.0);
      else dense_col.push_back(0.0);
    }
  b_row.build( dense_row );
  b_col.build( dense_col );
  graphblas::Matrix<float> c(nrows, max_ncols);
  graphblas::Semiring op;

  // Test cusparse
  /*desc.set( graphblas::GrB_MODE, graphblas::GrB_CUSPARSE );
  ROW_MAJOR = false;
  runTest( "cusparse", c, a, b_col, op, desc, max_ncols, nrows, nvals, NUM_ITER, DEBUG, ROW_MAJOR, row_indices, col_indices, values );*/
  
  // Test cusparse
  desc.set( graphblas::GrB_MODE, graphblas::GrB_CUSPARSE2 );
  ROW_MAJOR = false;
  runTest( "cusparse2", c, a, b_row, op, desc, max_ncols, nrows, nvals, NUM_ITER, DEBUG, ROW_MAJOR, row_indices, col_indices, values );

  // Test row splitting
  /*desc.set( graphblas::GrB_MODE, graphblas::GrB_FIXEDROW );
  ROW_MAJOR = true;
  runTest( "row split", c, a, b_row, op, desc, max_ncols, nrows, nvals, NUM_ITER, DEBUG, ROW_MAJOR, row_indices, col_indices, values );*/

  // Test mergepath
  /*desc.set( graphblas::GrB_MODE, graphblas::GrB_MERGEPATH );
  ROW_MAJOR = true;
  runTest( "merge path", c, a, b_row, op, desc, max_ncols, nrows, nvals, NUM_ITER, DEBUG, ROW_MAJOR, row_indices, col_indices, values );*/

  if( !DEBUG ) std::cout << "\n";

  /*std::vector<float> out_denseVal;
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
