#define GRB_USE_APSPIE
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
  std::vector<graphblas::Index> row_indices;
  std::vector<graphblas::Index> col_indices;
  std::vector<float> values;
  graphblas::Index nrows, ncols, nvals;

  // Parse arguments
  bool debug;
  bool transpose;
  int  directed;
  int  niter;
  po::variables_map vm;

  // Read in sparse matrix
  if (argc < 2) {
    fprintf(stderr, "Usage: %s [matrix-market-filename]\n", argv[0]);
    exit(1);
  } else { 
    parseArgs( argc, argv, vm );
    debug     = vm["debug"    ].as<bool>();
    transpose = vm["transpose"].as<bool>();
    directed  = vm["directed" ].as<int>();
    niter     = vm["niter"    ].as<int>();
    readMtx( argv[argc-1], row_indices, col_indices, values, nrows, ncols, 
        nvals, directed, debug );
  }

  // Matrix A
  graphblas::Matrix<float> a(nrows, ncols);
  CHECK( a.build(&row_indices, &col_indices, &values, nvals, GrB_NULL) );
  CHECK( a.nrows(&nrows) );
  CHECK( a.ncols(&ncols) );
  CHECK( a.nvals(&nvals) );
  if( debug ) CHECK( a.print() );

  // Vector x
  graphblas::Vector<float> x(nrows);
  std::vector<graphblas::Index> x_ind = {1,   2,   3};
  std::vector<float>            x_val = {1.f, 1.f, 1.f};
  CHECK( x.build(&x_ind, &x_val, 3, GrB_NULL) );
  CHECK( x.size(&nrows) );
  if( debug ) CHECK( x.print() );

  // Vector y
  graphblas::Vector<float> y(nrows);

  // Vector mask
  graphblas::Vector<float> m(nrows);
  CHECK( m.fill(-1.f) );
  CHECK( m.setElement(0.f, 1) );
  CHECK( m.size(&nrows) );

  // Descriptor
  graphblas::Descriptor desc;
  CHECK( desc.loadArgs(vm) );
  CHECK( desc.set(graphblas::GrB_MASK, graphblas::GrB_SCMP) );
  CHECK( desc.set(graphblas::GrB_MXVMODE, graphblas::GrB_PUSHONLY) );

  // Semiring
  graphblas::BinaryOp<float,float,float> GrB_PLUS_FP32;
  GrB_PLUS_FP32.nnew( graphblas::plus<float>() );
  graphblas::BinaryOp<float,float,float> GrB_TIMES_FP32;
  GrB_TIMES_FP32.nnew( graphblas::multiplies<float>() );
  /*graphblas::BinaryOp<float,float,float> GrB_PLUS_FP32;
  GrB_PLUS_FP32.nnew( std::plus<float>() );
  graphblas::BinaryOp<float,float,float> GrB_TIMES_FP32( 
      std::multiplies<float>() );*/
  float A = GrB_PLUS_FP32(3.f,2.f);
  float B = GrB_TIMES_FP32(3.f,2.f);
  //std::cout << A << std::endl;
  //std::cout << B << std::endl;
  graphblas::Monoid  <float> GrB_FP32Add;
  GrB_FP32Add.nnew( GrB_PLUS_FP32, 0.f );
  graphblas::Semiring<float,float,float> GrB_FP32AddMul;
  GrB_FP32AddMul.nnew( GrB_FP32Add, GrB_TIMES_FP32 );

  /*graphblas::BinaryOp GrB_LOR(  graphblas::logical_or() );
  graphblas::BinaryOp GrB_LAND( graphblas::logical_and() );
  graphblas::Monoid   GrB_Lor( GrB_LOR, false );
  graphblas::Semiring GrB_Boolean( GrB_Lor, GrB_LAND );*/

  // Warmup
  CpuTimer warmup;
  warmup.Start();
  //graphblas::vxm<float, float, float>( &y, &x, &GrB_PLUS_FP32, &GrB_FP32AddMul, 
  //    &x, &a, &desc );
  graphblas::vxm<float, float, float>(&y, &m, GrB_NULL, &GrB_FP32AddMul, 
      &x, &a, &desc);
  warmup.Stop();
 
  CpuTimer cpu_vxm;
  //cudaProfilerStart();
  cpu_vxm.Start();
  for( int i=0; i<niter; i++ )
  {
    graphblas::vxm<float, float, float>( &y, &m, GrB_NULL, 
        &GrB_FP32AddMul, &x, &a, &desc );
  }
  //cudaProfilerStop();
  cpu_vxm.Stop();

  float flop = 0;
  if( debug ) std::cout << "warmup, " << warmup.ElapsedMillis() << ", " <<
    flop/warmup.ElapsedMillis()/1000000.0 << "\n";
  float elapsed_vxm = cpu_vxm.ElapsedMillis();
  std::cout << "vxm, " << elapsed_vxm/niter << "\n";

  if( debug ) y.print();
  /*c.extractTuples( out_denseVal );
  for( int i=0; i<nvals; i++ )
  {
    graphblas::Index row = row_indices[i];
    graphblas::Index col = col_indices[i];
    float            val = values[i];
    if( col<max_ncols )
    {
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
