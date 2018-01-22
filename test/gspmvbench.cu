#define GRB_USE_APSPIE
#define private public

#include <iostream>
#include <algorithm>
#include <string>

#include <cstdio>
#include <cstdlib>

#include <boost/program_options.hpp>

#include "graphblas/graphblas.hpp"
#include "graphblas/backend/apspie/util.hpp"  // GpuTimer
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
  std::vector<graphblas::Index> x_ind = {0};
  std::vector<float>            x_val = {1.f};
  CHECK( x.build(&x_ind, &x_val, 1, GrB_NULL) );
  CHECK( x.size(&nrows) );
  if( debug ) CHECK( x.print() );

  // Vector y
  graphblas::Vector<float> y(nrows);

  // Vector mask
  graphblas::Vector<float> m(nrows);
  CHECK( m.fill(1.f) );
  CHECK( m.setElement(-1.f, 0) );
  CHECK( m.size(&nrows) );

  // Descriptor
  graphblas::Descriptor desc;
  CHECK( desc.loadArgs(vm) );
  CHECK( desc.set(graphblas::GrB_MASK, graphblas::GrB_SCMP) );
  CHECK( desc.set(graphblas::GrB_MXVMODE, graphblas::GrB_PULLONLY) );

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
  graphblas::vxm<float, float, float>(&y, GrB_NULL, GrB_NULL, &GrB_FP32AddMul, 
      &m, &a, &desc);
  warmup.Stop();

  std::vector<float> value(nrows,-1.f);

  std::vector<float> my_time;
  graphblas::backend::GpuTimer cpu_vxm;
  //cudaProfilerStart();
  cpu_vxm.Start();
  graphblas::vxm<float, float, float>(&y, GrB_NULL, GrB_NULL, &GrB_FP32AddMul, 
      &m, &a, &desc);
  cpu_vxm.Stop();
  my_time.push_back(cpu_vxm.ElapsedMillis());  

  for( int i=1000; i<nrows; i+=1000 )
  {
    m.clear();
    m.build(&value, i);
    cpu_vxm.Start();
    graphblas::vxm<float, float, float>( &y, GrB_NULL, GrB_NULL, 
        &GrB_FP32AddMul, &m, &a, &desc );
    cpu_vxm.Stop();
    my_time.push_back(cpu_vxm.ElapsedMillis());
  }
  //cudaProfilerStop();

  float flop = 0;
  std::cout << "warmup, " << warmup.ElapsedMillis() << std::endl;

  for( int i=0; i<my_time.size(); i++ )
    std::cout << (i)*1000 << ", " << my_time[i] << std::endl;

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
