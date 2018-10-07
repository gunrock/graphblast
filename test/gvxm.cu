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
  bool mtxinfo;
  int  directed;
  po::variables_map vm;

  // Read in sparse matrix
  if (argc < 2) {
    fprintf(stderr, "Usage: %s [matrix-market-filename]\n", argv[0]);
    exit(1);
  } else { 
    parseArgs(argc, argv, vm);
    debug    = vm["debug"   ].as<bool>();
    mtxinfo  = vm["mtxinfo" ].as<bool>();
    directed = vm["directed"].as<int>();
    readMtx( argv[argc-1], row_indices, col_indices, values, nrows, ncols, 
        nvals, directed, mtxinfo );
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
  CHECK( x.fill(1.0f) );
  CHECK( x.size(&nrows) );
  if( debug ) CHECK( x.print() );

  // Vector y
  graphblas::Vector<float> y(nrows);

  // Descriptor
  graphblas::Descriptor desc;
  CHECK( desc.loadArgs(vm) );
  //CHECK( desc.set(graphblas::GrB_MXVMODE, graphblas::GrB_PUSHONLY) );

  // Warmup
  CpuTimer warmup;
  warmup.Start();
  graphblas::vxm<float, float, float, float>(&y, GrB_NULL, GrB_NULL, 
      graphblas::PlusMultipliesSemiring<float>(), &x, &a, &desc);
  warmup.Stop();
 
  CpuTimer cpu_vxm;
  //cudaProfilerStart();
  cpu_vxm.Start();
  int NUM_ITER = 1;//0;
  for( int i=0; i<NUM_ITER; i++ )
  {
    graphblas::vxm<float, float, float, float>( &y, GrB_NULL, GrB_NULL, 
        graphblas::PlusMultipliesSemiring<float>(), &x, &a, &desc );
  }
  //cudaProfilerStop();
  cpu_vxm.Stop();

  float flop = 0;
  if( debug ) std::cout << "warmup, " << warmup.ElapsedMillis() << ", " <<
    flop/warmup.ElapsedMillis()/1000000.0 << "\n";
  float elapsed_vxm = cpu_vxm.ElapsedMillis();
  std::cout << "vxm, " << elapsed_vxm/NUM_ITER << "\n";

  if( debug ) y.print();
  return 0;
}
