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

  // Warmup
  CpuTimer warmup;
  warmup.Start();
  graphblas::vxm<float, float, float, float>(&y, GrB_NULL, GrB_NULL, 
      graphblas::PlusMultipliesSemiring<float>(), &m, &a, &desc);
  warmup.Stop();

  std::vector<float> value(nrows,-1.f);

  std::vector<float> my_time;
  graphblas::backend::GpuTimer cpu_vxm;
  //cudaProfilerStart();
  cpu_vxm.Start();
  graphblas::vxm<float, float, float, float>(&y, GrB_NULL, GrB_NULL, 
      graphblas::PlusMultipliesSemiring<float>(), &m, &a, &desc);
  cpu_vxm.Stop();
  my_time.push_back(cpu_vxm.ElapsedMillis());  

  for( int i=1000; i<nrows; i+=1000 )
  {
    //m.clear();
    m.build(&value, i);
    cpu_vxm.Start();
    graphblas::vxm<float, float, float, float>( &y, GrB_NULL, GrB_NULL, 
        graphblas::PlusMultipliesSemiring<float>(), &m, &a, &desc );
    cpu_vxm.Stop();
    my_time.push_back(cpu_vxm.ElapsedMillis());
  }
  //cudaProfilerStop();

  float flop = 0;
  std::cout << "warmup, " << warmup.ElapsedMillis() << std::endl;

  for( int i=0; i<my_time.size(); i++ )
    std::cout << (i)*1000 << ", " << my_time[i] << std::endl;

  if( debug ) y.print();
  return 0;
}
