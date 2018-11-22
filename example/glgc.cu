#define GRB_USE_CUDA
#define private public

#include <iostream>
#include <algorithm>
#include <string>

#include <cstdio>
#include <cstdlib>

//#include <cuda_profiler_api.h>

#include <boost/program_options.hpp>

#include "graphblas/graphblas.hpp"
#include "graphblas/algorithm/lgc.hpp"
#include "test/test.hpp"

bool debug_;
bool memory_;

int main( int argc, char** argv )
{
  std::vector<graphblas::Index> row_indices;
  std::vector<graphblas::Index> col_indices;
  std::vector<float> values;
  graphblas::Index nrows, ncols, nvals;

  // Parse arguments
  bool debug;
  bool transpose;
  bool mtxinfo;
  int  directed;
  int  niter;
  int  max_niter;
  int  source;
  char* dat_name;
  po::variables_map vm;

  // Read in sparse matrix
  if (argc < 2) {
    fprintf(stderr, "Usage: %s [matrix-market-filename]\n", argv[0]);
    exit(1);
  } else { 
    parseArgs( argc, argv, vm );
    debug     = vm["debug"    ].as<bool>();
    transpose = vm["transpose"].as<bool>();
    mtxinfo   = vm["mtxinfo"  ].as<bool>();
    directed  = vm["directed" ].as<int>();
    niter     = vm["niter"    ].as<int>();
    max_niter = vm["max_niter"].as<int>();
    source    = vm["source"   ].as<int>();

    // This is an imperfect solution, because this should happen in 
    // desc.loadArgs(vm) instead of application code!
    // TODO: fix this
    readMtx( argv[argc-1], row_indices, col_indices, values, nrows, ncols, 
        nvals, directed, mtxinfo, &dat_name );
  }

  // Descriptor desc
  graphblas::Descriptor desc;
  CHECK( desc.loadArgs(vm) );
  if( transpose )
    CHECK( desc.toggle(graphblas::GrB_INP1) );

  // Matrix A
  graphblas::Matrix<float> a(nrows, ncols);
  values.clear();
  values.insert(values.begin(), nvals, 1.f);
  CHECK( a.build(&row_indices, &col_indices, &values, nvals, GrB_NULL, 
      dat_name) );
  CHECK( a.nrows(&nrows) );
  CHECK( a.ncols(&ncols) );
  CHECK( a.nvals(&nvals) );
  if( debug ) CHECK( a.print() );

  // Vector v
  graphblas::Vector<float> v(nrows);
  float phi   = 0.5;
  //float alpha = 0.01;
  float alpha = pow(phi, 2) / (225.0 * log(100.0 * sqrt(nvals)));
  float eps   = 0.0000001;

  // Cpu LGC
  CpuTimer lgc_cpu;
  float* h_lgc_cpu = (float*)malloc(nrows*sizeof(float));
  lgc_cpu.Start();
  graphblas::algorithm::lgcCpu( h_lgc_cpu, &a, source, alpha, eps, max_niter,
      transpose );
  lgc_cpu.Stop();

  // Warmup
  CpuTimer warmup;
  warmup.Start();
  graphblas::algorithm::lgc(&v, &a, source, alpha, eps, &desc);
  warmup.Stop();

  std::vector<float> h_lgc_gpu;
  CHECK( v.extractTuples(&h_lgc_gpu, &nrows) );
  BOOST_ASSERT_LIST_FLOAT( h_lgc_cpu, h_lgc_gpu, nrows );

  // Benchmark
  graphblas::Vector<float> y(nrows);
  CpuTimer vxm_gpu;
  //cudaProfilerStart();
  vxm_gpu.Start();
  float tight = 0.f;
  float val;
  for (int i = 0; i < niter; i++)
  {
    val = graphblas::algorithm::lgc(&y, &a, source, alpha, eps, &desc);
    tight += val;
  }
  //cudaProfilerStop();
  vxm_gpu.Stop();

  float flop = 0;
  std::cout << "cpu, " << lgc_cpu.ElapsedMillis() << ", \n";
  std::cout << "warmup, " << warmup.ElapsedMillis() << ", " <<
    flop/warmup.ElapsedMillis()/1000000.0 << "\n";
  float elapsed_vxm = vxm_gpu.ElapsedMillis();
  std::cout << "tight, " << tight/niter << "\n";
  std::cout << "vxm, " << elapsed_vxm/niter << "\n";

  if( niter )
  {
    std::vector<float> h_lgc_gpu2;
    CHECK( y.extractTuples(&h_lgc_gpu2, &nrows) );
    //BOOST_ASSERT_LIST_FLOAT( h_lgc_cpu, h_lgc_gpu2, nrows );
  }

  return 0;
}
