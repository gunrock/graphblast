#define GRB_USE_CUDA
#define private public

#include <iostream>
#include <algorithm>
#include <string>

#include <cstdio>
#include <cstdlib>

#include <cuda_profiler_api.h>

#include <boost/program_options.hpp>

#include "graphblas/graphblas.hpp"
#include "graphblas/algorithm/bfs.hpp"
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
  int  source;
  po::variables_map vm;

  // Read in sparse matrix
  if (argc < 2) {
    fprintf(stderr, "Usage: %s [matrix-market-filename]\n", argv[0]);
    exit(1);
  } else { 
    parseArgs(argc, argv, &vm);
    debug     = vm["debug"    ].as<bool>();
    transpose = vm["transpose"].as<bool>();
    mtxinfo   = vm["mtxinfo"  ].as<bool>();
    directed  = vm["directed" ].as<int>();
    niter     = vm["niter"    ].as<int>();
    source    = vm["source"   ].as<int>();

    // This is an imperfect solution, because this should happen in 
    // desc.loadArgs(vm) instead of application code!
    // TODO: fix this
    readMtx(argv[argc-1], &row_indices, &col_indices, &values, &nrows, &ncols,
        &nvals, directed, mtxinfo );
  }

  // Descriptor desc
  graphblas::Descriptor desc;
  CHECK( desc.loadArgs(vm) );
  CHECK( desc.toggle(graphblas::GrB_INP1) );

  // Matrix A
  graphblas::Matrix<float> a(nrows, ncols);
  CHECK( a.build(&row_indices, &col_indices, &values, nvals, GrB_NULL, 
      argv[argc-1]) );
  CHECK( a.nrows(&nrows) );
  CHECK( a.ncols(&ncols) );
  CHECK( a.nvals(&nvals) );
  if( debug ) CHECK( a.print() );

  // Vector v
  graphblas::Vector<float> v(nrows);

  // Cpu BFS
  CpuTimer bfs_cpu;
  graphblas::Index* h_bfs_cpu = (graphblas::Index*)malloc(nrows*
      sizeof(graphblas::Index));
  int depth = 10000;
  bfs_cpu.Start();
  graphblas::algorithm::bfsCpu( source, &a, h_bfs_cpu, depth, transpose );
  bfs_cpu.Stop();

  // Warmup
  CpuTimer warmup;
  warmup.Start();
  graphblas::algorithm::bfs(&v, &a, source, &desc);
  warmup.Stop();

  std::vector<float> h_bfs_gpu;
  CHECK( v.extractTuples(&h_bfs_gpu, &nrows) );
  BOOST_ASSERT_LIST( h_bfs_cpu, h_bfs_gpu, nrows );

  // Source randomization
  std::mt19937 gen(0);
  std::uniform_int_distribution<> dis(0,nrows-1);

  // Benchmark
  CpuTimer vxm_gpu;
  //cudaProfilerStart();
  vxm_gpu.Start();
  float tight = 0.f;
  for( int i=0; i<niter; i++ )
  {
    source = dis(gen);
    tight += graphblas::algorithm::bfs(&v, &a, source, &desc);
  }
  //cudaProfilerStop();
  vxm_gpu.Stop();

  return 0;
}
