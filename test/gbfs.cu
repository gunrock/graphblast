#define GRB_USE_APSPIE
#define private public

#include <iostream>
#include <algorithm>
#include <string>

#include <cstdio>
#include <cstdlib>

//#include <cuda_profiler_api.h>

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
    parseArgs( argc, argv, vm );
    debug     = vm["debug"    ].as<bool>();
    transpose = vm["transpose"].as<bool>();
    mtxinfo   = vm["mtxinfo"  ].as<bool>();
    directed  = vm["directed" ].as<int>();
    niter     = vm["niter"    ].as<int>();
    source    = vm["source"   ].as<int>();

    // This is an imperfect solution, because this should happen in 
    // desc.loadArgs(vm) instead of application code!
    // TODO: fix this
    readMtx( argv[argc-1], row_indices, col_indices, values, nrows, ncols, 
        nvals, directed, mtxinfo );
  }

  // Descriptor desc
  graphblas::Descriptor desc;
  CHECK( desc.loadArgs(vm) );
  if( transpose )
    CHECK( desc.toggle(GrB_INP1) );

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
  int d = graphblas::algorithm::bfsCpu( source, &a, h_bfs_cpu, depth, 
      transpose );
  bfs_cpu.Stop();

  // Warmup
  CpuTimer warmup;
  warmup.Start();
  graphblas::algorithm::bfs(&v, &a, source, &desc);
  warmup.Stop();

  std::vector<float> h_bfs_gpu;
  CHECK( v.extractTuples(&h_bfs_gpu, &nrows) );
  BOOST_ASSERT_LIST( h_bfs_cpu, h_bfs_gpu, nrows );

  // Benchmark
  graphblas::Vector<float> y(nrows);
  CpuTimer vxm_gpu;
  //cudaProfilerStart();
  vxm_gpu.Start();
  float tight = 0.f;
  float val;
  if( !desc.descriptor_.reduce() )
    for( int i=0; i<niter; i++ )
    {
      val = graphblas::algorithm::bfs2(&y, &a, source, &desc, d);
      tight += val;
    }
  else
    for( int i=0; i<niter; i++ )
    {
      val = graphblas::algorithm::bfs(&y, &a, source, &desc);
      tight += val;
    }
  //cudaProfilerStop();
  vxm_gpu.Stop();

  float flop = 0;
  std::cout << "cpu, " << bfs_cpu.ElapsedMillis() << ", \n";
  std::cout << "warmup, " << warmup.ElapsedMillis() << ", " <<
    flop/warmup.ElapsedMillis()/1000000.0 << "\n";
  float elapsed_vxm = vxm_gpu.ElapsedMillis();
  std::cout << "tight, " << tight/niter << "\n";
  std::cout << "vxm, " << elapsed_vxm/niter << "\n";

  if( niter )
  {
    std::vector<float> h_bfs_gpu2;
    CHECK( y.extractTuples(&h_bfs_gpu2, &nrows) );
    BOOST_ASSERT_LIST( h_bfs_cpu, h_bfs_gpu2, nrows );
  }

  return 0;
}
