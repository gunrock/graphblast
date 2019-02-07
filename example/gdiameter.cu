#define GRB_USE_CUDA
#define private public

#include <iostream>
#include <algorithm>
#include <string>

#include <cstdio>
#include <cstdlib>

// #include <cuda_profiler_api.h>

#include <boost/program_options.hpp>

#include "graphblas/graphblas.hpp"
#include "graphblas/algorithm/diameter.hpp"
#include "test/test.hpp"

bool debug_;
bool memory_;

int main(int argc, char** argv) {
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
  int  source_start;
  int  source_end;
  char* dat_name;
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
    source_start = vm["source_start"].as<int>();
    source_end = vm["source_end"].as<int>();

    // This is an imperfect solution, because this should happen in
    // desc.loadArgs(vm) instead of application code!
    // TODO(@ctcyang): fix this
    readMtx(argv[argc-1], &row_indices, &col_indices, &values, &nrows, &ncols,
        &nvals, directed, mtxinfo, &dat_name);
  }

  // Descriptor desc
  graphblas::Descriptor desc;
  CHECK(desc.loadArgs(vm));
  if (transpose)
    CHECK(desc.toggle(graphblas::GrB_INP1));

  // Matrix A
  graphblas::Matrix<float> a(nrows, ncols);
  CHECK(a.build(&row_indices, &col_indices, &values, nvals, GrB_NULL,
      dat_name));
  CHECK(a.nrows(&nrows));
  CHECK(a.ncols(&ncols));
  CHECK(a.nvals(&nvals));
  if (debug) CHECK(a.print());

  // Vector v
  graphblas::Vector<float> v(nrows);

  // Cpu BFS
  /*CpuTimer bfs_cpu;
  graphblas::Index* h_bfs_cpu = reinterpret_cast<graphblas::Index*>(
      malloc(nrows*sizeof(graphblas::Index)));
  int depth = 10000;
  bfs_cpu.Start();
  int d = graphblas::algorithm::bfsCpu(source, &a, h_bfs_cpu, depth, transpose);
  bfs_cpu.Stop();*/

  // Warmup

  int diameter_max = 0;
  int diameter_ind = -1;
  source_start = std::min(nrows-1, std::max(0, source_start));
  source_end   = std::max(0, std::min(nrows-1, source_end));
  CpuTimer warmup;
  warmup.Start();
  std::pair<int, int> val = graphblas::algorithm::diameter(&v, &a, source_start,
      source_end, &desc);
  warmup.Stop();

  std::cout << "warmup, " << warmup.ElapsedMillis() << ", " <<
    source_end - source_start << ", " << warmup.ElapsedMillis()/(source_end - source_start) << ", \n";
  std::cout << "diameter " << source_start << ":" << source_end << ": " << val.first << " from " << val.second << std::endl;

  return 0;
}
