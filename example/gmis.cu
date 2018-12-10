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
#include "graphblas/algorithm/mis.hpp"
#include "test/test.hpp"

bool debug_;
bool memory_;

int main(int argc, char** argv) {
  std::vector<graphblas::Index> row_indices;
  std::vector<graphblas::Index> col_indices;
  std::vector<int> values;
  graphblas::Index nrows, ncols, nvals;

  // Parse arguments
  bool debug;
  bool transpose;
  bool mtxinfo;
  int  directed;
  int  niter;
  int  source;
  char* dat_name;
  po::variables_map vm;

  // Read in sparse matrix
  if (argc < 2) {
    fprintf(stderr, "Usage: %s [matrix-market-filename]\n", argv[0]);
    exit(1);
  } else {
    parseArgs(argc, argv, &vm);
    debug      = vm["debug"    ].as<bool>();
    transpose  = vm["transpose"].as<bool>();
    mtxinfo    = vm["mtxinfo"  ].as<bool>();
    directed   = vm["directed" ].as<int>();
    niter      = vm["niter"    ].as<int>();
    source     = vm["source"   ].as<int>();

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
  graphblas::Matrix<int> a(nrows, ncols);
  values.clear();
  values.resize(nvals, 1.f);
  CHECK(a.build(&row_indices, &col_indices, &values, nvals, GrB_NULL,
      dat_name));
  CHECK(a.nrows(&nrows));
  CHECK(a.ncols(&ncols));
  CHECK(a.nvals(&nvals));
  if (debug) CHECK(a.print());

  // Vector v
  graphblas::Vector<int> v(nrows);

  // Cpu graph coloring
  CpuTimer mis_cpu;
  std::vector<int> h_mis_cpu(nrows, 0);
  int depth = 10000;
  mis_cpu.Start();
  int d = graphblas::algorithm::misCpu(source, &a, h_mis_cpu);
  mis_cpu.Stop();
  graphblas::algorithm::verifyMis(&a, h_mis_cpu);

  // Warmup
  CpuTimer warmup;
  warmup.Start();
  graphblas::algorithm::mis(&v, &a, source, &desc);
  warmup.Stop();

  std::vector<int> h_mis_gpu;
  CHECK(v.extractTuples(&h_mis_gpu, &nrows));
  graphblas::algorithm::verifyMis(&a, h_mis_gpu);

  // Benchmark
  graphblas::Vector<int> y(nrows);
  CpuTimer vxm_gpu;
  // cudaProfilerStart();
  vxm_gpu.Start();
  float tight = 0.f;
  float val;
  for (int i = 0; i < niter; i++) {
    val = graphblas::algorithm::mis(&v, &a, source, &desc);
    tight += val;
  }
  // cudaProfilerStop();
  vxm_gpu.Stop();

  float flop = 0;
  std::cout << "cpu, " << mis_cpu.ElapsedMillis() << ", \n";
  std::cout << "warmup, " << warmup.ElapsedMillis() << ", " <<
    flop/warmup.ElapsedMillis()/1000000.0 << "\n";
  float elapsed_vxm = vxm_gpu.ElapsedMillis();
  std::cout << "tight, " << tight/niter << "\n";
  std::cout << "vxm, " << elapsed_vxm/niter << "\n";

  if (niter) {
    std::vector<int> h_mis_gpu2;
    graphblas::algorithm::verifyMis(&a, h_mis_gpu);
  }

  return 0;
}
