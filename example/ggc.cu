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
#include "graphblas/algorithm/gc.hpp"
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
  int  seed;
  int  max_colors;
  int  gc_algo;
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
    seed       = vm["seed"   ].as<int>();
    max_colors = vm["maxcolors"].as<int>();
    gc_algo    = vm["gcalgo"   ].as<int>();

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
  CpuTimer gc_cpu;
  std::vector<int> h_gc_cpu(nrows, 0);
  int depth = 10000;
  gc_cpu.Start();
  int d = graphblas::algorithm::gcCpu(seed, &a, &h_gc_cpu, max_colors);
  gc_cpu.Stop();
  graphblas::algorithm::verifyGc(&a, h_gc_cpu);

  // Warmup
  CpuTimer warmup;
  warmup.Start();
  if (gc_algo == 0)
    graphblas::algorithm::gcJP(&v, &a, seed, max_colors, &desc);
  else if (gc_algo == 1)
    graphblas::algorithm::gcMIS(&v, &a, seed, max_colors, &desc);
  else if (gc_algo == 2)
    graphblas::algorithm::gcIS(&v, &a, seed, max_colors, &desc);
  else
    std::cout << "Error: Invalid graph coloring algorithm selected!\n";
  warmup.Stop();

  std::vector<int> h_gc_gpu;
  CHECK(v.extractTuples(&h_gc_gpu, &nrows));
  graphblas::algorithm::verifyGc(&a, h_gc_gpu);

  // Benchmark
  graphblas::Vector<int> y(nrows);
  CpuTimer vxm_gpu;
  // cudaProfilerStart();
  vxm_gpu.Start();
  float tight = 0.f;
  float val;
  for (int i = 0; i < niter; i++) {
    if (gc_algo == 0) {
      val = graphblas::algorithm::gcJP(&v, &a, seed, max_colors, &desc);
    } else if (gc_algo == 1) {
      val = graphblas::algorithm::gcMIS(&v, &a, seed, max_colors, &desc);
    } else if (gc_algo == 2) {
      val = graphblas::algorithm::gcIS(&v, &a, seed, max_colors, &desc);
    } else {
      std::cout << "Error: Invalid graph coloring algorithm selected!\n";
      break;
    }
    tight += val;
  }
  // cudaProfilerStop();
  vxm_gpu.Stop();

  float flop = 0;
  std::cout << "cpu, " << gc_cpu.ElapsedMillis() << ", \n";
  std::cout << "warmup, " << warmup.ElapsedMillis() << ", " <<
    flop/warmup.ElapsedMillis()/1000000.0 << "\n";
  float elapsed_vxm = vxm_gpu.ElapsedMillis();
  std::cout << "tight, " << tight/niter << "\n";
  std::cout << "vxm, " << elapsed_vxm/niter << "\n";

  if (niter) {
    std::vector<int> h_gc_gpu2;
    graphblas::algorithm::verifyGc(&a, h_gc_gpu);
  }

  return 0;
}
