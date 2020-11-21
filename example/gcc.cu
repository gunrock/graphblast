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
#include "graphblas/algorithm/cc.hpp"
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
  int  cc_algo;
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
    seed       = vm["seed"     ].as<int>();
    cc_algo    = vm["ccalgo"   ].as<int>();

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
  graphblas::Matrix<int> A(nrows, ncols);
  values.clear();
  values.resize(nvals, 1.f);
  CHECK(A.build(&row_indices, &col_indices, &values, nvals, GrB_NULL,
      dat_name));
  CHECK(A.nrows(&nrows));
  CHECK(A.ncols(&ncols));
  CHECK(A.nvals(&nvals));
  if (debug) CHECK(A.print());

  // Vector v
  graphblas::Vector<int> v(nrows);

  // Cpu connected components.
  CpuTimer cc_cpu;
  std::vector<int> h_cc_cpu(nrows, 0);
  int depth = 10000;
  cc_cpu.Start();
  int d = graphblas::algorithm::ccCpu(seed, &A, &h_cc_cpu);
  cc_cpu.Stop();
  graphblas::algorithm::verifyCc(&A, h_cc_cpu, /*suppress_zero=*/true);

  // Warmup
  CpuTimer warmup;
  warmup.Start();
  if (cc_algo == 0) {
    graphblas::algorithm::cc(&v, &A, seed, &desc);
  } else if (cc_algo == 1) {
    std::cout << "Error: CC algorithm 1 not implemented!\n";
    //graphblas::algorithm::ccMIS(&v, &A, seed, &desc);
  } else if (cc_algo == 2) {
    std::cout << "Error: CC algorithm 2 not implemented!\n";
    //graphblas::algorithm::ccIS(&v, &A, seed, &desc);
  } else {
    std::cout << "Error: Invalid connected components algorithm selected!\n";
  }
  warmup.Stop();

  std::vector<int> h_cc_gpu;
  CHECK(v.extractTuples(&h_cc_gpu, &nrows));
  graphblas::algorithm::verifyCc(&A, h_cc_gpu, /*suppress_zero=*/true);

  // Benchmark
  graphblas::Vector<int> y(nrows);
  CpuTimer vxm_gpu;
  // cudaProfilerStart();
  vxm_gpu.Start();
  float tight = 0.f;
  float val;
  for (int i = 0; i < niter; i++) {
    if (cc_algo == 0) {
      val = graphblas::algorithm::cc(&v, &A, seed, &desc);
    } else if (cc_algo == 1) {
      std::cout << "Error: CC algorithm 1 not implemented!\n";
      //val = graphblas::algorithm::ccMIS(&v, &A, seed, &desc);
    } else if (cc_algo == 2) {
      std::cout << "Error: CC algorithm 2 not implemented!\n";
      //val = graphblas::algorithm::ccIS(&v, &A, seed, &desc);
    } else {
      std::cout << "Error: Invalid connected components algorithm selected!\n";
      break;
    }
    tight += val;
  }
  // cudaProfilerStop();
  vxm_gpu.Stop();

  float flop = 0;
  std::cout << "cpu, " << cc_cpu.ElapsedMillis() << ", \n";
  std::cout << "warmup, " << warmup.ElapsedMillis() << ", " <<
    flop/warmup.ElapsedMillis()/1000000.0 << "\n";
  float elapsed_vxm = vxm_gpu.ElapsedMillis();
  std::cout << "tight, " << tight/niter << "\n";
  std::cout << "vxm, " << elapsed_vxm/niter << "\n";

  if (niter) {
    std::vector<int> h_cc_gpu2;
    graphblas::algorithm::verifyCc(&A, h_cc_gpu, /*suppress_zero=*/true);
  }

  return 0;
}
