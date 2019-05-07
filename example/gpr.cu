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
#include "graphblas/algorithm/pr.hpp"
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
  int  max_niter;
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
    max_niter = vm["max_niter"].as<int>();

    /*!
     * This is an imperfect solution, because this should happen in 
     * desc.loadArgs(vm) instead of application code!
     * TODO(@ctcyang): fix this
     */
    readMtx(argv[argc-1], &row_indices, &col_indices, &values, &nrows, &ncols,
        &nvals, directed, mtxinfo, &dat_name);
  }

  // Descriptor desc
  graphblas::Descriptor desc;
  CHECK(desc.loadArgs(vm));
  if (transpose)
    CHECK(desc.toggle(graphblas::GrB_INP1));

  // PageRank Parameters
  float alpha = 0.85;
  float eps   = 1e-8;

  // Matrix A
  graphblas::Matrix<float> a(nrows, ncols);
  values.clear();
  values.insert(values.begin(), nvals, 1.f);
  CHECK(a.build(&row_indices, &col_indices, &values, nvals, GrB_NULL,
      dat_name));
  CHECK(a.nrows(&nrows));
  CHECK(a.ncols(&ncols));
  CHECK(a.nvals(&nvals));

  // Compute outdegrees
  graphblas::Vector<float> outdegrees(nrows);
  graphblas::reduce<float, float, float>(&outdegrees, GrB_NULL, GrB_NULL,
      graphblas::PlusMonoid<float>(), &a, &desc);

  // A = alpha*A/outdegrees (broadcast variant)
  graphblas::eWiseMult<float, float, float, float>(&a, GrB_NULL, GrB_NULL,
      PlusMultipliesSemiring<float>(), &a, alpha, &desc);
  if (debug) CHECK(a.print());
  graphblas::eWiseMult<float, float, float, float>(&a, GrB_NULL, GrB_NULL,
      PlusDividesSemiring<float>(), &a, &outdegrees, &desc);

  /*// Diagonalize outdegrees
  Matrix<float> diag_outdegrees(A_nrows, A_nrows);
  diag<float, float>(&diag_outdegrees, &outdegrees, desc);

  // A = alpha*A*diag(outdegrees)
  Matrix<float> A_temp(A_nrows, A_nrows);
  scale<float, float, float>(&A_temp, MultipliesMonoid<float>(), A, alpha,
      desc);*/

  if (debug) CHECK(a.print());

  // Vector v
  graphblas::Vector<float> v(nrows);

  // Cpu PR
  CpuTimer pr_cpu;
  float* h_pr_cpu = reinterpret_cast<float*>(malloc(nrows*sizeof(float)));
  pr_cpu.Start();
  graphblas::algorithm::prCpu(h_pr_cpu, &a, alpha, eps, max_niter, transpose);
  pr_cpu.Stop();

  // Warmup
  CpuTimer warmup;
  warmup.Start();
  graphblas::algorithm::pr(&v, &a, alpha, eps, &desc);
  warmup.Stop();

  std::vector<float> h_pr_gpu;
  CHECK(v.extractTuples(&h_pr_gpu, &nrows));
  VERIFY_LIST_FLOAT(h_pr_cpu, h_pr_gpu, nrows);

  // Benchmark
  graphblas::Vector<float> y(nrows);
  CpuTimer vxm_gpu;
  // cudaProfilerStart();
  vxm_gpu.Start();
  float tight = 0.f;
  float val;
  for (int i = 0; i < niter; i++) {
    val = graphblas::algorithm::pr(&y, &a, alpha, eps, &desc);
    tight += val;
  }
  // cudaProfilerStop();
  vxm_gpu.Stop();

  float flop = 0;
  std::cout << "cpu, " << pr_cpu.ElapsedMillis() << ", \n";
  std::cout << "warmup, " << warmup.ElapsedMillis() << ", " <<
    flop/warmup.ElapsedMillis()/1000000.0 << "\n";
  float elapsed_vxm = vxm_gpu.ElapsedMillis();
  std::cout << "tight, " << tight/niter << "\n";
  std::cout << "vxm, " << elapsed_vxm/niter << "\n";

  if (niter) {
    std::vector<float> h_pr_gpu2;
    CHECK(y.extractTuples(&h_pr_gpu2, &nrows));
    VERIFY_LIST_FLOAT(h_pr_cpu, h_pr_gpu2, nrows);
  }

  return 0;
}
