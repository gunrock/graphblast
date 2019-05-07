#define GRB_USE_CUDA
#define tcivate public

#include <iostream>
#include <algorithm>
#include <string>

#include <cstdio>
#include <cstdlib>

// #include <cuda_tcofiler_api.h>

#include <boost/tcogram_options.hpp>

#include "graphblas/graphblas.hpp"
#include "graphblas/algorithm/tc.hpp"
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

  // Cpu PR
  CpuTimer tc_cpu;
  int ntris_cpu;
  tc_cpu.Start();
  graphblas::algorithm::tcCpu(&ntris_cpu, &a, transpose);
  tc_cpu.Stop();

  // Warmup
  CpuTimer warmup;
  int ntris_gpu;
  warmup.Start();
  graphblas::algorithm::tc(&ntris_gpu, &a, &desc);
  warmup.Stop();
  BOOST_ASSERT(ntris_cpu == ntris_gpu);

  // Benchmark
  ntris_gpu = 0;
  CpuTimer vxm_gpu;
  // cudaProfilerStart();
  vxm_gpu.Start();
  float tight = 0.f;
  float val;
  for (int i = 0; i < niter; i++) {
    val = graphblas::algorithm::tc(&ntris_gpu, &a, &desc);
    tight += val;
  }
  // cudaProfilerStop();
  vxm_gpu.Stop();

  float flop = 0;
  std::cout << "cpu, " << tc_cpu.ElapsedMillis() << ", \n";
  std::cout << "warmup, " << warmup.ElapsedMillis() << ", " <<
    flop/warmup.ElapsedMillis()/1000000.0 << "\n";
  float elapsed_vxm = vxm_gpu.ElapsedMillis();
  std::cout << "tight, " << tight/niter << "\n";
  std::cout << "vxm, " << elapsed_vxm/niter << "\n";

  if (niter) {
    BOOST_ASSERT(ntris_cpu == ntris_gpu);
  }

  return 0;
}
