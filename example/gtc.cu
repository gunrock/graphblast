#define GRB_USE_CUDA
#define private public

#include <iostream>
#include <algorithm>
#include <string>

#include <cstdio>
#include <cstdlib>

// #include <cuda_tcofiler_api.h>

#include <boost/program_options.hpp>

#include "graphblas/graphblas.hpp"
#include "graphblas/algorithm/tc.hpp"
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

  values.clear();
  values.insert(values.begin(), nvals, 1.f);

  // Matrix A
  graphblas::Matrix<int> a(nrows, ncols);
  CHECK(a.build(&row_indices, &col_indices, &values, nvals, GrB_NULL,
      dat_name));
  CHECK(a.nrows(&nrows));
  CHECK(a.ncols(&ncols));
  CHECK(a.nvals(&nvals));

  // Get lower triangular of matrix A
  CHECK(desc.set(GrB_BACKEND, GrB_SEQUENTIAL));
  graphblas::tril<int, int>(&a, &a, &desc);
  CHECK(desc.set(GrB_BACKEND, GrB_CUDA));

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
  VERIFY(ntris_cpu, ntris_gpu);

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
    VERIFY(ntris_cpu, ntris_gpu);
  }

  return 0;
}
