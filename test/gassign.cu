#define GRB_USE_CUDA
#define private public

#include <iostream>
#include <algorithm>
#include <string>

#include <cstdio>
#include <cstdlib>

#include <boost/program_options.hpp>

#include "graphblas/graphblas.hpp"
#include "test/test.hpp"

int main( int argc, char** argv )
{
  std::vector<graphblas::Index> row_indices;
  std::vector<graphblas::Index> col_indices;
  std::vector<float> values;
  graphblas::Index nrows, ncols, nvals;

  // Parse arguments
  bool DEBUG = true;

  // Read in sparse matrix
  if (argc < 2) {
    fprintf(stderr, "Usage: %s [matrix-market-filename]\n", argv[0]);
    exit(1);
  } else { 
    readMtx(argv[argc-1], &row_indices, &col_indices, &values, &nrows, &ncols, 
        &nvals, 0, DEBUG);
  }

  // Vector mask
  graphblas::Vector<float> m(nrows);
  std::vector<graphblas::Index> m_ind = {1,   2,   3};
  std::vector<float>            m_val = {1.f, 1.f, 1.f};
  CHECK( m.build(&m_ind, &m_val, 3, GrB_NULL) );
  CHECK( m.size(&nrows) );
  if( DEBUG ) CHECK( m.print() );

  // Vector v
  graphblas::Vector<float> v(nrows);
  CHECK( v.fill(-1.f) );
  CHECK( v.setElement(0.f, 1) );
  CHECK( v.size(&nrows) );

  // Descriptor
  graphblas::Descriptor desc;
  //CHECK( desc.set(graphblas::GrB_MASK, graphblas::GrB_SCMP) );

  // Warmup
  CpuTimer warmup;
  warmup.Start();
  graphblas::assign<float, float>(&v, &m, GrB_NULL, (float)1.f, GrB_ALL, nrows, 
      &desc);
  warmup.Stop();
 
  CpuTimer cpu_vxm;
  //cudaProfilerStart();
  cpu_vxm.Start();
  int NUM_ITER = 1;//0;
  for( int i=0; i<NUM_ITER; i++ )
  {
    graphblas::assign<float, float>(&v, &m, GrB_NULL, (float)1.f, GrB_ALL,
        nrows, &desc);
  }
  //cudaProfilerStop();
  cpu_vxm.Stop();

  float flop = 0;
  if( DEBUG ) std::cout << "warmup, " << warmup.ElapsedMillis() << ", " <<
    flop/warmup.ElapsedMillis()/1000000.0 << "\n";
  float elapsed_vxm = cpu_vxm.ElapsedMillis();
  std::cout << "vxm, " << elapsed_vxm/NUM_ITER << "\n";

  if( DEBUG ) v.print();
  return 0;
}
