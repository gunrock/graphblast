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

  // Matrix A
  graphblas::Matrix<float> a(nrows, ncols);
  a.build( &row_indices, &col_indices, &values, nvals, GrB_NULL );
  a.nrows( &nrows );
  a.ncols( &ncols );
  a.nvals( &nvals );
  if( DEBUG ) a.print();

  // Vector x
  graphblas::Vector<float> x(nrows);
  CHECK( x.fill( 0.f ) );
  CHECK( x.setElement(1.f, 0) );
  CHECK( x.size( &nrows ) );
  if( DEBUG ) x.print();

  // Vector y
  graphblas::Vector<float> y(nrows);
  if( DEBUG ) y.print();

  // Mask
  graphblas::Vector<float> m(nrows);
  CHECK( m.fill(-1.f) );
  CHECK( m.setElement(0.f, 0) );
  CHECK( m.size(&nrows) );
  if( DEBUG ) CHECK( m.print() );

  // Descriptor
  graphblas::Descriptor desc;
  CHECK( desc.set(graphblas::GrB_MASK, graphblas::GrB_SCMP) );

  // Warmup
  CpuTimer warmup;
  warmup.Start();
  graphblas::vxm<float, float, float>( &y, &m, GrB_NULL,
      graphblas::PlusMultipliesSemiring<float>(), &x, &a, &desc );
  //graphblas::vxm<float, float, float>( &y, GrB_NULL, GrB_NULL, &GrB_FP32AddMul, 
  //    &x, &a, &desc );
  warmup.Stop();
 
  CpuTimer cpu_vxm;
  //cudaProfilerStart();
  cpu_vxm.Start();
  int NUM_ITER = 10;
  for( int i=0; i<NUM_ITER; i++ )
  {
    graphblas::vxm<float, float, float>( &y, &m, GrB_NULL, 
        graphblas::PlusMultipliesSemiring<float>(), &x, &a, &desc );
    //graphblas::vxm<float, float, float>( &y, GrB_NULL, GrB_NULL, 
    //    &GrB_FP32AddMul, &x, &a, &desc );
  }
  //cudaProfilerStop();
  cpu_vxm.Stop();

  float flop = 0;
  if( DEBUG ) std::cout << "warmup, " << warmup.ElapsedMillis() << ", " <<
    flop/warmup.ElapsedMillis()/1000000.0 << "\n";
  float elapsed_vxm = cpu_vxm.ElapsedMillis();
  std::cout << "vxm, " << elapsed_vxm/NUM_ITER << "\n";

  if( DEBUG ) y.print();
  /*c.extractTuples( out_denseVal );
  for( int i=0; i<nvals; i++ )
  {
    graphblas::Index row = row_indices[i];
    graphblas::Index col = col_indices[i];
    float            val = values[i];
    if( col<max_ncols )
    {
      // Row major order
      if( ROW_MAJOR )
      //std::cout << row << " " << col << " " << val << " " << out_denseVal[row*max_ncols+col] << std::endl;
        BOOST_ASSERT( val==out_denseVal[row*max_ncols+col] );
      else
      // Column major order
      //std::cout << row << " " << col << " " << val << " " << out_denseVal[col*nrows+row] << std::endl;
        BOOST_ASSERT( val==out_denseVal[col*nrows+row] );
    }
  }*/
  return 0;
}
