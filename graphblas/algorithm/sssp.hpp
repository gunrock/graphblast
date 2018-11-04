#ifndef GRB_ALGORITHM_SSSP_HPP
#define GRB_ALGORITHM_SSSP_HPP

#include <limits>

#include "graphblas/algorithm/testSssp.hpp"
#include "graphblas/backend/apspie/util.hpp" // GpuTimer

namespace graphblas
{
namespace algorithm
{
  // Use float for now for both v and A
  float sssp( Vector<float>*       v,
              const Matrix<float>* A, 
              Index                s,
		          Descriptor*          desc )
  {
    Index n;
    CHECK( A->nrows( &n ) );

    Desc_value desc_value;
    CHECK( desc->get(GrB_MXVMODE, &desc_value) );

    // Visited vector (use float for now)
    if( desc_value==GrB_PULLONLY )
    {
      CHECK( v->fill(std::numeric_limits<float>::max()) );
      CHECK( v->setElement(0.f,s) );
    }
    else
    {
      std::vector<Index> indices(1,s);
      std::vector<float>  values(1,0.f);
      CHECK( v->build(&indices, &values, 1, GrB_NULL) );
    }

    Index iter;
    Index d_last = 0;
    Index d_curr = 0;
    float succ_last = 0.f;
    float succ_curr = 0.f;
    Index A_nrows;
    CHECK( A->nrows(&A_nrows) );

    backend::GpuTimer cpu_tight;
    if( desc->descriptor_.timing_>0 )
      cpu_tight.Start();
    do
    {
      if( desc->descriptor_.debug() )
      {
        std::cout << "Iteration " << iter << ":\n";
        std::cout << "v:\n";
        v->print();
      }
      iter++;
      vxm<float,float,float,float>(v, GrB_NULL, GrB_NULL,
          MinimumPlusSemiring<float>(), v, A, desc);
      v->nvals(&d_curr);
      d_last = d_curr - d_last;
      reduce<float,float>(&succ_curr, GrB_NULL, PlusMonoid<float>(), v, desc);
      succ_last = succ_curr - succ_last;

      if( desc->descriptor_.debug() )
        std::cout << "succ: " << succ_last << std::endl;
    } while (succ_last > 0 || d_last > 0);
    if( desc->descriptor_.timing_>0 )
    {
      cpu_tight.Stop();
      std::cout << iter-1 << ", " << cpu_tight.ElapsedMillis() << "\n";
      return cpu_tight.ElapsedMillis();
    }
    return 0.f;
    //return GrB_SUCCESS;
  }

  template <typename T, typename a>
  int ssspCpu( Index        source,
               Matrix<a>*   A,
               T*           h_sssp_cpu,
							 Index        depth,
               bool         transpose=false )
  {
    int max_depth;

    if( transpose )
		  max_depth = SimpleReferenceSssp<T>( A->matrix_.nrows_, 
          A->matrix_.sparse_.h_cscColPtr_, A->matrix_.sparse_.h_cscRowInd_, 
          A->matrix_.sparse_.h_cscVal_, h_sssp_cpu, source, depth);
    else
		  max_depth = SimpleReferenceSssp<T>( A->matrix_.nrows_, 
          A->matrix_.sparse_.h_csrRowPtr_, A->matrix_.sparse_.h_csrColInd_, 
          A->matrix_.sparse_.h_csrVal_, h_sssp_cpu, source, depth);

		return max_depth; 
	}

}  // algorithm
}  // graphblas

#endif  // GRB_ALGORITHM_SSSP_HPP
