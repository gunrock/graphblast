#ifndef GRB_ALGORITHM_LGC_HPP
#define GRB_ALGORITHM_LGC_HPP

#include "graphblas/algorithm/testLgc.hpp"
#include "graphblas/backend/apspie/util.hpp" // GpuTimer

namespace graphblas
{
namespace algorithm
{
  // Use float for now for both v and A
  float lgc( Vector<float>*       v,      // PageRank result
             const Matrix<float>* A,      // graph
             Index                s,      // source vertex
						 double               alpha,  // teleportation constant in (0,1]
						 double               eps,    // tolerance
		         Descriptor*          desc )
  {
    Index n;
    CHECK( A->nrows( &n ) );

		// degrees: compute the degree of each node
		Vector<float> d(n);
		reduce<float, float>(&d, GrB_NULL, GrB_NULL, PlusMonoid<float>(), A, desc);

    // PageRank (v): initialized to 0
    CHECK( v->fill(0.f) );

		// residual vectors (r1, r2): initialized to 0 except source to 1
    Vector<float> r1(n);
		std::vector<Index> indices(1, s);
		std::vector<float> values(1, 1.f);
		CHECK( r1.build(&indices, &values, 1, GrB_NULL) );

		Vector<float> r2(n);

    Desc_value desc_value;
    CHECK( desc->get(GrB_MXVMODE, &desc_value) );
    if( desc_value==GrB_PULLONLY )
    {
      CHECK( r1.fill(0.f) );
      CHECK( r1.setElement(1.f,s) );
    }
    else
    {
      std::vector<Index> indices(1, s);
      std::vector<float>  values(1, 1.f);
      CHECK( r1.build(&indices, &values, 1, GrB_NULL) );
    }

		// degrees_eps (d x eps): precompute degree of each node times eps
	  Vector<float> eps_vector(n);
		Vector<float> degrees_eps(n);
		CHECK( eps_vector->fill(eps) );
		ewiseMult<float,float>( degrees_eps, GrB_NULL, GrB_NULL, TimesMonoid<float>(), d, eps_vector );

    // frontier vectors (f): portion of r(v) >= d(v) x eps
		// (use float for now)
		Vector<float> f(n);
    std::vector<Index> indices(1, s);
    std::vector<float>  values(1, 1.f);
    CHECK( r1.build(&indices, &values, 1, GrB_NULL) );

		// alpha vectors: TODO(@ctcyang): introduce vector-constant eWiseMult
		Vector<float> alpha_vector(n);
		CHECK( alpha_vector->fill(alpha) );
    Vector<float> alpha_vector2(n);
		CHECK( alpha_vector2->fill((1.-alpha)/2.) );

    Index nvals;
    CHECK( frontier->nvals(&nvals) );

    Index frontier_size;
    backend::GpuTimer cpu_tight;
    if( desc->descriptor_.timing_>0 )
      cpu_tight.Start();
    do
    {
      if( desc->descriptor_.debug() )
      {
        std::cout << "Iteration " << d << ":\n";
        v->print();
        q1.print();
      }
      if( desc->descriptor_.timing_==2 )
      {
        cpu_tight.Stop();
        if( d!=0 )
          std::cout << d-1 << ", " << frontier_size << ", " << unvisited << ", " << cpu_tight.ElapsedMillis() << "\n";
        frontier  = (int)succ;
        unvisited -= (int)succ;
        cpu_tight.Start();
      }
      d++;
      assign<float,float>(v, &q1, GrB_NULL, d, GrB_ALL, n, desc);
      CHECK( desc->toggle(GrB_MASK) );
      vxm<float,float,float>(&q2, v, GrB_NULL, 
          PlusMultipliesSemiring<float>(), &q1, A, desc);
      CHECK( desc->toggle(GrB_MASK) );
      CHECK( q2.swap(&q1) );
      reduce<float,float>(&succ, GrB_NULL, PlusMonoid<float>(), &q1, desc);

      if( desc->descriptor_.debug() )
        std::cout << "succ: " << succ << " " << (int)succ << std::endl;
    } while( succ>0 );
    if( desc->descriptor_.timing_>0 )
    {
      cpu_tight.Stop();
      std::cout << d-1 << ", " << frontier << ", " << unvisited << ", " << cpu_tight.ElapsedMillis() << "\n";
      return cpu_tight.ElapsedMillis();
    }
    return 0.f;
    //return GrB_SUCCESS;
  }

  // Use float for now for both v and A
  float bfs2( Vector<float>*       v,
             const Matrix<float>* A, 
             Index                s,
		         Descriptor*          desc,
             int                  depth )
  {
    Index n;
    CHECK( A->nrows( &n ) );

    // Visited vector (use float for now)
    CHECK( v->fill(0.f) );

    // Frontier vectors (use float for now)
    Vector<float> q1(n);
    Vector<float> q2(n);

    Desc_value desc_value;
    CHECK( desc->get(GrB_MXVMODE, &desc_value) );
    if( desc_value==GrB_PULLONLY )
    {
      CHECK( q1.fill(0.f) );
      CHECK( q1.setElement(1.f,s) );
    }
    else
    {
      std::vector<Index> indices(1,s);
      std::vector<float>  values(1,1.f);
      CHECK( q1.build(&indices, &values, 1, GrB_NULL) );
    }

    backend::GpuTimer cpu_tight;
    cpu_tight.Start();
    for( int i=1; i<=depth; i++ )
    {
      assign<float,float>(v, &q1, GrB_NULL, i, GrB_ALL, n, desc);
      CHECK( desc->toggle(GrB_MASK) );
      vxm<float,float,float>(&q2, v, GrB_NULL, 
          PlusMultipliesSemiring<float>(), &q1, A, desc);
      CHECK( desc->toggle(GrB_MASK) );
      CHECK( q2.swap(&q1) );
    }
    cpu_tight.Stop();
    return cpu_tight.ElapsedMillis();
  }

  template <typename T, typename a>
  int bfsCpu( Index        source,
               Matrix<a>*   A,
               T*           h_bfs_cpu,
							 Index        depth,
               bool         transpose=false )
  {
		Index* reference_check_preds = NULL;
    int max_depth;

    if( transpose )
		  max_depth = SimpleReferenceBfs<T>( A->matrix_.nrows_, 
          A->matrix_.sparse_.h_cscColPtr_, A->matrix_.sparse_.h_cscRowInd_, 
          h_bfs_cpu, reference_check_preds, source, depth);
    else
		  max_depth = SimpleReferenceBfs<T>( A->matrix_.nrows_, 
          A->matrix_.sparse_.h_csrRowPtr_, A->matrix_.sparse_.h_csrColInd_, 
          h_bfs_cpu, reference_check_preds, source, depth);

		//print_array(h_bfsResultCPU, m);
		return max_depth; 
	}

}  // algorithm
}  // graphblas

#endif  // GRB_ALGORITHM_LGC_HPP
