#ifndef GRB_ALGORITHM_LGC_HPP
#define GRB_ALGORITHM_LGC_HPP

//#include "graphblas/algorithm/testLgc.hpp"
#include "graphblas/backend/apspie/util.hpp" // GpuTimer

namespace graphblas
{
namespace algorithm
{
  // Use float for now for both p and A
  float lgc( Vector<float>*       p,      // PageRank result
             const Matrix<float>* A,      // graph
             Index                s,      // source vertex
						 double               alpha,  // teleportation constant in (0,1]
						 double               eps,    // tolerance
		         Descriptor*          desc )
  {
    Index n;
    CHECK( A->nrows( &n ) );

		// degrees: compute the degree of each node
		Vector<float> degrees(n);
		reduce<float, float, float>(&degrees, GrB_NULL, GrB_NULL, 
        PlusMonoid<float>(), A, desc);

    // pagerank (p): initialized to 0
    CHECK( p->fill(0.f) );

		// residual (r): initialized to 0 except source to 1
    Vector<float> r(n);
		std::vector<Index> indices(1, s);
		std::vector<float> values(1, 1.f);

    // residual2 (r2)
		Vector<float> r2(n);
    CHECK( r2.fill(0.f) );

    Desc_value desc_value;
    CHECK( desc->get(GrB_MXVMODE, &desc_value) );
    if( desc_value==GrB_PULLONLY )
    {
      CHECK( r.fill(0.f) );
      CHECK( r.setElement(1.f,s) );
    }
    else
    {
      CHECK( r.build(&indices, &values, 1, GrB_NULL) );
    }

		// degrees_eps (d x eps): precompute degree of each node times eps
	  Vector<float> eps_vector(n);
		Vector<float> degrees_eps(n);
		CHECK( eps_vector.fill(eps) );
		eWiseMult<float, float, float, float>( &degrees_eps, GrB_NULL, GrB_NULL,
        PlusMultipliesSemiring<float>(), &degrees, &eps_vector, desc );
    degrees_eps.print();

    // frontier (f): portion of r(v) >= degrees(v) x eps
		// (use float for now)
		Vector<float> f(n);
    CHECK( f.build(&indices, &values, 1, GrB_NULL) );

		// alpha: TODO(@ctcyang): introduce vector-constant eWiseMult
		Vector<float> alpha_vector(n);
		CHECK( alpha_vector.fill(alpha) );
    Vector<float> alpha_vector2(n);
		CHECK( alpha_vector2.fill((1.-alpha)/2.) );

    Index nvals;
    CHECK( f.nvals(&nvals) );

    std::cout << "frontier (f): " << std::endl;
    CHECK( f.print() );
    std::cout << "residual (r): " << std::endl;
    CHECK( r.print() );
    std::cout << "pagerank (p): " << std::endl;
    CHECK( p->print() );
    std::cout << "degrees (d): " << std::endl;
    CHECK( degrees.print() );
    std::cout << "degrees_eps (d x eps): " << std::endl;
    CHECK( degrees_eps.print() );

    Index iter = 0;
    float succ;
    backend::GpuTimer cpu_tight;
    if( desc->descriptor_.timing_>0 )
      cpu_tight.Start();
    do
    {
      iter++;
      std::cout << "=====Begin iteration " << iter << "=====\n";

      // p = p + alpha * r .* f
      CHECK( desc->toggle(GrB_MASK) );
      eWiseMult<float, float, float, float>(&r2, &f, GrB_NULL, 
          PlusMultipliesSemiring<float>(), &r, &alpha_vector, desc);
      CHECK( desc->toggle(GrB_MASK) );
      std::cout << "residual2 (r2): " << std::endl;
      CHECK( r2.print() );
      eWiseAdd<float, float, float, float>(p, GrB_NULL, GrB_NULL, 
          PlusMultipliesSemiring<float>(), p, &r2, desc);
      std::cout << "pagerank (p): " << std::endl;
      CHECK( p->print() );

      // r = (1 - alpha)/2 * r
      eWiseMult<float, float, float, float>(&r, GrB_NULL, GrB_NULL,
          PlusMultipliesSemiring<float>(), &r, &alpha_vector2, desc);
      std::cout << "residual (r): " << std::endl;
      CHECK( r.print() );

      // r2 = r/d .* f
      CHECK( desc->toggle(GrB_MASK) );
      //eWiseMult<float, float, float, float>(&r2, &f, GrB_NULL, divides<float>(),
      //    &r, &degrees, desc);
      eWiseMult<float, float, float, float>(&r2, &f, GrB_NULL, 
          PlusDividesSemiring<float>(), &r, &degrees, desc);
      CHECK( desc->toggle(GrB_MASK) );
      std::cout << "residual2 (r2): " << std::endl;
      CHECK( r2.print() );

      // r = r + A^T * r2
      mxv<float, float, float, float>(&r, GrB_NULL, PlusMonoid<float>(), 
          PlusMultipliesSemiring<float>(), A, &r2, desc);

      // f = {v | r(v) >= d* eps}
      eWiseAdd<float, float, float, float>(&f, GrB_NULL, GrB_NULL, 
          GreaterPlusSemiring<float>(), &r, &degrees_eps, desc);
      //eWiseMult<float, float, float, float>(&f, GrB_NULL, GrB_NULL, 
      //    PlusGreaterSemiring<float>(), &r, &degrees_eps, desc);

      // Update frontier size
      reduce<float, float>(&succ, GrB_NULL, PlusMonoid<float>(), &f, desc);

      std::cout << "succ: " << succ << std::endl;
      std::cout << "frontier (f): " << std::endl;
      CHECK( f.print() );
      std::cout << "residual (r): " << std::endl;
      CHECK( r.print() );
      std::cout << "pagerank (p): " << std::endl;
      CHECK( p->print() );
      if (iter >= desc->descriptor_.max_niter_)
        break;
    } while (succ > 0);
    if( desc->descriptor_.timing_>0 )
    {
      cpu_tight.Stop();
      std::cout << "elapsed time: " << cpu_tight.ElapsedMillis() << "\n";
      return cpu_tight.ElapsedMillis();
    }
    return 0.f;
    //return GrB_SUCCESS;
  }

  template <typename T, typename a>
  int lgcCpu( Index        source,
               Matrix<a>*   A,
               T*           h_lgc_cpu,
							 Index        depth,
               bool         transpose=false )
  {
		Index* reference_check_preds = NULL;
    int max_depth;

    /*if( transpose )
		  max_depth = SimpleReferenceLgc<T>( A->matrix_.nrows_, 
          A->matrix_.sparse_.h_cscColPtr_, A->matrix_.sparse_.h_cscRowInd_, 
          h_lgc_cpu, reference_check_preds, source, depth);
    else
		  max_depth = SimpleReferenceLgc<T>( A->matrix_.nrows_, 
          A->matrix_.sparse_.h_csrRowPtr_, A->matrix_.sparse_.h_csrColInd_, 
          h_lgc_cpu, reference_check_preds, source, depth);

		printArray(h_lgcResultCPU, m);*/
		return max_depth; 
	}

}  // algorithm
}  // graphblas

#endif  // GRB_ALGORITHM_LGC_HPP
