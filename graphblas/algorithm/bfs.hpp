#ifndef GRB_ALGORITHM_BFS_HPP
#define GRB_ALGORITHM_BFS_HPP

#include "graphblas/algorithm/testBfs.hpp"

namespace graphblas
{
namespace algorithm
{
  // Use float for now for both v and A
  Info bfs( Vector<float>*       v,
            const Matrix<float>* A, 
            Index                s,
		        Descriptor*          desc,
            bool                 transpose=false )
  {
    Index n;
    CHECK( A->nrows( &n ) );

    // Visited vector (use float for now)
    CHECK( v->fill(-1.f) );

    // Frontier vectors (use float for now)
    Vector<float> q1(n);
    Vector<float> q2(n);
    std::vector<Index> indices(1,s);
    std::vector<float>  values(1,1.f);
    CHECK( q1.build(&indices, &values, 1, GrB_NULL) );

    // Semiring
    /*BinaryOp GrB_LOR(  logical_or() );
    BinaryOp GrB_LAND( logical_and() );
    Monoid   GrB_Lor( GrB_LOR, false );
    Semiring GrB_Boolean( GrB_Lor, GrB_LAND );*/
		BinaryOp<float,float,float> GrB_PLUS_FP32;
		GrB_PLUS_FP32.nnew( plus<float>() );
		BinaryOp<float,float,float> GrB_TIMES_FP32;
		GrB_TIMES_FP32.nnew( multiplies<float>() );
		Monoid  <float> GrB_FP32Add;
		GrB_FP32Add.nnew( GrB_PLUS_FP32, 0.f );
		Semiring<float,float,float> GrB_FP32AddMul;
		GrB_FP32AddMul.nnew( GrB_FP32Add, GrB_TIMES_FP32 );

    CHECK( desc->set(GrB_MXVMODE, GrB_PUSHONLY) );

    float d    = 0;
    float succ = false;
    CpuTimer cpu_tight;
    cpu_tight.Start();
    do
    {
      if( GrB_DEBUG )
      {
        std::cout << "Iteration " << d << ":\n";
        v->print();
        q1.print();
        std::cout << "succ: " << succ << std::endl;
      }
      assign<float,float>(v, &q1, GrB_NULL, d, GrB_ALL, n, desc);
      CHECK( desc->toggle(GrB_MASK) );
      if( transpose )
        mxv<float,float,float>(&q2, v, GrB_NULL, &GrB_FP32AddMul, A, &q1, desc);
      else
        vxm<float,float,float>(&q2, v, GrB_NULL, &GrB_FP32AddMul, &q1, A, desc);
      CHECK( desc->toggle(GrB_MASK) );
      CHECK( q2.swap(&q1) );
      reduce<float,float>(&succ, GrB_NULL, &GrB_FP32Add, &q2, desc);
      d++;
    } while( succ>0 );
    cpu_tight.Stop();
    std::cout << "tight, " << cpu_tight.ElapsedMillis() << ", \n";

    return GrB_SUCCESS;
  }

  template <typename T, typename a>
  Info bfsCpu( Index        source,
               Matrix<a>*   A,
               T*           h_bfs_cpu,
							 Index        depth,
               bool         transpose=false )
  {
		Index* reference_check_preds = NULL;

    if( transpose )
		  SimpleReferenceBfs<T>( A->matrix_.nrows_, A->matrix_.sparse_.h_cscColPtr_,
          A->matrix_.sparse_.h_cscRowInd_, h_bfs_cpu, reference_check_preds, 
          source, depth);
    else
		  SimpleReferenceBfs<T>( A->matrix_.nrows_, A->matrix_.sparse_.h_csrRowPtr_,
          A->matrix_.sparse_.h_csrColInd_, h_bfs_cpu, reference_check_preds, 
          source, depth);

		//print_array(h_bfsResultCPU, m);
		return GrB_SUCCESS; 
	}

}  // algorithm
}  // graphblas

#endif  // GRB_ALGORITHM_BFS_HPP
