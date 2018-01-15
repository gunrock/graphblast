#ifndef GRB_BFS_HPP
#define GRB_BFS_HPP

namespace graphblas
{
  // Use float for now for both v and A
  Info bfs( Vector<float>*       v,
            const Matrix<float>* A, 
            Index                s )
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

    Descriptor desc;
    CHECK( desc.set(GrB_MXVMODE, GrB_PUSHONLY) );

    float d    = 0;
    float succ = false;
    do
    {
      std::cout << "Iteration " << d << ":\n";
      v->print();
      q1.print();
      d++;
      assign<float,float>(v, &q1, GrB_NULL, d, GrB_ALL, n, &desc);
      CHECK( desc.toggle(GrB_MASK) );
      vxm<float,float,float>(&q2, v, GrB_NULL, &GrB_FP32AddMul, &q1, A, &desc);
      CHECK( desc.toggle(GrB_MASK) );
      CHECK( q2.swap(&q1) );
      reduce<float,float>(&succ, GrB_NULL, &GrB_FP32Add, &q2, 
          &desc);
      std::cout << "succ: " << succ << std::endl;
    //} while( d==0 );
    } while( succ>0 );

    return GrB_SUCCESS;
  }
}  // graphblas

#endif  // GRB_BFS_HPP
