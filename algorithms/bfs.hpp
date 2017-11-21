#ifndef GRB_BFS_HPP
#define GRB_BFS_HPP

namespace graphblas
{
  Info bfs( const graphblas::Matrix<bool>* A, graphblas::Index s )
  {
    graphblas::Index n;
    CHECK( A->nrows( &n ) );

    graphblas::Vector<int>  v(n);
    graphblas::Vector<bool> q1(n);
    graphblas::Vector<bool> q2(n);
    std::vector<Index> indices(1,s);
    std::vector<bool>  values(1,true);
    CHECK( q1.build( &indices, &values, 1 ) );

    graphblas::BinaryOp GrB_LOR(  graphblas::logical_or() );
    graphblas::BinaryOp GrB_LAND( graphblas::logical_and() );
    graphblas::Monoid   GrB_Lor( GrB_LOR, false );
    graphblas::Semiring GrB_Boolean( GrB_Lor, GrB_LAND );

    graphblas::Descriptor desc_nomask;

    graphblas::Index d = 0;
    bool succ = false;
    do
    {
      d++;
      CHECK( assign( &v, &q1, GrB_NULL, d, GrB_ALL, n, &desc_nomask );
      CHECK( vxm(    &q2, &v, GrB_NULL, GrB_Boolean, &q1, A, &desc_nomask );
      CHECK( reduce( &succ, GrB_NULL, GrB_Lor, &q2, &desc_nomask );
    } while( succ );

    return GrB_SUCCESS;
  }
}  // graphblas

#endif  // GRB_BFS_HPP
