#ifndef GRB_BFS_HPP
#define GRB_BFS_HPP

namespace graphblas
{
  Info bfs( const graphblas::Matrix<bool>* A, graphblas::Index s )
  {
    graphblas::Index n;
    CHECK( A->nrows( &n ) );

    graphblas::Vector<int>  v(n);
    graphblas::Vector<bool> q(n);
    std::vector<Index> indices(1,s);
    std::vector<bool>  values(1,true);
    CHECK( q.build( &indices, &values, 1 ) );

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
      CHECK( assign( &v, &q, GrB_NULL, d, GrB_ALL, n, GrB_NULL );

    } while( succ );
  }
}  // graphblas

#endif  // GRB_BFS_HPP
