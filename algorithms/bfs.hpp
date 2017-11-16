#ifndef GRB_BFS_HPP
#define GRB_BFS_HPP

namespace graphblas
{
  Info bfs( const Matrix* A, Index s )
  {
    Index n;
    CHECK( A.nrows( &n ) );

    Vector v(n);
  }

}  // graphblas

#endif  // GRB_BFS_HPP
