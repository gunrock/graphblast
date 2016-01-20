// Compiles using:
//   g++ -o graphblas -std=c++11 graphblas.cpp
// Functional:
//   -BuildMatrix (builds matrix in CSC format)
//   -ExtractTuples
//   -MxM
// TO-DO:
//   

#include <unistd.h>
#include <ctype.h>
#include <stdint.h>
#include <vector>
#include <iostream>

#include "graphblas.hpp"

namespace GraphBLAS
{

  // Ignoring + operator, because it is optional argument per GraphBlas_Vops.pdf
  // Store as CSC by default
  // Don't have to assume tuple is ordered
  // TODO: Can be used for CSR by swapping I and J vectors in tuple A and swapping n.cols and n.rows dimensions
  template<typename Scalar>
  void buildMatrix( Matrix<Scalar>& C, std::vector<Index>& I, std::vector<Index>& J, std::vector<Scalar>& val ) {

    Index i, j;
    Index temp;
    Index row;
    Index dest;
    Index cumsum = 0;
    int nnz = I.size();
    C.val.resize(nnz);
    C.rowind.resize(nnz);
    C.colptr.assign(C.ncols+1,0);
    for( i=0; i<nnz; i++ ) {
      C.colptr[J[i]]++;                   // Go through all elements to see how many fall into each row
    }
    for( i=0; i<C.ncols; i++ ) {                  // Cumulative sum to obtain column pointer array
      temp = C.colptr[i];
      C.colptr[i] = cumsum;
      cumsum += temp;
    }
    C.colptr[C.ncols] = nnz;
    for( i=0; i<nnz; i++ ) {
      row = J[i];                         // Store every row index in memory location specified by colptr
      dest = C.colptr[row];
      C.rowind[dest] = I[i];              // Store row index
      C.val[dest] = val[i];                 // Store value
      C.colptr[row]++;                      // Shift destination to right by one
    }
    cumsum = 0;
    for( i=0; i<=C.ncols; i++ ) {                 // Undo damage done by moving destination
      temp = C.colptr[i];
      C.colptr[i] = cumsum;
      cumsum = temp;
  }}

  template<typename Scalar>
  void extractTuples(std::vector<Index>& I, std::vector<Index>& J, std::vector<Scalar>& val, Matrix<Scalar>& A) {
    Index i, j;
    int to_increment = 0;
    I.resize(A.val.size());
    J.resize(A.val.size());
    val.resize(A.val.size());
    for( i=0; i<A.val.size(); i++ ) {
      I[i] = A.rowind[i];                  // Copy from Tuple
      val[i] = A.val[i];                   // Copy from Tuple
    }
    for( i=0; i<A.colptr.size()-1; i++ ) {  // Get j-coordinate from colptr
      to_increment = A.colptr[i+1]-A.colptr[i];
      for( to_increment; to_increment>0; to_increment-- ) {      
        J[i] = i;
  }}}

  // This is overloaded ewiseMult C=A*u. 
  // TODO: C=A*B not written yet.
  template<typename Scalar>
  void ewiseMult( Scalar multiplicand, Vector<Scalar>& A, Vector<Scalar>& C, fnCallDesc& d) {
    Index i;
    C.num = A.num;
    C.rowind.resize(C.num);
    C.val.resize(C.num);
    if( d.getTransformArg1() == TRANSFORM_NEG ) // Only the first argument Transform gets checked
      multiplicand *= -1;                       // when only one argument vector in argument list
    for( i=0; i<A.num; i++ ) {
      C.rowind[i] = A.rowind[i];
      C.val[i] = A.val[i]*multiplicand;
    }
  }

  // This is overloaded ewiseMult C=A*B.
  // TODO: Test performance impact of using multiplicand*A[i]*B[j] vs. -A[i]*B[j]
  //      -there is savings in LOC, but is it worth the performance loss (if any)?
  template<typename Scalar>
  void ewiseMult( Vector<Scalar>& A, Vector<Scalar>& B, Vector<Scalar>& C, fnCallDesc& d ) {
    Index i = 0;
    Index j = 0;
    Scalar multiplicand = 1;
    if((d.getTransformArg1() == TRANSFORM_NEG && d.getTransformArg2() == TRANSFORM_NULL) || 
       (d.getTransformArg1() == TRANSFORM_NULL && d.getTransformArg2() == TRANSFORM_NEG ))
        multiplicand = -1;
    if( d.getAssign()==ASSIGN_NOOP ) {
      C.num = 0;
      C.rowind.clear();
      C.val.clear();
      while( i<A.num && j<B.num ) {
        if( A.rowind[i] == B.rowind[j] ) {
          C.val.push_back( multiplicand*A.val[i]*B.val[j]);
          C.rowind.push_back(A.rowind[i]);
          C.num++;
          i++;
          j++;
        } else if( A.rowind[i] < B.rowind[j] ) {
          i++;
        } else {
          j++;
        }
      }
    }
  }

  // This is overloaded ewiseAdd C=A+B. A+u not written yet. (It also seems redundant if we already have C=A*u?)
  // Standard merge algorithm
  // Checks d.Assign to see whether we are doing C += or C =
  // When B is empty, we do C+=A (implicit, could also have been implemented using templated ewiseAdd
  //   e.g. ewiseAdd( A, C, d )
  // TODO: C+=A+B
  //       C =A*u
  template<typename Scalar>
  void ewiseAdd( Vector<Scalar>& A, Vector<Scalar>& B, Vector<Scalar>& C, fnCallDesc& d ) {
    Index i = 0;
    Index j = 0;
    if( d.getAssign()==ASSIGN_NOOP ) {
      C.num = 0;
      C.rowind.clear();
      C.val.clear();
      if( d.getMultOp() == BINARY_OR ) {
        while( i<A.num && j<B.num ) {
          if( A.rowind[i] == B.rowind[j] ) {
            C.val.push_back(1);
            C.rowind.push_back(1);
            C.num++;
            break;
      }}} else if( d.getTransformArg1() == TRANSFORM_NULL && d.getTransformArg2() == TRANSFORM_NULL ) {
        while( i<A.num && j<B.num ) {
          if( A.rowind[i] == B.rowind[j] ) {
            C.val.push_back(A.val[i] + B.val[j]);
            C.rowind.push_back(A.rowind[i]);
            C.num++;
            i++;
            j++;
          } else if( A.rowind[i] < B.rowind[j] ) {
            C.val.push_back(A.val[i]);
            C.rowind.push_back(A.rowind[i]);
            C.num++;
            i++;
          } else {
          C.val.push_back(B.val[j]);
          C.rowind.push_back(B.rowind[j]);
          C.num++;
          j++;
        }} while( i<A.num ) {
          C.val.push_back(A.val[i]);
          C.rowind.push_back(A.rowind[i]);
          C.num++;
          i++;
        } while( j<B.num ) {
          C.val.push_back(B.val[j]);
          C.rowind.push_back(B.rowind[j]);
          C.num++;
          j++;
        }
      } else if( d.getTransformArg1() == TRANSFORM_NEG && d.getTransformArg2() == TRANSFORM_NULL ) {
        while( i<A.num && j<B.num ) {
          if( A.rowind[i] == B.rowind[j] ) {
            C.val.push_back(B.val[j] - A.val[i]);
            C.rowind.push_back(A.rowind[i]);
            C.num++;
            i++;
            j++;
          } else if( A.rowind[i] < B.rowind[j] ) {
            C.val.push_back(-A.val[i]);
            C.rowind.push_back(A.rowind[i]);
            C.num++;
            i++;
          } else {
          C.val.push_back(B.val[j]);
          C.rowind.push_back(B.rowind[j]);
          C.num++;
          j++;
        }} while( i<A.num ) {
          C.val.push_back(-A.val[i]);
          C.rowind.push_back(A.rowind[i]);
          C.num++;
          i++;
        } while( j<B.num ) {
          C.val.push_back(B.val[j]);
          C.rowind.push_back(B.rowind[j]);
          C.num++;
          j++;
        }
      } else if( d.getTransformArg1() == TRANSFORM_NULL && d.getTransformArg2() == TRANSFORM_NEG ) {
        while( i<A.num && j<B.num ) {
          if( A.rowind[i] == B.rowind[j] ) {
            C.val.push_back(A.val[i]-B.val[j]);
            C.rowind.push_back(A.rowind[i]);
            C.num++;
            i++;
            j++;
          } else if( A.rowind[i] < B.rowind[j] ) {
            C.val.push_back(A.val[i]);
            C.rowind.push_back(A.rowind[i]);
            C.num++;
            i++;
          } else {
          C.val.push_back(-B.val[j]);
          C.rowind.push_back(B.rowind[j]);
          C.num++;
          j++;
        }} while( i<A.num ) {
          C.val.push_back(A.val[i]);
          C.rowind.push_back(A.rowind[i]);
          C.num++;
          i++;
        } while( j<B.num ) {
          C.val.push_back(-B.val[j]);
          C.rowind.push_back(B.rowind[j]);
          C.num++;
          j++;
        }
      } else if( d.getTransformArg1() == TRANSFORM_NEG && d.getTransformArg2() == TRANSFORM_NEG ) {
        while( i<A.num && j<B.num ) {
          if( A.rowind[i] == B.rowind[j] ) {
            C.val.push_back(-A.val[i]-B.val[j]);
            C.rowind.push_back(A.rowind[i]);
            C.num++;
            i++;
            j++;
          } else if( A.rowind[i] < B.rowind[j] ) {
            C.val.push_back(-A.val[i]);
            C.rowind.push_back(A.rowind[i]);
            C.num++;
            i++;
          } else {
          C.val.push_back(-B.val[j]);
          C.rowind.push_back(B.rowind[j]);
          C.num++;
          j++;
        }} while( i<A.num ) {
          C.val.push_back(-A.val[i]);
          C.rowind.push_back(A.rowind[i]);
          C.num++;
          i++;
        } while( j<B.num ) {
          C.val.push_back(-B.val[j]);
          C.rowind.push_back(B.rowind[j]);
          C.num++;
          j++;
        }
      }
    } else if( d.getAssign()==ASSIGN_ADDOP && B.num==0) {
      B.rowind.clear();
      B.val.clear();
      if( d.getTransformArg1() == TRANSFORM_NULL ) {
        while( i<A.num && j<C.num ) {
          if( A.rowind[i] == C.rowind[j] ) {
            B.val.push_back(A.val[i] + C.val[j]);
            B.rowind.push_back(A.rowind[i]);
            B.num++;
            i++;
            j++;
          } else if( A.rowind[i] < C.rowind[j] ) {
            B.val.push_back(A.val[i]);
            B.rowind.push_back(A.rowind[i]);
            B.num++;
            i++;
          } else {
          B.val.push_back(C.val[j]);
          B.rowind.push_back(C.rowind[j]);
          B.num++;
          j++;
        }} while( i<A.num ) {
          B.val.push_back(A.val[i]);
          B.rowind.push_back(A.rowind[i]);
          B.num++;
          i++;
        } while( j<C.num ) {
          B.val.push_back(C.val[j]);
          B.rowind.push_back(C.rowind[j]);
          B.num++;
          j++;
        }
      } else if( d.getTransformArg1() == TRANSFORM_NEG ) {
        while( i<A.num && j<C.num ) {
          if( A.rowind[i] == C.rowind[j] ) {
            B.val.push_back(C.val[j]-A.val[i]);
            B.rowind.push_back(A.rowind[i]);
            B.num++;
            i++;
            j++;
          } else if( A.rowind[i] < C.rowind[j] ) {
            B.val.push_back(-A.val[i]);
            B.rowind.push_back(A.rowind[i]);
            B.num++;
            i++;
          } else {
          B.val.push_back(C.val[j]);
          B.rowind.push_back(C.rowind[j]);
          B.num++;
          j++;
        }} while( i<A.num ) {
          B.val.push_back(-A.val[i]);
          B.rowind.push_back(A.rowind[i]);
          B.num++;
          i++;
        } while( j<C.num ) {
          B.val.push_back(C.val[j]);
          B.rowind.push_back(C.rowind[j]);
          B.num++;
          j++;
        }
      }
      C.num = B.num;
      C.rowind = B.rowind;
      C.val = B.val;
      B.num = 0;
      B.rowind.clear();
      B.val.clear();
    }
  }

  template<typename Scalar>
  void mXv( Vector<Scalar>& C, Matrix<Scalar>& A, Vector<Scalar>& B, fnCallDesc& d ) {
    Index i, j, k;
    Index Acol;
    Scalar value;
    Vector<Scalar> temp;
    Vector<Scalar> empty;
    empty.num = 0;
    Index count = 0;
    C.num = 0;
    C.rowind.clear();
    C.val.clear();
    Scalar mask = d.getMaskDesc();

    Assign old_assign;
    old_assign = d.getAssign();
    d.setAssign(ASSIGN_ADDOP);
  // i = column in B (between 0 and N)
  // j = index of nonzero element in B column
  //    -used to pick out columns of A that we need to do ewisemult on
  // Iterate over nonzero elements of vector B
    if( d.getMultOp() == BINARY_MULT ) {
      for( j=0; j<B.num; j++ ) {
        value = B.val[j];
        Acol = A.colptr[j+1]-A.colptr[j];
		if( ((mask>0 && value == mask) || (mask==0)) && Acol > 0 ) {
          // ewiseMult, store result into temp
          // GraphBLAS::ewiseMult( value, A, A.colptr[j], A.colptr[j+1], temp, d );
          temp.num = Acol;
          temp.rowind.resize(Acol);
          temp.val.resize(Acol);
          count = 0;
          temp.rowind[count] = A.rowind[j];
          temp.val[count] = A.val[j]*value;
          GraphBLAS::ewiseAdd( temp, empty, C, d );
          //for( k=0; k<result.num; k++ )
          //  std::cout << j << result.rowind[k] << result.val[k] << std::endl;
	}}} else if( d.getMultOp() == BINARY_ADD ) {
      for( j=0; j<B.num; j++ ) {
        value = B.val[j];
        Acol = A.colptr[j+1]-A.colptr[j];
        if( Acol > 0 ) {
          // ewiseMult, store result into temp
          // GraphBLAS::ewiseMult( d, value, A, A.colptr[j], A.colptr[j+1], temp );
          temp.num = Acol;
          temp.rowind.resize(Acol);
          temp.val.resize(Acol);
          count = 0;
          temp.rowind[count] = A.rowind[j];
          temp.val[count] = 1;
          GraphBLAS::ewiseAdd( temp, empty, C, d );
          //for( k=0; k<result.num; k++ )
    }}}
    d.setAssign(old_assign);
  }

  // Could also have template where matrices A and B have different values as Manoj/Jose originally had in their signature, but sake of simplicity assume they have same ScalarType. Also omitted optional mask m for   sake of simplicity.
  // Also omitting safety check that sizes of A and B s.t. they can be multiplied
  template<typename Scalar>
  void mXm(Matrix<Scalar>& C, Matrix<Scalar>& A, Matrix<Scalar>& B, fnCallDesc& d) {
    Index i, j, k;
    Index N = B.colptr.size()-1;
    Index Acol, Bcol;
    Scalar value;
    Vector<Scalar> temp;
    Vector<Scalar> empty;
    empty.num = 0;
    Vector<Scalar> result;
    Index count = 0;
    C.colptr.clear();
    C.rowind.clear();
    C.val.clear();
    C.colptr.push_back(0);

    Assign old_assign;
    old_assign = d.getAssign();
    d.setAssign(ASSIGN_ADDOP);
  // i = column in B (between 0 and N)
  // j = index of nonzero element in B column
  //    -used to pick out columns of A that we need to do ewisemult on
    for( i=0; i<N; i++ ) {
      Bcol = B.colptr[i+1]-B.colptr[i];
  // Skip columns in B that are empty
      if( Bcol>0 ) {
  // Iterate over nonzero elements of first column of matrix B
        for( j=B.colptr[i]; j<B.colptr[i+1]; j++ ) {
          value = B.val[j];
          Acol = A.colptr[j+1]-A.colptr[j];
          result.num = 0;
          result.rowind.clear();
          result.val.clear();
          if( Acol > 0 ) {
            // ewiseMult, store result into temp
            // GraphBLAS::ewiseMult( d, value, A, A.colptr[j], A.colptr[j+1], temp );
            temp.num = Acol;
            temp.rowind.resize(Acol);
            temp.val.resize(Acol);
            count = 0;
            temp.rowind[count] = A.rowind[j];
            temp.val[count] = A.val[j]*value;
            GraphBLAS::ewiseAdd( temp, empty, result, d );
            //for( k=0; k<result.num; k++ )
            //  std::cout << j << result.rowind[k] << result.val[k] << std::endl;
          }
        }
        // Write result into C and advance colptr;
        // GraphBLAS::ewiseAdd( result, C );
        if( i>0 )
          C.colptr.push_back(C.colptr[i-1]+result.num);
        for( j=0; j<result.num; j++ ) {
          C.rowind.push_back(result.rowind[j]);
          C.val.push_back(result.val[j]);
  }}}
    d.setAssign(old_assign);
  }

  namespace app {

    // initFrontier is boolean vector initial frontier
    // Define 
    template<typename Scalar>
    void bfsMasked( Matrix<Scalar>& Graph, Vector<Scalar>& initFrontier, Vector<Scalar>& bfsResult ) {

      // BFS semi-ring
      fnCallDesc d;
      d.setAddOp( BINARY_OR );
      d.setMultOp( BINARY_AND );

      // Only update the values that are IDENTITY_MIN (i.e. infinity)
      fnCallDesc e;
      e.setAddOp( BINARY_MIN );
      e.setMultOp( BINARY_MULT );
      e.setAddId( IDENTITY_MIN );
      e.setAssign( ASSIGN_ADDOP );

      Vector<Scalar> tempFrontier = initFrontier;
      Vector<Scalar> tempFrontier2;
      Vector<Scalar> empty;

      Scalar depth = 0;

      for( depth; depth<1; depth++ ) {
        GraphBLAS::ewiseMult( depth, tempFrontier, tempFrontier2, e );
        tempFrontier.print();
		tempFrontier2.print();
		GraphBLAS::ewiseAdd( bfsResult, empty, tempFrontier2, e );

        // Only perform mXv on elements in bfsResult vector that == depth
        d.setMaskDesc( depth );
        GraphBLAS::mXv( tempFrontier, Graph, bfsResult, d );
    }}
}}

int main() {

  std::vector<GraphBLAS::Index> I = {0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4};
  std::vector<GraphBLAS::Index> J = {1, 0, 2, 3, 1, 3, 4, 1, 2, 4, 2, 3};
  std::vector<int> val            = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  
  // Construct 3x3 matrices
  GraphBLAS::Matrix<int> A(5, 5);
  //GraphBLAS::Matrix<int> B(3, 3);
  //GraphBLAS::Matrix<int> C(3, 3);

  // Initialize 3x3 matrices
  GraphBLAS::buildMatrix<int>(A, I, J, val);
  //GraphBLAS::buildMatrix<int>(B, I, J, val);

  GraphBLAS::fnCallDesc d;
  //GraphBLAS::mXm<int>(C, A, B, d);
  //GraphBLAS::extractTuples<int>(I, J, val, C);

  // Initialize vector a, b
  GraphBLAS::Vector<int> a, b;
  a.num = 1;
  a.rowind.push_back(1);
  a.val.push_back(1);
  GraphBLAS::mXv<int>(b, A, a, d);

  // Masked BFS
  GraphBLAS::app::bfsMasked<int>( A, a, b );

  //for( int i=0; i<J.size(); i++ )
  //  std::cout << J[i] << std::endl;
  //for( int i=0; i<val.size(); i++ )
  //  std::cout << val[i] << std::endl;
  for( GraphBLAS::Index i=0; i<b.num; i++ )
    std::cout << b.rowind[i] << b.val[i] << std::endl;

  return 0;
}
