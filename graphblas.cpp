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

namespace GraphBLAS
{

  typedef uint64_t Index;

  // Sparse (CSC format with single value instead of colptr) by default
  // Temporarily keeping member variables public to avoid having to use get
  template<typename T>
  class Vector {
    public:
      Index                 num;
      std::vector<Index> rowind;
      std::vector<T>        val;
  };

  // CSC format by default
  // Temporarily keeping member variables public to avoid having to use get
  template<typename T>
  class Matrix {
    public:
      std::vector<Index> rowind;
      std::vector<Index> colptr;
      std::vector<T>        val;
  };

  template<typename T>
  class Tuple {
    public:
      std::vector<Index> I;
      std::vector<Index> J;
      std::vector<T>   val;
  };
  
  // Input argument preprocessing functions.
  enum Transform {
    argDesc_null =	0,
    argDesc_neg  =	1,
    argDesc_T 	 =	2,
    argDesc_negT =	3,
    argDesc_notT =	4
  };

// Next is the relevant number of assignment operators.  Since Boolean data is
// of significant interest, I have added the stAnd and stOr ops for now
  enum Assign {
    assignDesc_st   =	0,	/* Simple assignment */
    assignDesc_stOp =	1	/* Store with Circle plus */
  };

// List of ops that can be used in map/reduce operations.
  enum BinaryOp {
    fieldOps_mult =	0,
    fieldOps_add  =	1,
    fieldOps_and  =	2,
    fieldOps_or	  =	3
  };

// List of additive identities that can be used
  enum AdditiveId {
    addId_add =         0,
    addId_min =         1,
    addId_max =         2
  };

  class fnCallDesc {
    Assign assignDesc ;
    Transform arg1Desc ;
    Transform arg2Desc ;
    Transform maskDesc ;
    int32_t dim ;			// dimension for reduction operation on matrices
    BinaryOp addOp ;
    BinaryOp multOp ;
    AdditiveId addId ;
    public:
      fnCallDesc( const std::string& semiring = "Matrix Multiply" ):
        assignDesc(assignDesc_st),
        arg1Desc(argDesc_null),
        arg2Desc(argDesc_null),
        maskDesc(argDesc_null),
        dim(1),
        addOp(fieldOps_add),
        multOp(fieldOps_mult),
        addId(addId_add)
      { }
      Assign getAssign() const { 
        return assignDesc; }
      void setAssign(Assign state) { 
        assignDesc = state; }
  };

  // Ignoring + operator, because it is optional argument per GraphBlas_Vops.pdf
  // Store as CSC by default
  // Don't have to assume tuple is ordered
  // Can be used for CSR by swapping I and J vectors in tuple A and swapping N and M dimensions
  template<typename Scalar>
  void buildMatrix(int M, int N, Tuple<Scalar>& A, Matrix<Scalar>& C) {
    Index i, j;
    Index temp;
    Index row;
    Index dest;
    Index cumsum = 0;
    int nnz = A.I.size();
    C.val.resize(nnz);
    C.rowind.resize(nnz);
    C.colptr.assign(N+1,0);
    for( i=0; i<nnz; i++ ) {
      C.colptr[A.J[i]]++;                   // Go through all elements to see how many fall into each row
    }
    for( i=0; i<N; i++ ) {                  // Cumulative sum to obtain column pointer array
      temp = C.colptr[i];
      C.colptr[i] = cumsum;
      cumsum += temp;
    }
    C.colptr[N] = nnz;
    for( i=0; i<nnz; i++ ) {
      row = A.J[i];                         // Store every row index in memory location specified by colptr
      dest = C.colptr[row];
      C.rowind[dest] = A.I[i];              // Store row index
      C.val[dest] = A.val[i];                 // Store value
      C.colptr[row]++;                      // Shift destination to right by one
    }
    cumsum = 0;
    for( i=0; i<=N; i++ ) {                 // Undo damage done by moving destination
      temp = C.colptr[i];
      C.colptr[i] = cumsum;
      cumsum = temp;
  }}

  template<typename Scalar>
  void extractTuples(Matrix<Scalar>& A, Tuple<Scalar>& C) {
    Index i, j;
    int to_increment = 0;
    C.I.resize(A.val.size());
    C.J.resize(A.val.size());
    C.val.resize(A.val.size());
    for( i=0; i<A.val.size(); i++ ) {
      C.I[i] = A.rowind[i];                  // Copy from Tuple
      C.val[i] = A.val[i];                   // Copy from Tuple
    }
    for( i=0; i<A.colptr.size()-1; i++ ) {  // Get j-coordinate from colptr
      to_increment = A.colptr[i+1]-A.colptr[i];
      for( to_increment; to_increment>0; to_increment-- ) {      
        C.J[i] = i;
  }}}

  // This is overloaded ewiseMult C=A*u. 
  // TODO: C=A*B not written yet.
  template<typename Scalar>
  void ewiseMult( fnCallDesc& d, Scalar multiplicand, Vector<Scalar>& A, Vector<Scalar>& C) {
    Index i;
    C.num = A.num;
    C.rowind.resize(C.num);
    C.val.resize(C.num);
    for( i=0; i<A.num; i++ ) {
      C.rowind[i] = A.rowind[i];
      C.val[i] = A.val[i]*multiplicand;
    }
  }

  // This is overloaded ewiseAdd C=A+B. A+u not written yet. (It also seems redundant if we already have C=A*u?)
  // Standard merge algorithm
  // Checks d.Assign to see whether we are doing C += or C =
  // When B is empty, we do C+=A
  // TODO: C+=A+B
  //       C =A*u
  template<typename Scalar>
  void ewiseAdd( fnCallDesc& d, Vector<Scalar>& A, Vector<Scalar>& B, Vector<Scalar>& C ) {
    Index i = 0;
    Index j = 0;
    if( d.getAssign()==assignDesc_st ) {
      C.num = 0;
      C.rowind.clear();
      C.val.clear();
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
      }
      } while( i<A.num ) {
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
    } else if( d.getAssign()==assignDesc_stOp && B.num==0) {
      Vector<Scalar> D;
      D.num = 0;
      D.rowind.clear();
      D.val.clear();
      while( i<A.num && j<C.num ) {
        if( A.rowind[i] == C.rowind[j] ) {
          D.val.push_back(A.val[i] + C.val[j]);
          D.rowind.push_back(A.rowind[i]);
          D.num++;
          i++;
          j++;
        } else if( A.rowind[i] < C.rowind[j] ) {
          D.val.push_back(A.val[i]);
          D.rowind.push_back(A.rowind[i]);
          D.num++;
          i++;
        } else {
        D.val.push_back(C.val[j]);
        D.rowind.push_back(C.rowind[j]);
        D.num++;
        j++;
      }
      } while( i<A.num ) {
        D.val.push_back(A.val[i]);
        D.rowind.push_back(A.rowind[i]);
        D.num++;
        i++;
      } while( j<C.num ) {
        D.val.push_back(C.val[j]);
        D.rowind.push_back(C.rowind[j]);
        D.num++;
        j++;
      }
      C.num = D.num;
      C.rowind = D.rowind;
      C.val = D.val;
    }
  }

  // Could also have template where matrices A and B have different values as Manoj/Jose originally had in their signature, but sake of simplicity assume they have same ScalarType. Also omitted optional mask m for   sake of simplicity.
  // Also omitting safety check that sizes of A and B s.t. they can be multiplied
  template<typename Scalar>
  void mXm(fnCallDesc& d, Matrix<Scalar>& A, Matrix<Scalar>& B, Matrix<Scalar>& C) {
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
    d.setAssign(assignDesc_stOp);
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
            GraphBLAS::ewiseAdd( d, temp, empty, result );
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
    if( old_assign == assignDesc_st )
      d.setAssign(assignDesc_stOp);
  }
}

int main() {

  GraphBLAS::Tuple<int> tuple;
  tuple.I   = {0, 1, 2};
  tuple.J   = {0, 1, 2};
  tuple.val = {1, 1, 1};
  
  GraphBLAS::Matrix<int> A;
  GraphBLAS::Matrix<int> B;
  GraphBLAS::Matrix<int> C;
  GraphBLAS::buildMatrix<int>(3, 3, tuple, A);
  GraphBLAS::buildMatrix<int>(3, 3, tuple, B);

  GraphBLAS::fnCallDesc d;
  GraphBLAS::mXm<int>(d, A, B, C);
  GraphBLAS::extractTuples<int>(C, tuple);

  for( int i=0; i<tuple.I.size(); i++ )
    std::cout << tuple.I[i] << std::endl;
  for( int i=0; i<tuple.J.size(); i++ )
    std::cout << tuple.J[i] << std::endl;
  for( int i=0; i<tuple.val.size(); i++ )
    std::cout << tuple.val[i] << std::endl;

  return 0;
}
