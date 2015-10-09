// Compiles using:
//   g++ -o graphblas -std=c++11 graphblas.cpp
// Functional:
//   -BuildMatrix (builds matrix in CSC format)
//   -ExtractTuples
// Incomplete:
//   -MxM (still needs EwiseMult and EwiseAdd, or versions of both that do same thing but are customized for mxm)

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
      std::vector<T>     V;
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
      {}
      fnCallDesc( const std::string& semiring = "Matrix Multiply Assign" ):
        assignDesc(assignDesc_st),
        arg1Desc(argDesc_null),
        arg2Desc(argDesc_null),
        maskDesc(argDesc_null),
        dim(1),
        addOp(fieldOps_add),
        multOp(fieldOps_mult),
        addId(addId_add)
      {}
      assignDesc getAssign() const { 
        return assignDesc };
      void setAssign(Assign state) { 
        state = argDesc_stOp };
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
      C.val[dest] = A.V[i];                 // Store value
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
    C.V.resize(A.val.size());
    for( i=0; i<A.val.size(); i++ ) {
      C.I[i] = A.rowind[i];                // Copy from Tuple
      C.V[i] = A.val[i];                   // Copy from Tuple
    }
    for( i=0; i<A.colptr.size()-1; i++ ) {  // Get j-coordinate from colptr
      to_increment = A.colptr[i+1]-A.colptr[i];
      for( to_increment; to_increment>0; to_increment-- ) {      
        C.J[i] = i;
  }}}

  // This is overloaded ewiseMult A*u. A*B not written yet.
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

  // This is overloaded ewiseAdd A+B. A+u not written yet. (It also seems redundant if we already have A*u?)
  // Standard mergesort merge.
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
  }

  // Could also have template where matrices A and B have different values as Manoj/Jose originally had in their signature, but sake of simplicity assume they have same ScalarType. Also omitted optional mask m for   sake of simplicity.
  // Also omitting safety check that sizes of A and B s.t. they can be multiplied
  template<typename Scalar>
  void mXm(fnCallDesc& d, Matrix<Scalar>& C, Matrix<Scalar>& A, Matrix<Scalar>& B) {
    Index i, j, k;
    Index N = B.colptr.size()-1;
    Index Acol, Bcol;
    Scalar value;
    Vector<Scalar> temp;
    Vector<Scalar> result;
    Index count = 0;
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
          if( Acol > 0 ) {
            //TODO: implement ewiseMult, store result into temp
            //GraphBLAS::ewiseMult( d, value, A, A.colptr[j], A.colptr[j+1], temp );
            temp.num = Acol;
            temp.rowind.resize(Acol);
            temp.val.resize(Acol);
            count = 0;
            temp.rowind[count] = A.rowind[j];
            temp.val[count] = A.val[j]*value;
            d.setAssign(1);
            //GraphBLAS::ewiseAdd( d, temp, result );
            count++;                                       // count is placeholder for if statement
          }
        }
        //TODO: write result into C and advance colptr;
        //GraphBLAS::ewiseAdd( result, C );
  }}}
}

int main() {

  GraphBLAS::Tuple<int> tuple1;
  GraphBLAS::Tuple<int> tuple2;
  tuple1.I = {0, 1, 2};
  tuple1.J = {0, 1, 2};
  tuple1.V = {1, 1, 1};
  
  GraphBLAS::Matrix<int> A;
  GraphBLAS::Matrix<int> B;
  GraphBLAS::Matrix<int> C;
  GraphBLAS::buildMatrix<int>(3, 3, tuple1, A);
  GraphBLAS::buildMatrix<int>(3, 3, tuple1, B);

  GraphBLAS::fnCallDesc d("Matrix Multiply");
  //GraphBLAS::mXm<int>(d, C, A, B);
  GraphBLAS::extractTuples<int>(B, tuple2);

  GraphBLAS::Vector<int> V;
  GraphBLAS::Vector<int> W;
  GraphBLAS::Vector<int> X;
  V.num = 3;
  V.rowind = {1, 3, 5};
  V.val = {1, 2, 3};
  W.num = 3;
  W.rowind = {2, 3, 6};
  W.val = {1, 2, 3};
  GraphBLAS::ewiseAdd<int> ( d, V, W, X );
  GraphBLAS::ewiseMult<int>( d, 2, V, W );

  for( int i=0; i<A.colptr.size(); i++ )
    std::cout << A.colptr[i] << std::endl;
  for( int i=0; i<A.rowind.size(); i++ )
    std::cout << A.rowind[i] << std::endl;
  for( int i=0; i<tuple2.I.size(); i++ )
    std::cout << tuple2.I[i] << std::endl;
  for( int i=0; i<tuple2.J.size(); i++ )
    std::cout << tuple2.J[i] << std::endl;
  for( int i=0; i<X.rowind.size(); i++ )
    std::cout << X.rowind[i] << std::endl;
  for( int i=0; i<X.val.size(); i++ )
    std::cout << X.val[i] << std::endl;

  return 0;
}
