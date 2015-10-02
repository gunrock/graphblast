// Compiles using:
//   g++ -o graphblas -std=c++11 graphblas.cpp
// Functional:
//   -buildmatrix (builds matrix in CSC format)
//   -extracttuples
// Incomplete:
//   -mxm (still needs ewisemult and ewiseadd, or versions of both that do same thing but are customized for mxm)

#include <unistd.h>
#include <ctype.h>
#include <stdint.h>
#include <vector>
#include <iostream>

namespace GraphBLAS
{

  typedef uint64_t Index;

  // Using transparent vector for simplicity (temporary)
  template<typename T>
  using Vector = std::vector<T>;

  // CSC format by default
  // Temporarily keeping member objects public to avoid having to use get
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
    addZero_add =       0,
    addZero_min =       1,
    addZero_max =       2
  };

  class fnCallDesc {
    Assign assignDesc ;
    Transform arg1Desc ;
    Transform arg2Desc ;
    Transform maskDesc ;
    int32_t dim ;			// dimension for reduction operation on matrices
    BinaryOp addOp ;
    BinaryOp multOp ;
    AdditiveId addZero ;
  };

  // Ignoring + operator, because it is optional argument per GraphBlas_Vops.pdf
  // Store as CSC by default
  // Don't have to assume tuple is ordered
  // Can be used for CSR by swapping I and J vectors in tuple A and swapping N and M dimensions
  template<typename Scalar>
  void buildmatrix(int M, int N, Tuple<Scalar>& A, Matrix<Scalar>& C) {
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
  void extracttuples(Matrix<Scalar>& A, Tuple<Scalar>& C) {
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

  template<typename Scalar>
  void ewisemult( fnCallDesc& d, Scalar multiplicand, Matrix<Scalar>& A, Index start, Index end, Vector<Scalar>& temp );

  template<typename Scalar>
  void ewiseadd( fnCallDesc& d, Vector<Scalar>& temp, Vector<Scalar>& A, Vector<Scalar>& C );

  // Could also have template where matrices A and B have different values as Manoj/Jose originally had in their signature, but sake of simplicity assume they have same ScalarType. Also omitted optional mask m for   sake of simplicity.
  // Also omitting safety check that sizes of A and B s.t. they can be multiplied
  // For simplicity, assume both are NxN square matrices
  template<typename Scalar>
  void mxm(fnCallDesc& d, Matrix<Scalar>& C, Matrix<Scalar>& A, Matrix<Scalar>& B) {
    Index i, j;
    Index N = B.colptr.size()-1;
    Index Acol, Bcol;
    Scalar value;
    int count = 0;
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
            //TODO: implement ewisemult, store result into temp
            //GraphBLAS::ewisemult( d, value, A, A.colptr[j], A.colptr[j+1], temp );  
            //GraphBLAS::ewiseadd( d, temp, result );
            count++;                                       // count is placeholder for if statement
          }
        }
        //TODO: write result into C and advance colptr;
        //GraphBLAS::ewiseadd( result, C );
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
  GraphBLAS::buildmatrix<int>(3, 3, tuple1, A);
  GraphBLAS::buildmatrix<int>(3, 3, tuple1, B);
  GraphBLAS::extracttuples<int>(A, tuple2);

  for( int i=0; i<A.colptr.size(); i++ )
    std::cout << A.colptr[i] << std::endl;
  for( int i=0; i<A.rowind.size(); i++ )
    std::cout << A.rowind[i] << std::endl;
  for( int i=0; i<tuple2.I.size(); i++ )
    std::cout << tuple2.I[i] << std::endl;
  for( int i=0; i<tuple2.J.size(); i++ )
    std::cout << tuple2.J[i] << std::endl;

  return 0;
}
