#include <unistd.h>
#include <ctype.h>
#include <stdint.h>

namespace GraphBLAS
{
  template<typename T>
  class Matrix
  {
  };

  template<typename T>
  class Vector
  {
  };
  
  // Input argument preprocessing functions.
  enum Transform
  {
    argDesc_null =	0,
    argDesc_neg  =	1,
    argDesc_T 	 =	2,
    argDesc_negT =	3,
    argDesc_notT =	4
  };

// Next is the relevant number of assignment operators.  Since Boolean data is
// of significant interest, I have added the stAnd and stOr ops for now
  enum Assign
  {
    assignDesc_st   =	0,	/* Simple assignment */
    assignDesc_stOp =	1	/* Store with Circle plus */
  };

// List of ops that can be used in map/reduce operations.
  enum BinaryOp
  {
    fieldOps_mult =	0,
    fieldOps_add  =	1,
    fieldOps_and  =	2,
    fieldOps_or	  =	3
  };

// List of additive identities that can be used
  enum AdditiveId
  {
    addZero_add =       0,
    addZero_min =       1,
    addZero_max =       2
  };

  class fnCallDesc
  {
	Assign assignDesc ;
	Transform arg1Desc ;
	Transform arg2Desc ;
	Transform maskDesc ;
	int32_t dim ;			// dimension for reduction operation on matrices
	BinaryOp addOp ;
	BinaryOp multOp ;
        AdditiveId addZero ;
  };

  void mxm(fnCallDesc& d, Matrix<int>& C, Matrix<bool>& A, Matrix<int>& B, Vector<bool>& m);
}
