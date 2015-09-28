#include <unistd.h>
#include <ctype.h>
#include <stdint.h>

typedef void* GB_Matrix;
typedef void* GB_Vector;

// Input argument preprocessing functions.
#define GB_argDesc_null		0
#define GB_argDesc_neg 		1
#define GB_argDesc_T 		2
#define GB_argDesc_negT 	3
#define GB_argDesc_notT 	4

// Next is the relevant number of assignment operators.  
#define GB_assignDesc_st	0	/* Simple assignment */
#define GB_assignDesc_stOp	1	/* Store with Circle plus of reduce operator */
//  ....

// List of ops that can be used in map/reduce operations.
#define GB_fieldOps_mul	0
#define GB_fieldOps_add	1
#define GB_fieldOps_and	2
#define GB_fieldOps_or	3

typedef struct {
	int64_t assignDesc ;
	int64_t arg1Desc ;
	int64_t arg2Desc ;
	int64_t maskDesc ;
	int32_t dim ;			// dimension for reduction operation on matrices
	int32_t mapOp ;
	int32_t reduceOp ;
} GB_fnCallDesc ;

// We can provide an init function that sets these default values
GB_fnCallDesc desc = {
	.assignDesc = GB_assignDesc_st ,
	.arg1Desc = GB_argDesc_null ,
	.arg2Desc = GB_argDesc_null ,
	.maskDesc = GB_argDesc_null ,
	.dim = 0 ,			
	.mapOp = GB_fieldOps_mul ,
	.reduceOp = GB_fieldOps_add
} ;


void GB_mxm (GB_fnCallDesc *desc, GB_Matrix C, GB_Matrix A, GB_Matrix B, GB_Vector M);


