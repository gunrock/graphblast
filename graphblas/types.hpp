#ifndef GRB_TYPES_HPP
#define GRB_TYPES_HPP

#define GrB_NULL NULL

#include <cstddef>
#include <cstdint>

namespace graphblas
{
  typedef int           Index;
  typedef float         T;

  class Semiring{};

  enum Storage {GrB_UNKNOWN,
                GrB_SPARSE,
                GrB_DENSE};

  enum Major {GrB_ROWMAJOR,
              GrB_COLMAJOR};

  enum Info {GrB_SUCCESS,
             GrB_UNINITIALIZED_OBJECT, // API errors
             GrB_NULL_POINTER,
             GrB_INVALID_VALUE,
             GrB_INVALID_INDEX,
             GrB_DOMAIN_MISMATCH,
             GrB_DIMENSION_MISMATCH,
             GrB_OUTPUT_NOT_EMPTY,
             GrB_NO_VALUE,
             GrB_OUT_OF_MEMORY,        // Execution errors
             GrB_INSUFFICIENT_SPACE,
             GrB_INVALID_OBJECT,
             GrB_INDEX_OUT_OF_BOUNDS,
             GrB_PANIC};

  enum Desc_field {GrB_OUTP,
                   GrB_MASK,
                   GrB_INP0,
                   GrB_INP1,
                   GrB_MODE, 
                   GrB_TA, 
                   GrB_TB, 
                   GrB_NT};

  enum Desc_value {GrB_SCMP,             // for GrB_OUTP, GrB_MASK, GrB_INP0,
                   GrB_TRAN,             //     GrB_INP1
                   GrB_REPLACE,
                   GrB_DEFAULT,
                   GrB_CUSPARSE,         // for SpMV, SpMM
                   GrB_CUSPARSE2, 
                   GrB_FIXEDROW,
                   GrB_FIXEDCOL,
                   GrB_MERGEPATH =   9,
                   GrB_8         =   8,  // for GrB_TA, GrB_TB, GrB_NT
                   GrB_16        =  16,
                   GrB_32        =  32,
                   GrB_64        =  64,
                   GrB_128       = 128,
                   GrB_256       = 256,
                   GrB_512       = 512,
                   GrB_1024      =1024};

  enum Operator {GrB_IDENTITY,           // for UnaryOp
                 GrB_AINV,
                 GrB_MINV,
                 GrB_LNOT,
                 GrB_LOR,                // for BinaryOp
                 GrB_LAND,
                 GrB_LXOR,
                 GrB_EQ,
                 GrB_NE,
                 GrB_GT,
                 GrB_LT,
                 GrB_GE,
                 GrB_LE,
                 GrB_FIRST,
                 GrB_SECOND,
                 GrB_MIN,
                 GrB_MAX,
                 GrB_PLUS,
                 GrB_MINUS,
                 GrB_TIMES,
                 GrB_DIV};
}

#endif  // GRB_TYPES_HPP
