#ifndef GRB_TYPES_HPP
#define GRB_TYPES_HPP

#define GrB_NULL      NULL
#define GrB_ALL       NULL
// TODO: change GrB_MEMORY into commandline parameter
// (requires some trickery with templates and singleton idiom)
#define GrB_MEMORY    falsee  // print memory usage info
//#define GrB_NULL       nullptr
//#define GrB_NULL       static_cast<int*>(NULL)

// This controls up to how much space temporary buffers ought to hold
// Measured in multiples of |E|
#define GrB_THRESHOLD 1.0 //0.5

#include <cstddef>
#include <cstdint>

namespace graphblas
{
  typedef int           Index;
  typedef float         T;

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

  enum Desc_field {GrB_MASK,
                   GrB_OUTP,
                   GrB_INP0,
                   GrB_INP1,
                   GrB_MODE, 
                   GrB_TA, 
                   GrB_TB, 
                   GrB_NT,
                   GrB_MXVMODE,
                   GrB_SPMSPVMODE,
                   GrB_TOL,
                   GrB_NDESCFIELD};

  enum Desc_value {GrB_SCMP,             // for GrB_MASK
                   GrB_REPLACE,          // for GrB_OUTP
                   GrB_TRAN,             // for GrB_INP0, GrB_INP1
                   GrB_DEFAULT,
                   GrB_CUSPARSE,         // for SpMV, SpMM
                   GrB_CUSPARSE2, 
                   GrB_FIXEDROW,
                   GrB_FIXEDCOL,
                   GrB_MERGEPATH =   9,
                   GrB_PUSHPULL  =  10,  // for GrB_MXVMODE
                   GrB_PUSHONLY  =  11,  // for GrB_MXVMODE
                   GrB_PULLONLY  =  12,  // for GrB_MXVMODE
                   GrB_APSPIE    =  13,  // for GrB_SPMSPVMODE
                   GrB_APSPIELB  =  14,  // for GrB_SPMSPVMODE
                   GrB_GUNROCKLB =  15,  // for GrB_SPMSPVMODE
                   GrB_GUNROCKTWC=  17,  // for GrB_SPMSPVMODE
                   GrB_8         =   8,  // for GrB_TA, GrB_TB, GrB_NT
                   GrB_16        =  16,  // for GrB_TOL
                   GrB_32        =  32,
                   GrB_64        =  64,
                   GrB_128       = 128,
                   GrB_256       = 256,
                   GrB_512       = 512,
                   GrB_1024      =1024};

}

#endif  // GRB_TYPES_HPP
