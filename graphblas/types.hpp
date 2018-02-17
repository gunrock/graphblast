#ifndef GRB_TYPES_HPP
#define GRB_TYPES_HPP

#define GrB_MEMORY 0

#include <cstddef>
//#include <cstdint>

namespace graphblas
{
  typedef int Index;
  typedef int BinaryOp;

  class Semiring{};
  //template<typename T> class Matrix;

  static int GrB_NULL = 0;

  enum Storage {GrB_UNKNOWN,
                GrB_SPARSE,
                GrB_DENSE};

  enum Major {GrB_ROWMAJOR,
              GrB_COLMAJOR};

  enum Info {GrB_SUCCESS,
             GrB_OUT_OF_MEMORY,
             GrB_INDEX_OUT_OF_BOUNDS,
             GrB_PANIC,
             GrB_UNINITIALIZED_OBJECT,  
             GrB_DIMENSION_MISMATCH};

  enum Desc_field {GrB_MODE, 
                   GrB_TA, 
                   GrB_TB, 
                   GrB_NT};

  enum Desc_value {GrB_CUSPARSE,
                   GrB_CUSPARSE2,
                   GrB_FIXEDROW,
                   GrB_FIXEDROW2,
                   GrB_FIXEDROW3,
                   GrB_FIXEDROW4,
                   GrB_FIXEDCOL,
                   GrB_MERGEPATH,
                   GrB_8   =  8,
                   GrB_16  = 16,
                   GrB_32  = 32,
                   GrB_64  = 64,
                   GrB_128 =128,
                   GrB_256 =256,
                   GrB_512 =512,
                   GrB_1024=1024};
}

#endif  // GRB_TYPES_HPP
