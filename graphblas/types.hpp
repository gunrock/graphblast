#ifndef GRB_TYPES_HPP

#define GRB_TYPES_HPP

#include <cstddef>
#include <cstdint>

namespace graphblas
{
  typedef int Index;

  class BinaryOp{};
  class Semiring{};

  enum Storage {GrB_UNKNOWN,
                GrB_SPARSE,
                GrB_DENSE};
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
                   GrB_FIXEDCOL,
                   GrB_MERGEPATH};
}

#endif  // GRB_TYPES_HPP
