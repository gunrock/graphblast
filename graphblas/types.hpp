#ifndef GRB_TYPES_HPP

#define GRB_TYPES_HPP

#include <cstddef>
#include <cstdint>

namespace graphblas
{
  typedef int Index;
  typedef int Info;

  class Descriptor{};
  class BinaryOp{};
  class Semiring{};

  static const uint8_t GrB_SUCCESS             = 0;
  static const uint8_t GrB_INDEX_OUT_OF_BOUNDS = 1;
  static const uint8_t GrB_DIMENSION_MISMATCH  = 2;
  static const uint8_t GrB_UNINITIALIZED_OBJECT= 3;

  enum Storage {Unknown,Sparse,Dense};
}

#endif  // GRB_TYPES_HPP
