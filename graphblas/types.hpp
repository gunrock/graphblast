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

  static const uint8_t GrB_SUCCESS       = 0;
  static const uint8_t GrB_INDEX_OUT_OF_BOUNDS = 1;
}

#endif  // GRB_TYPES_HPP
