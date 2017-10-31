#ifndef GRB_STDDEF_HPP
#define GRB_STDDEF_HPP

#include <cstddef>
#include <cstdint>

namespace graphblas
{

  BinaryOp* GrB_PLUS_FP32 = new BinaryOp<float>( std::plus() );
  BinaryOp* GrB_
  enum Operator {GrB_IDENTITY,           // for UnaryOp
                 GrB_AINV,
                 GrB_MINV,
                 GrB_LNOT,
                 GrB LOR,                // for BinaryOp
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

#endif  // GRB_STDDEF_HPP
