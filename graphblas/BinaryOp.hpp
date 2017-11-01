#ifndef GRB_BINARYOP_HPP
#define GRB_BINARYOP_HPP

#include <vector>
#include <functional>

#include "graphblas/types.hpp"

// Opaque data members from the right backend
#define __GRB_BACKEND_BINARYOP_HEADER <graphblas/backend/__GRB_BACKEND_ROOT/BinaryOp.hpp>
#include __GRB_BACKEND_BINARYOP_HEADER
#undef __GRB_BACKEND_BINARYOP_HEADER

namespace graphblas
{
  template<typename T_out=bool, typename T_in1=T_out, typename T_in2=T_out>
  class BinaryOp
  {
    public:
    // Default Constructor, Standard Constructor (Replaces new in C++)
    //   -it's imperative to call constructor using descriptor or else the 
    //     constructed object won't be tied to this outermost layer
    BinaryOp() : binary_op_(std::plus<T_out>()) {}
    template <typename Op>
    BinaryOp( Op* op ) : binary_op_(op) {}

    // Default Destructor is good enough for this layer
    ~BinaryOp() {}

    // C API Methods
    template <typename Op>
    Info nnew( Op* op );

    private:
    // Data members that are same for all backends
    backend::BinaryOp<T_out, T_in1, T_in2> binary_op_;

  };

  template <typename T_out, typename T_in1, typename T_in2>
  template <typename Op>
  Info BinaryOp<T_out,T_in1,T_in2>::nnew( Op* op )
  {
    return binary_op_.nnew( op );
  }

}  // graphblas

#endif  // GRB_BINARYOP_HPP
