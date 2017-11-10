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
  template<typename T_in1=bool, typename T_in2=T_in1, typename T_out=T_in1>
  class BinaryOp
  {
    public:
    // Default Constructor, Standard Constructor (Replaces new in C++)
    //   -it's imperative to call constructor using descriptor or else the 
    //     constructed object won't be tied to this outermost layer
    BinaryOp() : op_(std::plus<T_in1>()) {}
    template <typename Op>
    BinaryOp( Op op ) : op_(op) {}

    // Default Destructor is good enough for this layer
    ~BinaryOp() {}

    // C API Methods
    template <typename Op>
    Info nnew( Op op );

    T_out operator()( T_in1 lhs, T_in2 rhs ) const
    {
      return op_.operator()(lhs,rhs);
    }

    private:
    // Data members that are same for all backends
    backend::BinaryOp<T_in1, T_in2, T_out> op_;

  };

  template <typename T_in1, typename T_in2, typename T_out>
  template <typename Op>
  Info BinaryOp<T_in1,T_in2,T_out>::nnew( Op op )
  {
    return op_.nnew( op );
  }

}  // graphblas

#endif  // GRB_BINARYOP_HPP
