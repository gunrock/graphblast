#ifndef GRB_UNARYOP_HPP
#define GRB_UNARYOP_HPP

#include <vector>
#include <functional>

#include "graphblas/types.hpp"

// Opaque data members from the right backend
#define __GRB_BACKEND_UNARYOP_HEADER <graphblas/backend/__GRB_BACKEND_ROOT/UnaryOp.hpp>
#include __GRB_BACKEND_UNARYOP_HEADER
#undef __GRB_BACKEND_UNARYOP_HEADER

namespace graphblas
{
  template<typename T_in=bool, typename T_out=T_in>
  class UnaryOp
  {
    public:
    // Default Constructor, Standard Constructor (Replaces new in C++)
    //   -it's imperative to call constructor using descriptor or else the 
    //     constructed object won't be tied to this outermost layer
    UnaryOp() : op_() {}
    template <typename Op>
    UnaryOp( Op op ) : op_(op) {}

    // Default Destructor is good enough for this layer
    ~UnaryOp() {}

    // C API Methods
    template <typename Op>
    Info nnew( Op op );

    T_out operator()( T_in rhs ) const
    {
      return op_.operator()(rhs);
    }

    private:
    // Data members that are same for all backends
    backend::UnaryOp<T_in, T_out> op_;

  };

  template <typename T_in, typename T_out>
  template <typename Op>
  Info UnaryOp<T_in,T_out>::nnew( Op op )
  {
    return op_.nnew( op );
  }

}  // graphblas

#endif  // GRB_UNARYOP_HPP
