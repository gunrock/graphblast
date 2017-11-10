#ifndef GRB_SEMIRING_HPP
#define GRB_SEMIRING_HPP

#include <vector>
#include <functional>

#include "graphblas/types.hpp"

// Opaque data members from the right backend
#define __GRB_BACKEND_SEMIRING_HEADER <graphblas/backend/__GRB_BACKEND_ROOT/Semiring.hpp>
#include __GRB_BACKEND_SEMIRING_HEADER
#undef __GRB_BACKEND_SEMIRING_HEADER

namespace graphblas
{
  template<typename T_in1=bool, typename T_in2=T_in1, typename T_out=T_in1>
  class Semiring
  {
    public:
    // Default Constructor, Standard Constructor (Replaces new in C++)
    //   -it's imperative to call constructor using descriptor or else the 
    //     constructed object won't be tied to this outermost layer
    Semiring() : op_() {}
    template <typename MulOp, typename AddOp>
    Semiring( MulOp mul, AddOp add ) : op_(mul, add) {}

    // Default Destructor is good enough for this layer
    ~Semiring() {}

    // C API Methods
    template <typename MulOp, typename AddOp>
    Info nnew( MulOp mul, AddOp add );

    T_out identity() const
    {
      return op_.identity();
    }

    T_out add( T_out lhs, T_out rhs ) const
    {
      return op_.add(lhs,rhs);
    }

    T_out mul( T_in1 lhs, T_in2 rhs ) const
    {
      return op_.mul(lhs,rhs);
    }

    private:
    // Data members that are same for all backends
    backend::Semiring<T_in1, T_in2, T_out> op_;

  };

  template <typename T_in1, typename T_in2, typename T_out>
  template <typename MulOp, typename AddOp>
  Info Semiring<T_in1,T_in2,T_out>::nnew( MulOp mul, AddOp add )
  {
    return op_.nnew( mul, add );
  }

}  // graphblas

#endif  // GRB_SEMIRING_HPP
