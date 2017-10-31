#ifndef GRB_MONOID_HPP
#define GRB_MONOID_HPP

#include <vector>

#include "graphblas/types.hpp"

// Opaque data members from the right backend
#define __GRB_BACKEND_MONOID_HEADER <graphblas/backend/__GRB_BACKEND_ROOT/Monoid.hpp>
#include __GRB_BACKEND_MONOID_HEADER
#undef __GRB_BACKEND_MONOID_HEADER

namespace graphblas
{
  template<typename T>
  class Monoid
  {
    public:
    // Default Constructor, Standard Constructor (Replaces new in C++)
    //   -it's imperative to call constructor using descriptor or else the 
    //     constructed object won't be tied to this outermost layer
    Monoid() : monoid_(NULL, 0) {}
    Monoid( BinaryOp* binary_op, T identity ) : monoid_(binary_op, identity) {}

    // Default Destructor is good enough for this layer
    ~Monoid() {}

    // C API Methods
    Info nnew( BinaryOp* binary_op,
               T         identity );

    private:
    // Data members that are same for all backends
    backend::Monoid<T> monoid_;

  };

  template <typename T>
  Info Monoid::nnew( BinaryOp* binary_op, T identity )
  {
    if( binary_op==NULL ) return GrB_NULL_POINTER;
    return monoid_.nnew( binary_op, identity );
  }

}  // graphblas

#endif  // GRB_MONOID_HPP
