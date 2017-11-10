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
    Monoid() : op_(std::plus<T>(), 0) {}
    template <typename Op>
    Monoid( Op binary_op, T identity_t ) : op_(binary_op, identity_t) {}

    // Default Destructor is good enough for this layer
    ~Monoid() {}

    // C API Methods
    template <typename Op>
    Info nnew( Op binary_op,
               T  identity_t );

    T identity() const
    {
      return op_.identity();
    }

    T operator()( T lhs, T rhs ) const
    {
      return op_.operator()(lhs,rhs);
    }

    private:
    // Data members that are same for all backends
    backend::Monoid<T> op_;

  };

  template <typename T>
  template <typename Op>
  Info Monoid<T>::nnew( Op binary_op, T identity_t )
  {
    return op_.nnew( binary_op, identity_t );
  }

}  // graphblas

#endif  // GRB_MONOID_HPP
