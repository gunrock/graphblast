#ifndef GRB_DESCRIPTOR_HPP
#define GRB_DESCRIPTOR_HPP

#include <vector>

#include "graphblas/types.hpp"

// Opaque data members from the right backend
#define __GRB_BACKEND_DESCRIPTOR_HEADER <graphblas/backend/__GRB_BACKEND_ROOT/Descriptor.hpp>
#include __GRB_BACKEND_DESCRIPTOR_HEADER
#undef __GRB_BACKEND_DESCRIPTOR_HEADER

namespace graphblas
{
  class Descriptor
  {
    public:
    // Default Constructor, Standard Constructor (Replaces new in C++)
    //   -it's imperative to call constructor using matrix or the constructed object
    //     won't be tied to this outermost layer
    Descriptor();

    // Assignment Constructor
    // TODO:
    void operator=( Descriptor& rhs );

    // Destructor
    ~Descriptor() {};

    // C API Methods
    //
    // Mutators

    private:
    // Data members that are same for all backends
    backend::Descriptor desc;
  };

  Info Descriptor::set( Descriptor& desc_, Desc_value& desc_value )
  {
    return desc.set( desc_, desc_value );
  }

  Info Descriptor::get( Descriptor& desc_, Desc_value& desc_value ) const
  {
    return desc.get( desc_, desc_value );
  }

}  // graphblas

#endif  // GRB_DESCRIPTOR_HPP
