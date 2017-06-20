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
    Info set( Desc_field& field, Desc_value& value );

    // Accessors
    Info get( Desc_field& field, Desc_value& value ) const;

    private:
    // Data members that are same for all backends
    backend::Descriptor desc;
  };

  Info Descriptor::set( Desc_field& field, Desc_value& value )
  {
    return desc.set( field, value );
  }

  Info Descriptor::get( Desc_field& field, Desc_value& value ) const
  {
    return desc.get( field, value );
  }

}  // graphblas

#endif  // GRB_DESCRIPTOR_HPP
