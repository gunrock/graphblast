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
  template<typename T>
  class Matrix;

  class Descriptor
  {
    public:
    // Default Constructor, Standard Constructor (Replaces new in C++)
    //   -it's imperative to call constructor using descriptor or else the 
    //     constructed object won't be tied to this outermost layer
    Descriptor() : descriptor_() {}

    // Default Destructor is good enough for this layer
    ~Descriptor() {}

    // C API Methods
    Info set( Desc_field field, Desc_value value );
    Info set( Desc_field field, int value );
    Info get( Desc_field field, Desc_value* value ) const;

    // Useful methods
    Info toggle( Desc_field field );
    Info loadArgs( const po::variables_map& vm );

    private:
    // Data members that are same for all backends
    backend::Descriptor descriptor_;
  };

  Info Descriptor::set( Desc_field field, Desc_value value )
  {
    return descriptor_.set( field, value );
  }

  Info Descriptor::set( Desc_field field, int value )
  {
    return descriptor_.set( field, static_cast<Desc_value>(value) );
  }

  Info Descriptor::get( Desc_field field, Desc_value* value ) const
  {
    if( value==NULL ) return GrB_NULL_POINTER;
    return descriptor_.get( field, value );
  }

  Info Descriptor::toggle( Desc_field field )
  {
    return descriptor_.toggle( field );
  }

  Info Descriptor::loadArgs( const po::variables_map& vm )
  {
    return descriptor_.loadArgs( vm );
  }

}  // graphblas

#endif  // GRB_DESCRIPTOR_HPP
