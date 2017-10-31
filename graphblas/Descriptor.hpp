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

    private:
    // Data members that are same for all backends
    backend::Descriptor descriptor_;

    /*template <typename c, typename m, typename a, typename b>
    friend Info mxm( Matrix<c>&        C,
                     const Matrix<m>&  mask,
                     const BinaryOp&   accum,
                     const Semiring&   op,
                     const Matrix<a>&  A,
                     const Matrix<b>&  B,
                     const Descriptor& desc );*/

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

}  // graphblas

#endif  // GRB_DESCRIPTOR_HPP
