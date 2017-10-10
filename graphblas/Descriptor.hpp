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
    //   -it's imperative to call constructor using descriptor_ or else the 
    //     constructed object won't be tied to this outermost layer
    Descriptor() : descriptor_() {}

    // Assignment Constructor
    // TODO:
    void operator=( Descriptor& rhs );

    // Destructor
    ~Descriptor() {};

    // C API Methods
    //
    // Mutators
    Info set( const Desc_field field, Desc_value value );
    Info set( const Desc_field field, int value );

    // Accessors
    Info get( const Desc_field field, Desc_value& value ) const;

    private:
    // Data members that are same for all backends
    backend::Descriptor descriptor_;

    template <typename c, typename m, typename a, typename b>
    friend Info mxm( Matrix<c>&        C,
                     const Matrix<m>&  mask,
                     const BinaryOp&   accum,
                     const Semiring&   op,
                     const Matrix<a>&  A,
                     const Matrix<b>&  B,
                     const Descriptor& desc );

    template <typename c, typename a, typename b>
    friend Info mxm( Matrix<c>&        C,
                     const int         mask,
                     const int         accum,
                     const Semiring&   op,
                     const Matrix<a>&  A,
                     const Matrix<b>&  B,
                     const Descriptor& desc );

    template <typename c, typename a, typename b>
    friend Info mxv( Matrix<c>&        C,
                     const int         mask,
                     const int         accum,
                     const Semiring&   op,
                     const Matrix<a>&  A,
                     const Matrix<b>&  B,
                     const Descriptor& desc );
  };

  Info Descriptor::set( const Desc_field field, Desc_value value )
  {
    return descriptor_.set( field, value );
  }

  Info Descriptor::set( const Desc_field field, int value )
  {
    return descriptor_.set( field, static_cast<Desc_value>(value) );
  }

  Info Descriptor::get( const Desc_field field, Desc_value& value ) const
  {
    return descriptor_.get( field, value );
  }

}  // graphblas

#endif  // GRB_DESCRIPTOR_HPP
