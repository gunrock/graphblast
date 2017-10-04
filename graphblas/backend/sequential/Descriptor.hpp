#ifndef GRB_BACKEND_SEQUENTIAL_DESCRIPTOR_HPP
#define GRB_BACKEND_SEQUENTIAL_DESCRIPTOR_HPP

#include <vector>

namespace graphblas
{
namespace backend
{
  class Descriptor
  {
    public:
    // Default Constructor, Standard Constructor (Replaces new in C++)
    //   -it's imperative to call constructor using matrix or the constructed object
    //     won't be tied to this outermost layer
    Descriptor() : desc{ GrB_FIXEDROW, GrB_32, GrB_32, GrB_128 } {}

    // Assignment Constructor
    // TODO:
    void operator=( Descriptor& rhs );

    // Destructor
    ~Descriptor() {};

    // C API Methods
    //
    // Mutators
    Info set( const Desc_field field, Desc_value value );

    // Accessors
    Info get( const Desc_field field, Desc_value& value ) const;

    private:
    // Data members that are same for all backends
    Desc_value desc[4];
  };

  Info Descriptor::set( const Desc_field field, Desc_value value )
  {
    desc[field] = value;
    return GrB_SUCCESS;
  }

  Info Descriptor::get( const Desc_field field, Desc_value& value ) const
  {
    value = desc[field];
    return GrB_SUCCESS;
  }

}  // backend
}  // graphblas

#endif  // GRB_BACKEND_SEQUENTIAL_DESCRIPTOR_HPP
