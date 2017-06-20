#ifndef GRB_BACKEND_APSPIE_DESCRIPTOR_HPP
#define GRB_BACKEND_APSPIE_DESCRIPTOR_HPP

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
    Desc_value desc[4];
  };

  Info Descriptor::set( Desc_field& field, Desc_value& value )
  {
    desc[field] = value;
    return GrB_SUCCESS;
  }

  Info Descriptor::get( Desc_field& field, Desc_value& value ) const
  {
    value = desc[field];
    return GrB_SUCCESS;
  }

}  // backend
}  // graphblas

#endif  // GRB_DESCRIPTOR_HPP
