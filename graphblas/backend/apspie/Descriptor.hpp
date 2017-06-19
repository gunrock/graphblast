#ifndef GRB_BACKEND_APSPIE_DESCRIPTOR_HPP
#define GRB_BACKEND_APSPIE_DESCRIPTOR_HPP

#include <vector>

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
    GrB_MODE mode;
    GrB_TA   ta;
    GrB_TB   tb;
    GrB_NT   nt;
  };

  Info Descriptor::set( Desc_field& field, Desc_value& value )
  {
    return GrB_SUCCESS;
  }

  Info Descriptor::get( Desc_field& field, Desc_value& value ) const
  {
    
    return GrB_SUCCESS;
  }

}  // graphblas

#endif  // GRB_DESCRIPTOR_HPP
