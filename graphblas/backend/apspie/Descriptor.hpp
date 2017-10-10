#ifndef GRB_BACKEND_APSPIE_DESCRIPTOR_HPP
#define GRB_BACKEND_APSPIE_DESCRIPTOR_HPP

#include <vector>

#include <moderngpu.cuh>

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
    Descriptor() : desc_{ GrB_FIXEDROW, GrB_32, GrB_32, GrB_128 },
      d_limits_(NULL), d_carryin_(NULL), d_carryout_(NULL),
      d_context_(mgpu::CreateCudaDevice(0)) {}

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
    Desc_value desc_[4];

    int*             d_limits_;
    float*           d_carryin_;
    float*           d_carryout_;
    mgpu::ContextPtr d_context_;
  };

  Info Descriptor::set( const Desc_field field, Desc_value value )
  {
    desc_[field] = value;
    return GrB_SUCCESS;
  }

  Info Descriptor::get( const Desc_field field, Desc_value& value ) const
  {
    value = desc_[field];
    return GrB_SUCCESS;
  }

}  // backend
}  // graphblas

#endif  // GRB_DESCRIPTOR_HPP
