#ifndef GRB_BACKEND_APSPIE_DESCRIPTOR_HPP
#define GRB_BACKEND_APSPIE_DESCRIPTOR_HPP

#include <vector>

#include "graphblas/backend/apspie/apspie.hpp"

namespace graphblas
{
namespace backend
{
  class Descriptor
  {
    public:
    Descriptor() : desc{ GrB_DEFAULT, GrB_DEFAULT, GrB_DEFAULT, GrB_DEFAULT, 
                         GrB_FIXEDROW, GrB_32, GrB_32, GrB_128 },
                   h_buffer_(NULL), h_size_(0), d_buffer_(NULL), d_size_(0) {}

    // Default Destructor
    ~Descriptor();

    // C API Methods
    Info set( Desc_field field, Desc_value  value );
    Info get( Desc_field field, Desc_value* value ) const;

    private:
    Desc_value desc[8];
    void*      h_buffer_;
    size_t     h_size_;
    void*      d_buffer_;
    size_t     d_size_;
  };

  Descriptor::~Descriptor()
  {
    if( h_buffer_!=NULL ) free(h_buffer_);
    if( d_buffer_!=NULL ) CUDA( cudaFree(d_buffer_) );
  }

  Info Descriptor::set( Desc_field field, Desc_value value )
  {
    desc[field] = value;
    return GrB_SUCCESS;
  }

  Info Descriptor::get( Desc_field field, Desc_value* value ) const
  {
    *value = desc[field];
    return GrB_SUCCESS;
  }

}  // backend
}  // graphblas

#endif  // GRB_DESCRIPTOR_HPP
