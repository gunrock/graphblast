#ifndef GRB_BACKEND_APSPIE_MONOID_HPP
#define GRB_BACKEND_APSPIE_MONOID_HPP

#include <vector>

#include "graphblas/stddef.hpp"
#include "graphblas/backend/apspie/apspie.hpp"

namespace graphblas
{
namespace backend
{
  template<typename T>
  class Monoid
  {
    public:
    Monoid() : op_(graphblas::plus<T>()), identity_(0) {}
    Monoid( BinaryOp<T,T,T> binary_op, T identity_t ) 
             : op_(binary_op), identity_(identity_t) {}

    // Default Destructor
    ~Monoid() {}

    // C API Methods
    Info nnew( BinaryOp<T,T,T> binary_op, T identity_t );

    inline __host__ __device__ T identity() const
    {
      return identity_;
    }

    inline __host__ __device__ T operator()( T lhs, T rhs ) const
    {
      return op_(lhs,rhs);
    }

    private:
    nvstd::function<T(T,T)> op_;
    T                       identity_;
  };

  template <typename T>
  Info Monoid<T>::nnew( BinaryOp<T,T,T> binary_op, T identity_t )
  {
    op_       = binary_op;
    identity_ = identity_t;
    return GrB_SUCCESS;
  }

}  // backend
}  // graphblas

#endif  // GRB_BACKEND_APSPIE_MONOID_HPP
