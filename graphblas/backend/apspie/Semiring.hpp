#ifndef GRB_BACKEND_APSPIE_SEMIRING_HPP
#define GRB_BACKEND_APSPIE_SEMIRING_HPP

#include <vector>

#include "graphblas/stddef.hpp"
#include "graphblas/backend/apspie/apspie.hpp"

namespace graphblas
{
namespace backend
{
  template<typename T_in1, typename T_in2, typename T_out>
  class Semiring
  {
    public:
    Semiring() : add_(std::plus<T>()), mul_(std::multiplies<T>()), 
                 identity_(0) {}
    Semiring( Monoid<T_out> add_t, BinaryOp<T_in1,T_in2,T_out> mul_t )
               : add_(add_t), mul_(mul_t), identity_(add_t.identity()) {}

    // Default Destructor
    ~Semiring() {}

    // C API Methods
    Info nnew( Monoid<T_out> add_t, BinaryOp<T_in1,T_in2,T_out> mul_t );

    inline __host__ __device__ T_out identity() const
    {
      return identity_;
    }

    inline __host__ __device__ T_out add( T_out lhs, T_out rhs ) const
    {
      return add_(lhs,rhs);
    }

    inline __host__ __device__ T_out mul( T_in1 lhs, T_in2 rhs ) const
    {
      return mul_(lhs,rhs);
    }

    private:
    nvstd::function<T_out(T_out,T_out)> add_;
    nvstd::function<T_out(T_in1,T_in2)> mul_;
    T_out                               identity_;
  };

  template <typename T_in1, typename T_in2, typename T_out>
  Info Semiring<T_in1,T_in2,T_out>::nnew( Monoid<T_out>               add_t,
                                          BinaryOp<T_in1,T_in2,T_out> mul_t ) 
  {
    add_ = add_t;
    mul_ = mul_t;
    identity_ = add_t.identity();
    return GrB_SUCCESS;
  }

}  // backend
}  // graphblas

#endif  // GRB_BACKEND_APSPIE_SEMIRING_HPP
