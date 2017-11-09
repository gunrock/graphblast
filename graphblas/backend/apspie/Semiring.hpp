#ifndef GRB_BACKEND_APSPIE_SEMIRING_HPP
#define GRB_BACKEND_APSPIE_SEMIRING_HPP

#include <vector>

#include "graphblas/backend/apspie/apspie.hpp"

namespace graphblas
{
namespace backend
{
  template<typename T_in1, typename T_in2, typename T_out>
  class Semiring
  {
    public:
    Semiring() {}
    template <typename MulOp, typename AddOp>
    Semiring( MulOp mul_t, AddOp add_t ) : identity_(add_t.identity())
    {
      mul_ = mul_t;
      add_ = add_t;
    }

    // Default Destructor
    ~Semiring();

    // C API Methods
    template <typename MulOp, typename AddOp>
    Info nnew( MulOp mul_t, AddOp add_t );

    T_out identity() const
    {
      return identity_;
    }

    T_out add( T_out lhs, T_out rhs ) const
    {
      return add_(lhs,rhs);
    }

    T_out mul( T_in1 lhs, T_in2 rhs ) const
    {
      return mul_(lhs,rhs);
    }

    private:
    std::function<T_out(T_out,T_out)> add_;
    std::function<T_out(T_in1,T_in2)> mul_;
    T_out                             identity_;
  };

  template <typename T_in1, typename T_in2, typename T_out>
  template <typename MulOp, typename AddOp>
  Info Semiring<T_in1,T_in2,T_out>::nnew( MulOp mul_t, AddOp add_t )
  {
    mul_ = mul_t;
    add_ = add_t;
    identity_ = add_t.identity();
    return GrB_SUCCESS;
  }

}  // backend
}  // graphblas

#endif  // GRB_BACKEND_APSPIE_SEMIRING_HPP
