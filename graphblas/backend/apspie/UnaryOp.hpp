#ifndef GRB_BACKEND_APSPIE_UNARYOP_HPP
#define GRB_BACKEND_APSPIE_UNARYOP_HPP

#include <vector>

#include "graphblas/backend/apspie/apspie.hpp"

namespace graphblas
{
namespace backend
{
  template<typename T_in=bool, typename T_out=T_in>
  class UnaryOp
  {
    public:
    UnaryOp() {}
    template <typename Op>
    UnaryOp( Op op )
    {
      op_ = op;
    }

    // Default Destructor
    ~UnaryOp() {}

    // C API Methods
    template <typename Op>
    Info nnew( Op op );

    __host__ __device__ inline T_out operator()( T_in rhs ) const
    {
      return op_(rhs);
    }

    private:
    std::function<T_out(T_in)> op_;
  };

  template <typename T_in, typename T_out>
  template <typename Op>
  Info UnaryOp<T_in,T_out>::nnew( Op op )
  {
    op_ = op;
    return GrB_SUCCESS;
  }

}  // backend
}  // graphblas

#endif  // GRB_BACKEND_APSPIE_UNARYOP_HPP
