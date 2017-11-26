#ifndef GRB_BACKEND_APSPIE_BINARYOP_HPP
#define GRB_BACKEND_APSPIE_BINARYOP_HPP

#include <vector>

#include "graphblas/stddef.hpp"
#include "graphblas/backend/apspie/apspie.hpp"

namespace graphblas
{
namespace backend
{
  template<typename T_in1=bool, typename T_in2=T_in1, typename T_out=T_in1>
  class BinaryOp
  {
    public:
    BinaryOp() : op_(graphblas::plus<T>()) {}
    template <typename Op>
    BinaryOp( Op op ) : op_(op) {}

    // Default Destructor
    ~BinaryOp() {}

    // C API Methods
    template <typename Op>
    Info nnew( Op op );

    inline __host__ __device__ T_out operator()( T_in1 lhs, T_in2 rhs ) const
    {
      return op_(lhs,rhs);
    }

    private:
    std::function<T_out(T_in1,T_in2)> op_;
  };

  template <typename T_in1, typename T_in2, typename T_out>
  template <typename Op>
  Info BinaryOp<T_in1,T_in2,T_out>::nnew( Op op )
  {
    op_ = op;
    return GrB_SUCCESS;
  }

}  // backend
}  // graphblas

#endif  // GRB_BACKEND_APSPIE_BINARYOP_HPP
