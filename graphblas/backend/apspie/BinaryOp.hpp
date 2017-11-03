#ifndef GRB_BACKEND_APSPIE_BINARYOP_HPP
#define GRB_BACKEND_APSPIE_BINARYOP_HPP

#include <vector>

#include "graphblas/backend/apspie/apspie.hpp"

namespace graphblas
{
namespace backend
{
  template<typename T_out, typename T_in1=T_out, typename T_in2=T_out>
  class BinaryOp
  {
    public:
    BinaryOp() {}
    template <typename Op>
    BinaryOp( Op* op )
    {
      if( op==NULL ) return;
      op_ = *op;
    }

    // Default Destructor
    ~BinaryOp() {}

    // C API Methods
    template <typename Op>
    Info nnew( Op* op );

    T_out operator()( T_in1 lhs, T_in2 rhs ) const
    {
      return op_(lhs,rhs);
    }

    private:
    std::function<T_out(T_in1,T_in2)> op_;
  };

  template <typename T_out, typename T_in1, typename T_in2>
  template <typename Op>
  Info BinaryOp<T_out,T_in1,T_in2>::nnew( Op* op )
  {
    if( op==NULL ) return GrB_NULL_POINTER;
    op_ = *op;
    return GrB_SUCCESS;
  }

}  // backend
}  // graphblas

#endif  // GRB_BACKEND_APSPIE_BINARYOP_HPP
