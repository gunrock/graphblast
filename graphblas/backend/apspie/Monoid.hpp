#ifndef GRB_BACKEND_APSPIE_MONOID_HPP
#define GRB_BACKEND_APSPIE_MONOID_HPP

#include <vector>

#include "graphblas/backend/apspie/apspie.hpp"

namespace graphblas
{
namespace backend
{
  template<typename T>
  class Monoid
  {
    public:
    Monoid() {}
    Monoid( BinaryOp* binary_op, T identity ) : identity_(identity)
    {
      op = binary_op;
    }

    // Default Destructor
    ~Monoid();

    // C API Methods
    Info nnew( BinaryOp* binary_op, T identity );

    private:
    BinaryOp op;
    T        identity_;
  };

  template <typename T>
  Info Monoid::nnew( BinaryOp* binary_op, T identity )
  {
    return GrB_SUCCESS;
  }

}  // backend
}  // graphblas

#endif  // GRB_MONOID_HPP
