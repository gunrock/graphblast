#ifndef GRAPHBLAS_STDDEF_HPP_
#define GRAPHBLAS_STDDEF_HPP_

#include <cstddef>
#include <cstdint>
#include <limits>
#include <algorithm>

namespace graphblas {
// Unary Operations
// TODO(@ctcyang): add unary ops

// Binary Operations
template <typename T_in1 = bool, typename T_in2 = bool, typename T_out = bool>
struct logical_or {
  inline GRB_HOST_DEVICE T_out operator()(T_in1 lhs, T_in2 rhs) {
    return lhs || rhs;
  }
};

template <typename T_in1 = bool, typename T_in2 = bool, typename T_out = bool>
struct logical_and {
  inline GRB_HOST_DEVICE T_out operator()(T_in1 lhs, T_in2 rhs) {
    return lhs && rhs;
  }
};

template <typename T_in1 = bool, typename T_in2 = bool, typename T_out = bool>
struct logical_xor {
  inline GRB_HOST_DEVICE T_out operator()(T_in1 lhs, T_in2 rhs) {
    return (lhs && !rhs) || (!lhs && rhs);
  }
};

template <typename T_in1, typename T_in2 = T_in1, typename T_out = T_in1>
struct equal {
  inline GRB_HOST_DEVICE T_out operator()(T_in1 lhs, T_in2 rhs) {
    return lhs == rhs;
  }
};

template <typename T_in1, typename T_in2 = T_in1, typename T_out = T_in1>
struct not_equal_to {
  inline GRB_HOST_DEVICE T_out operator()(T_in1 lhs, T_in2 rhs) {
    return lhs != rhs;
  }
};

template <typename T_in1, typename T_in2 = T_in1, typename T_out = bool>
struct greater {
  inline GRB_HOST_DEVICE T_out operator()(T_in1 lhs, T_in2 rhs) {
    return lhs > rhs;
  }
};

template <typename T_in1, typename T_in2 = T_in1, typename T_out = bool>
struct less {
  inline GRB_HOST_DEVICE T_out operator()(T_in1 lhs, T_in2 rhs) {
    return lhs < rhs;
  }
};

template <typename T_in1, typename T_in2 = T_in1, typename T_out = bool>
struct greater_equal {
  inline GRB_HOST_DEVICE T_out operator()(T_in1 lhs, T_in2 rhs) {
    return lhs >= rhs;
  }
};

template <typename T_in1, typename T_in2 = T_in1, typename T_out = bool>
struct less_equal {
  inline GRB_HOST_DEVICE T_out operator()(T_in1 lhs, T_in2 rhs) {
    return lhs <= rhs;
  }
};
namespace fixme {
template <typename T_in1, typename T_in2 = T_in1, typename T_out = T_in1>
struct first {
  inline GRB_HOST_DEVICE T_out operator()(T_in1 lhs, T_in2 rhs) {
    return lhs;
  }
};
}
namespace fixme {
template <typename T_in1, typename T_in2 = T_in1, typename T_out = T_in1>
struct second {
  inline GRB_HOST_DEVICE T_out operator()(T_in1 lhs, T_in2 rhs) {
    return rhs;
  }
};
}

template <typename T_in1, typename T_in2 = T_in1, typename T_out = T_in1>
struct minimum {
  inline GRB_HOST_DEVICE T_out operator()(T_in1 lhs, T_in2 rhs) {
    return min(lhs, rhs);
  }
};

template <typename T_in1, typename T_in2 = T_in1, typename T_out = T_in1>
struct maximum {
  inline GRB_HOST_DEVICE T_out operator()(T_in1 lhs, T_in2 rhs) {
    return max(lhs, rhs);
  }
};

template <typename T_in1, typename T_in2 = T_in1, typename T_out = T_in1>
struct plus {
  inline GRB_HOST_DEVICE T_out operator()(T_in1 lhs, T_in2 rhs) {
    return lhs + rhs;
  }
};

template <typename T_in1, typename T_in2 = T_in1, typename T_out = T_in1>
struct minus {
  inline GRB_HOST_DEVICE T_out operator()(T_in1 lhs, T_in2 rhs) {
    return lhs - rhs;
  }
};

template <typename T_in1, typename T_in2 = T_in1, typename T_out = T_in1>
struct multiplies {
  inline GRB_HOST_DEVICE T_out operator()(T_in1 lhs, T_in2 rhs) {
    return lhs * rhs;
  }
};

template <typename T_in1, typename T_in2 = T_in1, typename T_out = T_in1>
struct divides {
  inline GRB_HOST_DEVICE T_out operator()(T_in1 lhs, T_in2 rhs) {
    return lhs / rhs;
  }
};

template <typename T_in1, typename T_in2 = T_in1, typename T_out = T_in1>
struct select_second {
  inline GRB_HOST_DEVICE T_out operator()(T_in1 lhs, T_in2 rhs) {
    return rhs;
  }
};
}  // namespace graphblas

// Monoid generator macro provided by Scott McMillan.
#define REGISTER_MONOID(M_NAME, BINARYOP, IDENTITY)                          \
template <typename T_out>                                                    \
struct M_NAME                                                                \
{                                                                            \
  inline T_out identity() const                                              \
  {                                                                          \
    return static_cast<T_out>(IDENTITY);                                     \
  }                                                                          \
                                                                             \
  inline __host__ __device__ T_out operator()(T_out lhs, T_out rhs) const    \
  {                                                                          \
    return BINARYOP<T_out>()(lhs, rhs);                                      \
  }                                                                          \
};

namespace graphblas {
// Monoids
REGISTER_MONOID(PlusMonoid, plus, 0)
REGISTER_MONOID(MultipliesMonoid, multiplies, 1)
REGISTER_MONOID(MinimumMonoid, minimum, std::numeric_limits<T_out>::max())
REGISTER_MONOID(MaximumMonoid, maximum, 0)
REGISTER_MONOID(LogicalOrMonoid, logical_or, false)
REGISTER_MONOID(LogicalAndMonoid, logical_and, false)

// New monoids
REGISTER_MONOID(GreaterMonoid, greater, std::numeric_limits<T_out>::min());
// Less is not a monoid because:
// 1) has different left and right identity
// 2) not associative
REGISTER_MONOID(CustomLessMonoid, less, std::numeric_limits<T_out>::max());
REGISTER_MONOID(NotEqualToMonoid, not_equal_to, std::numeric_limits<T_out>::max())
}  // namespace graphblas

// Semiring generator macro provided by Scott McMillan
#define REGISTER_SEMIRING(SR_NAME, ADD_MONOID, MULT_BINARYOP)             \
template <typename T_in1, typename T_in2 = T_in1, typename T_out = T_in1> \
struct SR_NAME                                                            \
{                                                                         \
  typedef T_out result_type;                                              \
  typedef T_out T_out_type;                                               \
                                                                          \
  static inline __host__ __device__ T_out identity()                                    \
  { return ADD_MONOID<T_out>().identity(); }                              \
                                                                          \
  inline __host__ __device__ T_out add_op(const T_out& lhs, const T_out& rhs) const     \
  { return ADD_MONOID<T_out>()(lhs, rhs); }                               \
                                                                          \
  inline __host__ __device__ T_out mul_op(const T_in1& lhs, const T_in2& rhs) const     \
  { return MULT_BINARYOP<T_in1, T_in2, T_out>()(lhs, rhs); }              \
};

namespace graphblas {
// Semirings
REGISTER_SEMIRING(LogicalOrAndSemiring, LogicalOrMonoid, logical_and)
REGISTER_SEMIRING(PlusMultipliesSemiring, PlusMonoid, multiplies)
REGISTER_SEMIRING(MinimumPlusSemiring, MinimumMonoid, plus)
REGISTER_SEMIRING(MaximumMultipliesSemiring, MaximumMonoid, multiplies)

// New semirings
REGISTER_SEMIRING(PlusDividesSemiring, PlusMonoid, divides)
REGISTER_SEMIRING(PlusGreaterSemiring, PlusMonoid, greater)
REGISTER_SEMIRING(GreaterPlusSemiring, GreaterMonoid, plus)
REGISTER_SEMIRING(PlusMinusSemiring, PlusMonoid, minus)
REGISTER_SEMIRING(PlusLessSemiring, PlusMonoid, less)
REGISTER_SEMIRING(CustomLessPlusSemiring, CustomLessMonoid, plus)
REGISTER_SEMIRING(MinimumMultipliesSemiring, MinimumMonoid, multiplies)
REGISTER_SEMIRING(MultipliesMultipliesSemiring, MultipliesMonoid, multiplies)
REGISTER_SEMIRING(NotEqualToPlusSemiring, NotEqualToMonoid, plus)
REGISTER_SEMIRING(MinimumSelectSecondSemiring, MinimumMonoid, select_second)
REGISTER_SEMIRING(PlusNotEqualToSemiring, PlusMonoid, not_equal_to)
REGISTER_SEMIRING(CustomLessLessSemiring, CustomLessMonoid, less)
REGISTER_SEMIRING(MinimumNotEqualToSemiring, MinimumMonoid, not_equal_to)

// AddOp and MulOp extraction provided by Peter Zhang:
//   www.github.com/cmu-sei/gbtl/
template <typename SemiringT>
struct AdditiveMonoidFromSemiring {
 public:
  typedef typename SemiringT::T_out_type T_out_type;
  typedef typename SemiringT::T_out_type result_type;

  typedef typename SemiringT::T_out_type first_argument_type;
  typedef typename SemiringT::T_out_type second_argument_type;

  AdditiveMonoidFromSemiring() : sr() {}
  explicit AdditiveMonoidFromSemiring(SemiringT const &sr) : sr(sr) {}

  inline GRB_HOST_DEVICE T_out_type identity() const {
    return sr.identity();
  }

  template <typename T_in1, typename T_in2>
  inline GRB_HOST_DEVICE T_out_type operator()(T_in1 lhs, T_in2 rhs) {
    return sr.add_op(lhs, rhs);
  }

 private:
  SemiringT sr;
};

template <typename SemiringT>
struct MultiplicativeMonoidFromSemiring {
 public:
  typedef typename SemiringT::T_out_type T_out_type;
  typedef typename SemiringT::T_out_type result_type;

  typedef typename SemiringT::T_out_type first_argument_type;
  typedef typename SemiringT::T_out_type second_argument_type;

  MultiplicativeMonoidFromSemiring() : sr() {}
  explicit MultiplicativeMonoidFromSemiring(SemiringT const &sr) : sr(sr) {}

  inline GRB_HOST_DEVICE T_out_type identity() const {
    return sr.identity();
  }

  template <typename T_in1, typename T_in2>
  inline GRB_HOST_DEVICE T_out_type operator()(T_in1 lhs, T_in2 rhs) {
    return sr.mul_op(lhs, rhs);
  }

 private:
  SemiringT sr;
};

template <typename SemiringT>
AdditiveMonoidFromSemiring<SemiringT>
extractAdd(SemiringT const &sr) {
  return AdditiveMonoidFromSemiring<SemiringT>(sr);
}

template <typename SemiringT>
MultiplicativeMonoidFromSemiring<SemiringT>
extractMul(SemiringT const &sr) {
  return MultiplicativeMonoidFromSemiring<SemiringT>(sr);
}
}  // namespace graphblas

#endif  // GRAPHBLAS_STDDEF_HPP_
