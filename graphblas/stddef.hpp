#ifndef GRB_STDDEF_HPP
#define GRB_STDDEF_HPP

#include <cstddef>
#include <cstdint>
#include <limits>

namespace graphblas
{
  // Unary Operations
  // TODO

  // Binary Operations
  template <typename T_in1=bool, typename T_in2=bool, typename T_out=bool>
  struct logical_or
  {
    inline GRB_HOST_DEVICE T_out operator()(T_in1 lhs, T_in2 rhs) 
    { return lhs || rhs; }
  };

  template <typename T_in1=bool, typename T_in2=bool, typename T_out=bool>
  struct logical_and
  {
    inline GRB_HOST_DEVICE T_out operator()(T_in1 lhs, T_in2 rhs) 
    { return lhs && rhs; }
  };

  template <typename T_in1=bool, typename T_in2=bool, typename T_out=bool>
  struct logical_xor
  {
    inline GRB_HOST_DEVICE T_out operator()(T_in1 lhs, T_in2 rhs) 
    { return lhs ^ rhs; }
  };

  template <typename T_in1, typename T_in2, typename T_out=bool>
  struct equal
  {
    inline GRB_HOST_DEVICE T_out operator()(T_in1 lhs, T_in2 rhs) 
    { return lhs == rhs; }
  };

  template <typename T_in1, typename T_in2=T_in1, typename T_out=bool>
  struct not_equal_to
  {
    inline GRB_HOST_DEVICE T_out operator()(T_in1 lhs, T_in2 rhs) 
    { return lhs != rhs; }
  };

  template <typename T_in1, typename T_in2=T_in1, typename T_out=bool>
  struct greater
  {
    inline GRB_HOST_DEVICE T_out operator()(T_in1 lhs, T_in2 rhs) 
    { return lhs > rhs; }
  };

  template <typename T_in1, typename T_in2=T_in1, typename T_out=bool>
  struct less
  {
    inline GRB_HOST_DEVICE T_out operator()(T_in1 lhs, T_in2 rhs) 
    { return lhs < rhs; }
  };

  template <typename T_in1, typename T_in2=T_in1, typename T_out=bool>
  struct greater_equal
  {
    inline GRB_HOST_DEVICE T_out operator()(T_in1 lhs, T_in2 rhs) 
    { return lhs >= rhs; }
  };

  template <typename T_in1, typename T_in2=T_in1, typename T_out=bool>
  struct less_equal
  {
    inline GRB_HOST_DEVICE T_out operator()(T_in1 lhs, T_in2 rhs) 
    { return lhs <= rhs; }
  };

  template <typename T_in1, typename T_in2=T_in1, typename T_out=T_in1>
  struct first
  {
    inline GRB_HOST_DEVICE T_out operator()(T_in1 lhs, T_in2 rhs)
    { return lhs; }
  };

  template <typename T_in1, typename T_in2=T_in1, typename T_out=T_in1>
  struct second
  {
    inline GRB_HOST_DEVICE T_out operator()(T_in1 lhs, T_in2 rhs)
    { return rhs; }
  };

  template <typename T_in1, typename T_in2=T_in1, typename T_out=T_in1>
  struct minimum
  {
    inline GRB_HOST_DEVICE T_out operator()(T_in1 lhs, T_in2 rhs)
    { return min(lhs, rhs); }
  };

  template <typename T_in1, typename T_in2=T_in1, typename T_out=T_in1>
  struct maximum
  {
    inline GRB_HOST_DEVICE T_out operator()(T_in1 lhs, T_in2 rhs)
    { return max(lhs, rhs); }
  };

  template <typename T_in1, typename T_in2=T_in1, typename T_out=T_in1>
  struct plus
  {
    inline GRB_HOST_DEVICE T_out operator()(T_in1 lhs, T_in2 rhs) 
    { return lhs + rhs; }
  };

  template <typename T_in1, typename T_in2=T_in1, typename T_out=T_in1>
  struct minus
  {
    inline GRB_HOST_DEVICE T_out operator()(T_in1 lhs, T_in2 rhs)
    { return lhs - rhs; }
  };

  template <typename T_in1, typename T_in2=T_in1, typename T_out=T_in1>
  struct multiplies
  {
    inline GRB_HOST_DEVICE T_out operator()(T_in1 lhs, T_in2 rhs) 
    { return lhs * rhs; }
  };

  template <typename T_in1, typename T_in2=T_in1, typename T_out=T_in1>
  struct divides
  {
    inline GRB_HOST_DEVICE T_out operator()(T_in1 lhs, T_in2 rhs) 
    { return lhs / rhs; }
  };
}  // graphblas

// Monoid generator macro provided by Scott McMillan
#define REGISTER_MONOID(M_NAME, BINARYOP, IDENTITY)                            \
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

namespace graphblas
{
  // Monoids
  REGISTER_MONOID( PlusMonoid, plus, 0 )
  REGISTER_MONOID( MultipliesMonoid, multiplies, 1 )
  REGISTER_MONOID( MinimumMonoid, minimum, std::numeric_limits<T_out>::max() )
  REGISTER_MONOID( MaximumMonoid, maximum, std::numeric_limits<T_out>::min() )
  REGISTER_MONOID( LogicalOrMonoid, logical_or, false )
  REGISTER_MONOID( LogicalAndMonoid, logical_and, false )

  // New monoids
  REGISTER_MONOID( GreaterMonoid, greater, std::numeric_limits<T_out>::min() );
  REGISTER_MONOID( LessMonoid, less, std::numeric_limits<T_out>::max() );
}  // graphblas

// Semiring generator macro provided by Scott McMillan
#define REGISTER_SEMIRING(SR_NAME, ADD_MONOID, MULT_BINARYOP)           \
	template <typename T_in1, typename T_in2=T_in1, typename T_out=T_in1> \
	struct SR_NAME                                                        \
	{                                                                     \
    typedef T_out result_type;                                          \
    typedef T_out T_out_type;                                           \
                                                                        \
		inline T_out identity() const                                       \
		{ return ADD_MONOID<T_out>().identity(); }                          \
                                                                        \
    /*template<BinaryOp op>                                                   \
    inline __host__ __device__ T_out operator()(T_in1 lhs, T_in2 rhs) const \
    {                                                                       \
      if( op==GrB_ADD )                                                     \
        return ADD_MONOID<T_out>()(lhs, rhs);                               \
      else if( op==GrB_MUL )                                                \
        return MULT_BINARYOP<T_in1,T_in2,T_out>()(lhs,rhs);                 \
    } */                                                                      \
		inline __host__ __device__ T_out add_op(T_out lhs, T_out rhs)       \
		{ return ADD_MONOID<T_out>()(lhs, rhs); }                           \
																																				\
		inline __host__ __device__ T_out mul_op(T_in1 lhs, T_in2 rhs)       \
		{ return MULT_BINARYOP<T_in1,T_in2,T_out>()(lhs, rhs); }            \
  };

namespace graphblas
{
  // Semirings
  REGISTER_SEMIRING( LogicalOrAndSemiring, LogicalOrMonoid, logical_and )
  REGISTER_SEMIRING( PlusMultipliesSemiring, PlusMonoid, multiplies )
  REGISTER_SEMIRING( MinimumPlusSemiring, MinimumMonoid, plus )
  REGISTER_SEMIRING( MaximumMultipliesSemiring, MaximumMonoid, multiplies )

  // New semirings
  REGISTER_SEMIRING( PlusDividesSemiring, PlusMonoid, divides )
  REGISTER_SEMIRING( PlusGreaterSemiring, PlusMonoid, greater )
  REGISTER_SEMIRING( GreaterPlusSemiring, GreaterMonoid, plus )
  REGISTER_SEMIRING( LessPlusSemiring, LessMonoid, plus )
  REGISTER_SEMIRING( PlusLessSemiring, PlusMonoid, less )

  // AddOp and MulOp extraction provided by Peter Zhang
	template <typename SemiringT>
	struct AdditiveMonoidFromSemiring
	{
	public:
		typedef typename SemiringT::T_out_type T_out_type;
		typedef typename SemiringT::T_out_type result_type;

		typedef typename SemiringT::T_out_type first_argument_type;
		typedef typename SemiringT::T_out_type second_argument_type;

		AdditiveMonoidFromSemiring() : sr() {}
		AdditiveMonoidFromSemiring(SemiringT const &sr) : sr(sr) {}

		inline GRB_HOST_DEVICE T_out_type identity() const
		{
			return sr.identity();
		}

		template <typename T_in1, typename T_in2>
		inline GRB_HOST_DEVICE T_out_type operator()(T_in1 lhs, T_in2 rhs)
		{
			return sr.add_op(lhs, rhs);
		}

	private:
		SemiringT sr;
	};

	template <typename SemiringT>
	struct MultiplicativeMonoidFromSemiring
	{
	public:
		typedef typename SemiringT::T_out_type T_out_type;
		typedef typename SemiringT::T_out_type result_type;

		typedef typename SemiringT::T_out_type first_argument_type;
		typedef typename SemiringT::T_out_type second_argument_type;

		MultiplicativeMonoidFromSemiring() : sr() {}
		MultiplicativeMonoidFromSemiring(SemiringT const &sr) : sr(sr) {}

		inline GRB_HOST_DEVICE T_out_type identity() const
		{
			return sr.identity();
		}

		template <typename T_in1, typename T_in2>
		inline GRB_HOST_DEVICE T_out_type operator()(T_in1 lhs, T_in2 rhs)
		{
			return sr.mul_op(lhs, rhs);
		}

	private:
		SemiringT sr;
	};

	template <typename SemiringT>
	AdditiveMonoidFromSemiring<SemiringT>
	extractAdd(SemiringT const &sr)
	{
	  return AdditiveMonoidFromSemiring<SemiringT>(sr);
	}

	template <typename SemiringT>
	MultiplicativeMonoidFromSemiring<SemiringT>
	extractMul(SemiringT const &sr)
	{
		return MultiplicativeMonoidFromSemiring<SemiringT>(sr);
	}
}

#endif  // GRB_STDDEF_HPP
