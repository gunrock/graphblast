#ifndef GRB_STDDEF_HPP
#define GRB_STDDEF_HPP

#include <cstddef>
#include <cstdint>

namespace graphblas
{
  template <typename T_in1=bool, typename T_in2=bool, typename T_out=bool>
  struct logical_or
  {
    inline T_out operator()(T_in1 lhs, T_in2 rhs) { return lhs || rhs; }
  };

  template <typename T_in1=bool, typename T_in2=bool, typename T_out=bool>
  struct logical_and
  {
    inline T_out operator()(T_in1 lhs, T_in2 rhs) { return lhs && rhs; }
  };

  template <typename T_in1=bool, typename T_in2=bool, typename T_out=bool>
  struct logical_xor
  {
    inline T_out operator()(T_in1 lhs, T_in2 rhs) { return lhs ^ rhs; }
  };

  template <typename T_in1, typename T_in2, typename T_out=bool>
  struct equal
  {
    inline T_out operator()(T_in1 lhs, T_in2 rhs) { return lhs == rhs; }
  };

  template <typename T_in1, typename T_in2=T_in1, typename T_out=T_in1>
  struct not_equal_to
  {
    inline T_out operator()(T_in1 lhs, T_in2 rhs) { return lhs != rhs; }
  };

  template <typename T_in1, typename T_in2=T_in1, typename T_out=T_in1>
  struct greater
  {
    inline T_out operator()(T_in1 lhs, T_in2 rhs) { return lhs > rhs; }
  };

  template <typename T_in1, typename T_in2=T_in1, typename T_out=T_in1>
  struct less
  {
    inline T_out operator()(T_in1 lhs, T_in2 rhs) { return lhs < rhs; }
  };

  template <typename T_in1, typename T_in2=T_in1, typename T_out=T_in1>
  struct greater_equal
  {
    inline T_out operator()(T_in1 lhs, T_in2 rhs) { return lhs >= rhs; }
  };

  template <typename T_in1, typename T_in2=T_in1, typename T_out=T_in1>
  struct less_equal
  {
    inline T_out operator()(T_in1 lhs, T_in2 rhs) { return lhs <= rhs; }
  };

  template <typename T_in1, typename T_in2=T_in1, typename T_out=T_in1>
  struct plus
  {
    inline __host__ __device__ T_out operator()(T_in1 lhs, T_in2 rhs) { return lhs+rhs; }
  };

  template <typename T_in1, typename T_in2=T_in1, typename T_out=T_in1>
  struct multiplies
  {
    inline __host__ __device__ T_out operator()(T_in1 lhs, T_in2 rhs) { return lhs*rhs; }
  };
}

/*#define GEN_BINARYOP_BOOL( S_NAME, D_NAME )                                    \
  typedef BinaryOp<bool,bool    ,bool    >( S_NAME<bool    >() ) D_NAME_BOOL;  \
  typedef BinaryOp<bool,int8_t  ,int8_t  >( S_NAME<int8_t  >() ) D_NAME_INT8;  \
  typedef BinaryOp<bool,uint8_t ,uint8_t >( S_NAME<uint8_t >() ) D_NAME_UINT8; \
  typedef BinaryOp<bool,int16_t ,int16_t >( S_NAME<int16_t >() ) D_NAME_INT16; \
  typedef BinaryOp<bool,uint16_t,uint16_t>( S_NAME<uint16_t>() ) D_NAME_UINT16;\
  typedef BinaryOp<bool,int32_t ,int32_t >( S_NAME<int32_t >() ) D_NAME_INT32; \
  typedef BinaryOp<bool,uint32_t,uint32_t>( S_NAME<uint32_t>() ) D_NAME_UINT32;\
  typedef BinaryOp<bool,float   ,float   >( S_NAME<float   >() ) D_NAME_FP32;  \
  typedef BinaryOp<bool,double  ,double  >( S_NAME<double  >() ) D_NAME_FP64;

namespace graphblas
{
  typedef BinaryOp<bool,bool,bool>( logical_or <bool>() ) GrB_LOR;
  typedef BinaryOp<bool,bool,bool>( logical_and<bool>() ) GrB_LAND;
  typedef BinaryOp<bool,bool,bool>( logical_xor<bool>() ) GrB_XOR;
  GEN_BINARYOP_BOOL( equal,         GrB_EQ );
  GEN_BINARYOP_BOOL( not_equal_to,  GrB_NE );
  GEN_BINARYOP_BOOL( greater,       GrB_GT );
  GEN_BINARYOP_BOOL( less,          GrB_LT );
  GEN_BINARYOP_BOOL( greater_equal, GrB_GE );
  GEN_BINARYOP_BOOL( less_equal,    GrB_LE );
}

#define GEN_BINARYOP_ONE( S_NAME, D_NAME )                                     \
 typedef BinaryOp<bool    ,bool    ,bool    >(S_NAME<bool    >())D_NAME_BOOL;  \
 typedef BinaryOp<int8_t  ,int8_t  ,int8_t  >(S_NAME<int8_t  >())D_NAME_INT8;  \
 typedef BinaryOp<uint8_t ,uint8_t ,uint8_t >(S_NAME<uint8_t >())D_NAME_UINT8; \
 typedef BinaryOp<int16_t ,int16_t ,int16_t >(S_NAME<int16_t >())D_NAME_INT16; \
 typedef BinaryOp<uint16_t,uint16_t,uint16_t>(S_NAME<uint16_t>())D_NAME_UINT16;\
 typedef BinaryOp<int32_t ,int32_t ,int32_t >(S_NAME<int32_t >())D_NAME_INT32; \
 typedef BinaryOp<uint32_t,uint32_t,uint32_t>(S_NAME<uint32_t>())D_NAME_UINT32;\
 typedef BinaryOp<float   ,float   ,float   >(S_NAME<float   >())D_NAME_FP32;  \
 typedef BinaryOp<double  ,double  ,double  >(S_NAME<double  >())D_NAME_FP64;

#define GEN_BINARYOP_ONE( S_NAME, D_NAME )
  template <typename T>                    \
  struct D_NAME                            \
  {                                        \
    T operator() (T lhs, T rhs)
    {

namespace graphblas
{
  GEN_BINARYOP_ONE( first,   GrB_FIRST  );
  GEN_BINARYOP_ONE( second,  GrB_SECOND );
  GEN_BINARYOP_ONE( min,     GrB_MIN    );
  GEN_BINARYOP_ONE( max,     GrB_MAX    );
  GEN_BINARYOP_ONE( plus,    GrB_PLUS   );
  GEN_BINARYOP_ONE( minus,   GrB_MINUS  );
  GEN_BINARYOP_ONE( times,   GrB_TIMES  );
  GEN_BINARYOP_ONE( divides, GrB_DIV    );
}

namespace graphblas
{
  enum Operator {GrB_IDENTITY,           // for UnaryOp
                 GrB_AINV,
                 GrB_MINV,
                 GrB_LNOT,
                 GrB LOR,                // for BinaryOp
                 GrB_LAND,
                 GrB_LXOR,
                 GrB_EQ,
                 GrB_NE,
                 GrB_GT,
                 GrB_LT,
                 GrB_GE,
                 GrB_LE,
                 GrB_FIRST,
                 GrB_SECOND,
                 GrB_MIN,
                 GrB_MAX,
                 GrB_PLUS,
                 GrB_MINUS,
                 GrB_TIMES,
                 GrB_DIV};
}*/

#endif  // GRB_STDDEF_HPP
