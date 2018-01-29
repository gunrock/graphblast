#define GRB_USE_APSPIE

#include <vector>
#include <iostream>

#include "graphblas/graphblas.hpp"
#include "test/test.hpp"

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE vec_suite

#include <boost/test/included/unit_test.hpp>
#include <boost/program_options.hpp>

using namespace graphblas;

void testSet( const std::vector<Desc_field>& field, 
              const std::vector<Desc_value>& value )
{
  Descriptor desc;
  for( int i=0; i<field.size(); i++ )
    CHECKVOID( desc.set(field[i], value[i]) );

  Desc_value val;
  for( int i=0; i<field.size(); i++ )
  {
    CHECKVOID( desc.get(field[i], &val) );
    BOOST_ASSERT( val==value[i] );
  }
}

void testSetInt( const std::vector<Desc_field>& field, 
                 const std::vector<Desc_value>& value )
{
  Descriptor desc;
  for( int i=0; i<field.size(); i++ )
    CHECKVOID( desc.set(field[i], static_cast<int>(value[i])) );

  Desc_value val;
  for( int i=0; i<field.size(); i++ )
  {
    CHECKVOID( desc.get(field[i], &val) );
    BOOST_ASSERT( val==value[i] );
  }
}

void testGet( const std::vector<Desc_field>& field, 
              const std::vector<Desc_value>& value )
{
  Descriptor desc;
  for( int i=0; i<field.size(); i++ )
    CHECKVOID( desc.set(field[i], value[i]) );

  Desc_value val;
  for( int i=0; i<field.size(); i++ )
  {
    CHECKVOID( desc.get(field[i], &val) );
    BOOST_ASSERT( val==value[i] );
  }
}

void testToggle( const std::vector<Desc_field>& field, 
                 const std::vector<Desc_value>& value )
{
  Descriptor desc;
  for( int i=0; i<field.size(); i++ )
    CHECKVOID( desc.toggle(field[i]) );

  Desc_value val;
  for( int i=0; i<4; i++ )
  {
    CHECKVOID( desc.get(field[i], &val) );
    BOOST_ASSERT( val==value[i] );
    CHECKVOID( desc.toggle(field[i]) );
    CHECKVOID( desc.get(field[i], &val) );
    BOOST_ASSERT( val==GrB_DEFAULT );
  }
}

struct TestDescriptor
{
  TestDescriptor() :
    DEBUG(true) {}

  bool DEBUG;
};

BOOST_AUTO_TEST_SUITE( desc_suite )

BOOST_FIXTURE_TEST_CASE( desc1, TestDescriptor )
{
  std::vector<Desc_field> field = { GrB_MASK, GrB_OUTP, GrB_INP0, 
      GrB_INP1, GrB_MODE, GrB_TA, GrB_TB, GrB_NT, GrB_MXVMODE, GrB_SPMSPVMODE, 
      GrB_TOL };
  std::vector<Desc_value> value = { GrB_SCMP, GrB_REPLACE, GrB_TRAN, 
      GrB_TRAN, GrB_FIXEDROW, GrB_8, GrB_8, GrB_32, GrB_PUSHONLY, GrB_GUNROCKLB,
      GrB_16 };

  testSet(    field, value );
  testSetInt( field, value );
  testGet(    field, value );
}

BOOST_FIXTURE_TEST_CASE( desc2, TestDescriptor )
{
  std::vector<Desc_field> field = { GrB_MASK, GrB_OUTP, GrB_INP0, 
      GrB_INP1, GrB_MODE, GrB_TA, GrB_TB, GrB_NT, GrB_MXVMODE, GrB_SPMSPVMODE, 
      GrB_TOL };
  std::vector<Desc_value> value = { GrB_SCMP, GrB_REPLACE, GrB_TRAN, 
      GrB_TRAN, GrB_FIXEDROW, GrB_8, GrB_8, GrB_32, GrB_PUSHONLY, GrB_GUNROCKLB,
      GrB_16 };

  testToggle( field, value );
}

BOOST_AUTO_TEST_SUITE_END()
