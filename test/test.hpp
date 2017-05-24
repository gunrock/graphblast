#ifndef GRB_TEST_HPP
#define GRB_TEST_HPP

#include <vector>

template <typename T, typename S>
void BOOST_ASSERT_LIST( const T* lhs, const S* rhs, int length=5 )
{
  for( int i=0; i<length; i++ )
			BOOST_ASSERT( lhs[i] == rhs[i] );
}

template <typename T, typename S>
void BOOST_ASSERT_LIST( std::vector<T>& lhs, const S* rhs, int length=5 )
{
  //length = lhs.size();
	for( int i=0; i<length; i++ )
    BOOST_ASSERT( lhs[i] == rhs[i] );
}

template <typename T, typename S>
void BOOST_ASSERT_LIST( std::vector<T>& lhs, std::vector<S>& rhs, int length=5 )
{
	length = std::min( lhs.size(), rhs.size() );
  for( int i=0; i<length; i++ )
    BOOST_ASSERT( lhs[i] == rhs[i] );
}

template <typename T, typename S>
void BOOST_ASSERT_LIST( const T* lhs, std::vector<S>& rhs, int length=5 )
{
	//length = rhs.size();
  for( int i=0; i<length; i++ )
    BOOST_ASSERT( lhs[i] == rhs[i] );
}

#endif  // GRB_TEST_HPP
