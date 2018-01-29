#ifndef GRB_TEST_HPP
#define GRB_TEST_HPP

#include <vector>
#include <iostream>

template <typename T>
void BOOST_ASSERT_FLOAT( const T lhs,
                         const T rhs,
                         const T tol=0.001 )
{
  if( rhs==0 || lhs==0 )
    BOOST_ASSERT( fabs(lhs)<tol );
  else
    BOOST_ASSERT( fabs(lhs-rhs)<tol*fabs(rhs) );
}

template <typename T, typename S, typename L>
void BOOST_ASSERT_LIST( const T* lhs, 
                        const S* rhs, 
                        L length=5 )
{
  for( L i=0; i<length; i++ )
    BOOST_ASSERT( lhs[i] == rhs[i] );
}

template <typename T, typename S, typename L>
void BOOST_ASSERT_LIST( const std::vector<T>& lhs, 
                        const S* rhs, 
                        L length=5 )
{
  //length = lhs.size();
	for( L i=0; i<length; i++ )
    BOOST_ASSERT( lhs[i] == rhs[i] );
}

template <typename T, typename S, typename L>
void BOOST_ASSERT_LIST( const std::vector<T>& lhs,
                        const std::vector<S>& rhs, 
                        L length=5 )
{
	length = std::min( lhs.size(), rhs.size() );
  for( L i=0; i<length; i++ )
  {
    //std::cout << lhs[i] << "==" << rhs[i] << std::endl;
    BOOST_ASSERT( lhs[i] == rhs[i] );
  }
}

template <typename T, typename S, typename L>
void BOOST_ASSERT_LIST( const T* lhs, 
                        const std::vector<S>& rhs,
                        L length=5 )
{
  int flag = 0;
	//length = rhs.size();
  for( L i=0; i<length; i++ )
  {
    if( lhs[i]!=rhs[i] && flag==0 )
    {
			printf("\nINCORRECT: [%lu]: ", (unsigned long) i);
			std::cout << rhs[i] << " != " << lhs[i] << "\nresult[...";

			for( size_t j = (i >= 5) ? i - 5 : 0; (j < i + 5) && (j < length); j++ )
			  std::cout << rhs[j] << ", ";
			std::cout << "...]\nlhs[...";

			for( size_t j = (i >= 5) ? i - 5 : 0; (j < i + 5) && (j < length); j++ )
				std::cout << lhs[j] << ", ";
			std::cout << "...]";
      flag += 1;
    }
    if( lhs[i]!=rhs[i] && flag>0 )
      flag += 1;
    //BOOST_ASSERT( lhs[i] == rhs[i] );
  }
  std::cout << "\n";
  if( flag==0 )
	  std::cout << "CORRECT\n";
  else
    std::cout << flag << " errors occured.\n";
}

template <typename T>
void check( const graphblas::backend::SparseMatrix<T>& A )
{
  std::cout << "Begin check:\n";
  //printArray( "rowptr", h_csrRowPtr_ );
  //printArray( "colind", h_csrColInd_+23 );
  // Check csrRowPtr is monotonically increasing
  for( graphblas::Index row=0; row<A.nrows_; row++ )
  {
    //std::cout << "Comparing " << A.h_csrRowPtr_[row+1] << " >= " << A.h_csrRowPtr_[row] << std::endl;
    BOOST_ASSERT( A.h_csrRowPtr_[row+1]>=A.h_csrRowPtr_[row] );
  }

  // Check that: 1) there are no -1's in ColInd
  //             2) monotonically increasing
  for( graphblas::Index row=0; row<A.nrows_; row++ )
  {
    graphblas::Index row_start = A.h_csrRowPtr_[row];
    graphblas::Index row_end   = A.h_csrRowPtr_[row+1];
    //graphblas::Index row_end   = row_start+A.h_rowLength_[row];
    //std::cout << row << " ";
    //printArray( "colind", h_csrColInd_+row_start, h_rowLength_[row] );
    for( graphblas::Index col=row_start; col<row_end-1; col++ )
    {
      //std::cout << "Comparing " << A.h_csrColInd_[col+1] << " > " << A.h_csrColInd_[col] << std::endl;
      BOOST_ASSERT( A.h_csrColInd_[col]!=-1 );
      BOOST_ASSERT( A.h_csrColInd_[col+1]>A.h_csrColInd_[col] );
    }
  }
}

#endif  // GRB_TEST_HPP
