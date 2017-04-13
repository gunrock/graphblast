#ifndef GRB_BACKEND_APSPIE_HPP
#define GRB_BACKEND_APSPIE_HPP

#include <vector>
//#include <iostream>

#include <graphblas/backend/apspie/CooMatrix.hpp>

namespace graphblas
{
namespace backend
{
  template <typename T, MatrixType S=Sparse>
  class Matrix : public CooMatrix<T,S>
  {
    public:
    // Default Constructor, Standard Constructor and Assignment Constructor
    Matrix()
                : CooMatrix<T,S>() {}
    Matrix( const Index nrows, const Index ncols )
                : CooMatrix<T,S>( nrows, ncols ) {}
    void operator=( Matrix& rhs ) {}

    // Destructor
    ~Matrix() {};

    // C API Methods
    Info build( const std::vector<Index>& row_indices,
                const std::vector<Index>& col_indices,
                const std::vector<T>&     values,
                const Index nvals,
                const Matrix& mask,
                const BinaryOp& dup )
    {
      CooMatrix<T,S>::build( row_indices, col_indices, values, nvals, 
					mask, dup );
    }

    Info build( const std::vector<Index>& row_indices,
                const std::vector<Index>& col_indices,
                const std::vector<T>&     values,
                const Index nvals )
    {
			// Need this-> to refer to base class data members
			//std::cout << this->nrows_ << " " << this->ncols_ << std::endl;
      CooMatrix<T,S>::build( row_indices, col_indices, values, nvals );
    }

    Info build( const std::vector<T>& values )
    {
      CooMatrix<T,S>::build( values );
    }

		Info print()
		{
			CooMatrix<T,S>::print();
		}

    Info nnew( const Index nrows, const Index ncols ) {} // possibly unnecessary in C++
    Info dup( Matrix& C ) {}
    Info clear() {}
    Info nrows( Index& nrows ) const 
		{
		  CooMatrix<T,S>::nrows( nrows );
		}
    Info ncols( Index& ncols ) const
		{
			CooMatrix<T,S>::nrows( ncols );
		}
    Info nvals( Index& nvals ) const 
		{
			CooMatrix<T,S>::nvals( nvals );
		}

    private:
    // Data members that are same for all matrix formats
    //Index nrows_;
    //Index ncols_;
    //Index nvals_;
  };

} // backend
} // graphblas

#endif  // GRB_BACKEND_APSPIE_HPP
