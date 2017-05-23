#ifndef GRB_BACKEND_APSPIE_MXM_HPP
#define GRB_BACKEND_APSPIE_MXM_HPP

#include <iostream>

#include <cuda.h>

#include "graphblas/backend/apspie/Matrix.hpp"
#include "graphblas/backend/apspie/SparseMatrix.hpp"
#include "graphblas/backend/apspie/DenseMatrix.hpp"
#include "graphblas/backend/apspie/spmm.hpp"
#include "graphblas/backend/apspie/spgemm.hpp"
#include "graphblas/types.hpp"

namespace graphblas
{
namespace backend
{
	template <typename c, typename m, typename a, typename b>
  Info mxm( Matrix<c>&        C,
					  const Matrix<m>&  mask,
						const BinaryOp&   accum,
						const Semiring&   op,
						const Matrix<a>&  A,
						const Matrix<b>&  B,
						const Descriptor& desc ); 

  template <typename c, typename a, typename b>
  Info mxm( Matrix<c>&       C,
	  			  const Semiring&  op,
		  			const Matrix<a>& A,
			  		const Matrix<b>& B );

	// For testing
  template <typename c, typename a, typename b>
  Info mxm( Matrix<c>&       C,
	  			  const Semiring&  op,
		  			const Matrix<a>& A,
			  		const Matrix<b>& B,
				    const int TA,
				    const int TB,
				    const int NT,
				    const bool ROW_MAJOR )
  {
		Storage A_storage, B_storage, C_storage;
		A.getStorage( A_storage );
		B.getStorage( B_storage );
		C.getStorage( C_storage );

		// Decision tree:
		// a) Sp x Sp: SpGEMM (cusparse only)
		// b) Sp x De:   SpMM (DONE) 
		// c) De x Sp:   SpMM (TODO)
		// c) De x De:   GEMM (TODO)
		// TODO:
		// -currently have static Sp x Sp = Sp
		// -would like to have heuristic that decides when Sp x Sp = De
		Info err;
		if( A_storage == Sparse && B_storage == Sparse) {
      if( C_storage == Unknown )
				err = C.setStorage( Sparse );
			GpuTimer myspmm, cusparse;
			cusparse.Start();
			err = cusparse_spgemm( C.sparse, op, A.sparse, B.sparse );
			cusparse.Stop();
			std::cout << "cusparse, " << cusparse.ElapsedMillis() << "\n";
		} else if( A_storage == Sparse && B_storage == Dense ) {
			if( C_storage == Unknown )
			  err = C.setStorage( Dense );
			Index A_nvals;
			Index B_ncols;
			Index B_nvals;
			A.nvals( A_nvals );
			B.ncols( B_ncols );
			B.nvals( B_nvals );
			GpuTimer myspmm;
      if( !ROW_MAJOR ) {
			  GpuTimer cusparse;
			  cusparse.Start();
			  err = cusparse_spmm( C.dense, op, A.sparse, B.dense );
			  cusparse.Stop();
			  float cusparse_flop = 2.0*A_nvals*B_ncols;
        std::cout << "cusparse, " << cusparse.ElapsedMillis() << ", " <<
					cusparse_flop/cusparse.ElapsedMillis()/1000000.0 << "\n";
			  C.dense.clear();
			  C.dense.allocate();
			}
			myspmm.Start();
			err = spmm( C.dense, op, A.sparse, B.dense, TA, TB, NT, ROW_MAJOR );
			myspmm.Stop();
			float myspmm_flop   = 2.0*A_nvals*B_ncols;
      std::cout << "my, " << myspmm.ElapsedMillis() << ", " <<
					myspmm_flop/myspmm.ElapsedMillis()/1000000.0 << "\n";
    }
		return err;
	}

	// For testing
  template <typename c, typename a, typename b>
  Info mxmAnalyze( Matrix<c>&       C,
	  			         const Semiring&  op,
		  		         const Matrix<a>& A,
			  	         const Matrix<b>& B,
				           const int TA,
				           const int TB,
				           const int NT,
				           const bool ROW_MAJOR )
  {
		Storage A_storage, B_storage, C_storage;
		A.getStorage( A_storage );
		B.getStorage( B_storage );
		C.getStorage( C_storage );

		// Decision tree:
		// a) Sp x Sp: SpGEMM (cusparse only)
		// b) Sp x De:   SpMM (TODO) 
		// c) De x Sp:   SpMM (TODO)
		// c) De x De:   GEMM (TODO)
		Info err;
		if( A_storage == Sparse && B_storage == Sparse) {
      if( C_storage == Unknown )
				err = C.setStorage( Sparse );
			err = cusparse_spgemm_analyze( C.sparse, op, A.sparse, B.sparse );
    }
		return err;
	}

	// For testing
  template <typename c, typename a, typename b>
  Info mxmCompute( Matrix<c>&       C,
	  			         const Semiring&  op,
		  		         const Matrix<a>& A,
			  	         const Matrix<b>& B,
				           const int TA,
				           const int TB,
				           const int NT,
				           const bool ROW_MAJOR )
  {
		Storage A_storage, B_storage, C_storage;
		A.getStorage( A_storage );
		B.getStorage( B_storage );
		C.getStorage( C_storage );

		// Decision tree:
		// a) Sp x Sp: SpGEMM (cusparse only)
		// b) Sp x De:   SpMM (TODO) 
		// c) De x Sp:   SpMM (TODO)
		// c) De x De:   GEMM (TODO)
		Info err;
		if( A_storage == Sparse && B_storage == Sparse) {
      if( C_storage == Unknown )
				err = C.setStorage( Sparse );
			err = cusparse_spgemm_compute( C.sparse, op, A.sparse, B.sparse );
    }
		return err;
	}
}  // backend
}  // graphblas

#endif  // GRB_BACKEND_APSPIE_MXM_HPP
