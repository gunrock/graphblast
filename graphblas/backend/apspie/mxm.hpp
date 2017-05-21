#ifndef GRB_BACKEND_APSPIE_MXM_HPP
#define GRB_BACKEND_APSPIE_MXM_HPP

#include <iostream>

#include <cuda.h>

#include "graphblas/backend/apspie/Matrix.hpp"
#include "graphblas/backend/apspie/SparseMatrix.hpp"
#include "graphblas/backend/apspie/DenseMatrix.hpp"
#include "graphblas/backend/apspie/spmm.hpp"
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
			  		const Matrix<b>& B )
  {
		Storage A_storage, B_storage;
		A.get_storage( A_storage );
		B.get_storage( B_storage );

		// Decision tree:
		// a) Sp x Sp: SpGEMM (TODO)
		// b) Sp x De:   SpMM (DONE) 
		// c) De x Sp:   SpMM (TODO)
		// c) De x De:   GEMM (TODO)
		Info err;
		if( A_storage == Sparse && B_storage == Dense ) {
			err = C.set_storage( Dense );
			Index A_nvals;
			Index B_ncols;
			Index B_nvals;
			A.nvals( A_nvals );
			B.ncols( B_ncols );
			B.nvals( B_nvals );
			GpuTimer myspmm;
      #ifdef COL_MAJOR
			/*GpuTimer cusparse;
			cusparse.Start();
			err = cusparse_spmm( C.dense, op, A.sparse, B.dense );
			cusparse.Stop();
			float cusparse_flop = 2.0*A_nvals*B_ncols;
      std::cout << "cusparse mxm: " << cusparse.ElapsedMillis() << " ms, " <<
					cusparse_flop/cusparse.ElapsedMillis()/1000000.0 << " gflops\n";*/
			C.dense.clear();
			C.dense.allocate();
      #endif
			myspmm.Start();
			err = spmm( C.dense, op, A.sparse, B.dense );
			myspmm.Stop();
			float myspmm_flop   = 2.0*A_nvals*B_ncols;
      std::cout << "my spmm  mxm: " << myspmm.ElapsedMillis() << " ms, " <<
					myspmm_flop/myspmm.ElapsedMillis()/1000000.0 << " gflops\n";
    }
		return err;
	}
}  // backend
}  // graphblas

#endif  // GRB_BACKEND_APSPIE_MXM_HPP
