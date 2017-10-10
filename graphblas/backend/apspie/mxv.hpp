#ifndef GRB_BACKEND_APSPIE_MXV_HPP
#define GRB_BACKEND_APSPIE_MXV_HPP

#include <iostream>

#include <cuda.h>

#include "graphblas/backend/apspie/Matrix.hpp"
#include "graphblas/backend/apspie/SparseMatrix.hpp"
#include "graphblas/backend/apspie/DenseMatrix.hpp"
#include "graphblas/backend/apspie/spmv.hpp"
#include "graphblas/types.hpp"

namespace graphblas
{
namespace backend
{

  /*template <typename c, typename a, typename b>
  Info mxv( Matrix<c>&       C,
            const Semiring&  op,
            const Matrix<a>& A,
            const Matrix<b>& B );*/

  template <typename c, typename m, typename a, typename b>
  Info mxv( Matrix<c>&        C,
            const Matrix<m>&  mask,
            const BinaryOp&   accum,
            const Semiring&   op,
            const Matrix<a>&  A,
            const Matrix<b>&  B,
            Descriptor&       desc ) 
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
    Desc_value mode;
    desc.get( GrB_MODE, mode );
    
    if( A_storage == GrB_SPARSE && B_storage == GrB_DENSE ) {
      if( C_storage == GrB_UNKNOWN )
        err = C.setStorage( GrB_DENSE );
      if( mode == GrB_FIXEDROW ) {
        //std::cout << "fixedrow\n";
        err = spmv( C.dense_, mask.sparse_, accum, op, A.sparse_, B.dense_, desc );
        err = C.dense_.setMajor( GrB_ROWMAJOR );
      } else if( mode == GrB_FIXEDCOL ) {
        //std::cout << "fixedcol\n";
        err = spmv( C.dense_, mask.sparse_, accum, op, A.sparse_, B.dense_, desc );
        err = C.dense_.setMajor( GrB_COLMAJOR );
      }
    }
    return err;
  }

  // For testing
  /*template <typename c, typename a, typename b>
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
      if( TA==0 && TB==0 && NT==1 )
        err = cusparse_spgemm2_compute( C.sparse, op, A.sparse, B.sparse );
      else
        err = cusparse_spgemm_compute( C.sparse, op, A.sparse, B.sparse );
    }
    return err;
  }*/

}  // backend
}  // graphblas

#endif  // GRB_BACKEND_APSPIE_MXV_HPP
