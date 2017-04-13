#ifndef GRB_GRAPHBLAS_HPP
#define GRB_GRAPHBLAS_HPP

#include <graphblas/backend.hpp>
#include <graphblas/Matrix.hpp>
#include <graphblas/mxm.hpp>
//#include <graphblas/types.hpp>

#define __GRB_BACKEND_HEADER <graphblas/backend/__GRB_BACKEND_ROOT/__GRB_BACKEND_ROOT.hpp>
#include __GRB_BACKEND_HEADER
#undef __GRB_BACKEND_HEADER

#endif  // GRB_GRAPHBLAS_HPP
