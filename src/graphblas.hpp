#ifdef GRAPHBLAS_HPP
#define GRAPHBLAS_HPP

#include <backend.hpp>
#include <Matrix.hpp>

#define __GRB_BACKEND_HEADER <backend/__GRB_BACKEND_ROOT/__GRB_BACKEND_ROOT.hpp>
#include __GRB_BACKEND_HEADER
#undef __GRB_BACKEND_HEADER

#endif  // GRAPHBLAS_HPP
