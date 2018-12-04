#ifndef GRAPHBLAS_GRAPHBLAS_HPP_
#define GRAPHBLAS_GRAPHBLAS_HPP_

#include "graphblas/backend.hpp"
#include "graphblas/mmio.hpp"
#include "graphblas/types.hpp"
#include "graphblas/stddef.hpp"
#include "graphblas/util.hpp"
#include "graphblas/dimension.hpp"
#include "graphblas/Descriptor.hpp"
#include "graphblas/Vector.hpp"
#include "graphblas/Matrix.hpp"
#include "graphblas/operations.hpp"

#define __GRB_BACKEND_HEADER <graphblas/backend/__GRB_BACKEND_ROOT/__GRB_BACKEND_ROOT.hpp>
#include __GRB_BACKEND_HEADER
#undef __GRB_BACKEND_HEADER

#endif  // GRAPHBLAS_GRAPHBLAS_HPP_
