#ifndef GRAPHBLAS_ALGORITHM_COMMON_HPP_
#define GRAPHBLAS_ALGORITHM_COMMON_HPP_

namespace graphblas {

template <typename T_in1, typename T_out = T_in1>
struct set_random {
  set_random() {
    seed_ = getEnv("GRB_SEED", 0);
    srand(seed_);
  }

  inline GRB_HOST_DEVICE T_out operator()(T_in1 lhs) {
    return rand();
  }

  int seed_;
};
}  // namespace graphblas

#endif  // GRAPHBLAS_ALGORITHM_COMMON_HPP_
