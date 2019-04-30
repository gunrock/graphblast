#ifndef GRAPHBLAS_ALGORITHM_COMMON_HPP_
#define GRAPHBLAS_ALGORITHM_COMMON_HPP_

#include <random>

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

template <typename T_in1, typename T_out = T_in1>
struct set_uniform_random {
  set_uniform_random() {
    seed_ = getEnv("GRB_SEED", 0);
    start_ = getEnv("GRB_UNIFORM_START", 0);
    end_ = getEnv("GRB_UNIFORM_END", 1);
    gen_ = std::default_random_engine(seed_);
    dist_ = std::uniform_int_distribution<int>(start_, end_);
  }

  inline GRB_HOST_DEVICE T_out operator()(T_in1 lhs) {
    return static_cast<T_out>(dist_(gen_));
  }

  int seed_;
  int start_;
  int end_;

  std::default_random_engine gen_;
  std::uniform_int_distribution<int> dist_;
};
}  // namespace graphblas

#endif  // GRAPHBLAS_ALGORITHM_COMMON_HPP_
