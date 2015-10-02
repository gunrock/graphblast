#include <vector>
#include <iostream>
#include <tuple>

  typedef uint64_t Index;
template<typename T>
  using Vector = std::vector<T>;

  template<typename T>
  class Tuple
  {
    public:
      std::vector<Index> I;
      std::vector<Index> J;
      std::vector<T>     V;
  };

int main() {
  int i, j;
  std::vector<Index> rowind = {1, 2, 3};
  std::vector<Index> A;
  A=rowind;

  Tuple<int> blah;
  blah.I = rowind;

  for( i=0; i<blah.I.size(); i++ ) {
    std::cout << blah.I[i] << std::endl;
  }
  return 0;
}
