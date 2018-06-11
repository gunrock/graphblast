#include <vector>
#include <algorithm>
#include <random>
#include <set>
#include <iostream>

std::set<int> pickSet(int N, int k, std::mt19937& gen)
{
  std::set<int> elems;
  for (int r = N - k; r < N; ++r) {
    int v = std::uniform_int_distribution<>(0, r)(gen);

    // there are two cases.
    // v is not in candidates ==> add it
    // v is in candidates ==> well, r is definitely not, because
    // this is the first iteration in the loop that we could've
    // picked something that big.

    if (!elems.insert(v).second) {
      elems.insert(r);
    }   
  }
  return elems;
}

std::vector<int> pick(int N, int k, int row, std::mt19937& gen)
{
  std::set<int> elems = pickSet(N, k, gen);

  // Ensure no self-loops
  // If there is self-loop, find first zero element in row and set it to nnz
  if (elems.find(row) != elems.end())
  {
    elems.erase(row);
    for (unsigned i = 0; i < N; ++i)
    {
      if (i != row && elems.find(i) == elems.end())
      elems.insert(i);
    }
  }

  std::vector<int> result(elems.begin(), elems.end());
  return result;
}

int main( int argc, char** argv )
{
  if (argc != 4)
  {
    std::cout << "Usage: ./gen_matrix n_rows n_cols n_nnz_per_row\n";
    std::cout << "       ./gen_matrix 10 10 5\n";
    std::cout << "x x x x x 0 0 0 0 0\n";
    std::cout << "0 x x x 0 0 0 x x 0\n";
    std::cout << "x x 0 0 0 0 0 x x x\n";
    std::cout << "x x 0 x x 0 0 x 0 0\n";
    std::cout << "x x x x 0 x 0 0 0 0\n";
    std::cout << "x 0 x x 0 x 0 x 0 0\n";
    std::cout << "x x 0 x x 0 0 x 0 0\n";
    std::cout << "0 x x x 0 0 0 x x 0\n";
    std::cout << "x x 0 0 0 0 0 x x x\n";
    std::cout << "x x 0 0 0 0 x 0 x x\n";
  }

  int n_rows = atoi(argv[1]);
  int n_cols = atoi(argv[2]);
  int n_nnzs = atoi(argv[3]);

  std::random_device rd;
  std::mt19937 gen(0);

  std::cout << "%%MatrixMarket matrix coordinate pattern general\n";
  std::cout << n_rows << " " << n_cols << " " << n_nnzs*n_rows << std::endl;

  for (int i = 0; i < n_rows; ++i)
  {
    std::vector<int> pick_row = pick(n_cols, n_nnzs, i, gen);
    for (unsigned j = 0; j < pick_row.size(); ++j)
    {
      std::cout << pick_row[j] << " " << i << std::endl;
    }
  }
}
