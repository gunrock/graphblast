# Gunrock GraphBLAS on GPU (GunrockGrB)

[GraphBLAS](https://graphblas.org) is an open standard for building blocks of graph algorithms in the language of linear algebra. By virtue of it being an open standard, it gives data scientists who have no GPU programming experience the power to implement graph algorithms on the GPU. 

[Gunrock](https://github.com/gunrock/gunrock) is the fastest GPU graph framework in the world. Gunrock is featured on NVIDIA's [list of GPU Accelerated Libraries](https://developer.nvidia.com/gpu-accelerated-libraries) as the only non-NVIDIA library for GPU graph analytics. 

Our project seeks to combine the elegance of the GraphBLAS interface with the state-of-the-art, breakneck speed Gunrock provides. Our goal is to be:

- The first high-performance GPU implementation of GraphBLAS
- Possessing many Gunrock GPU optimizations for common graph operations
- A graph algorithm library containing commonly used graph algorithms
- Capable of generalized direction-optimization introduced by [Ligra](https://www.cs.cmu.edu/~jshun/ligra.pdf)

## Prerequisites

This software has been tested on the following dependencies:

* CUDA 8.0, 9.2
* Boost 1.58
* CMake 3.11.1
* g++ 4.9.3, 5.4.0

## Install

A step by step series of instructions that tell you have to get a development env running.

1. First, you must download the software:

```
git clone --recursive https://github.com/gunrock/gunrock-grb.git
```

2. The current library is set up as a header-only library. To install this library, copy the graphblas directory, its subdirectories and the specific platform subdirectory (sans the platform's test directories) to a location in your include path. However, there are 2 source files that need to be compiled with your program (`ext/moderngpu/src/mgpucontext.cu` and `ext/moderngpu/src/mgpuutil.cpp`).

We provide two sample build paths using `Makefile` and `CMake`.

### Using Makefile

```
cd gunrock-grb
make -j16
```

### Using CMake

```
cd gunrock-grb
mkdir build
cd build
cmake ..
make -j$(nproc)
```

## Concepts

This library is based on the concept that a graph traversal can be formulated as a sparse matrix-vector multiplication. GraphBLAS core principles are based on linear algebra operations, which describe the memory access pattern common to most graph algorithms:

- `mxv` (matrix-vector multiply)
- `mxm` (matrix-matrix multiply)
- `eWiseAdd` (elementwise addition)
- `eWiseMult` (elementwise multiplication)

As well, the other GraphBLAS core principle is the concept of generalized semirings, which means replacing the standard (+, x) of matrix multiplication with a different operation. These represent operations on vertices and edges of a graph. Together these two concepts---operation and semiring---can be used to implement many graph algorithms.

## Usage

Single Source-Shortest Path (Bellman-Ford SSSP) example (see the [graphblas/algorithm]() directory for more examples):

```c++
#include "graphblas/graphblas.hpp"

// Use float for now for both v and A
graphblas::Info sssp_simple( Vector<float>*       v,
                             const Matrix<float>* A,
                             Index                s,
                             Descriptor*          desc )
{
  // Get number of vertices
  graphblas::Index A_nrows;
  A->nrows(&A_nrows);

  // Distance vector (v)
  std::vector<graphblas::Index> indices(1, s);
  std::vector<float>  values(1, 0.f);
  v->build(&indices, &values, 1, GrB_NULL);

  // Buffer vector (w)
  graphblas::Vector<float> w(A_nrows);

  // Semiring zero vector (zero)
  graphblas::Vector<float> zero(A_nrows);
  zero.fill(std::numeric_limits<float>::max());

  // Initialize loop variables
  graphblas::Index iter = 1;
  float succ_last = 0.f;
  float succ = 1.f;

  do
  {
    succ_last = succ;
    
    // v = v + v * A^T (do relaxation on distance vector v)
    graphblas::vxm<float,float,float,float>(&w, GrB_NULL, GrB_NULL,
        MinimumPlusSemiring<float>(), v, A, desc);
    graphblas::eWiseAdd<float,float,float,float>(v, GrB_NULL, GrB_NULL,
        MinimumPlusSemiring<float>(), v, &w, desc);

    // w = v < FLT_MAX (get all reachable vertices)
    graphblas::eWiseMult<float, float, float, float>(&w, GrB_NULL, GrB_NULL,
        PlusLessSemiring<float>(), v, &zero, desc);

    // succ = reduce(w) (do reduction on all reachable distances)
    graphblas::reduce<float, float>(&succ, GrB_NULL, PlusMonoid<float>(), &w,
        desc);
    iter++;

    // Loop until total reachable distance has converged
  } while (succ_last != succ);

  return GrB_SUCCESS;
}
```

## Publications

1. Carl Yang, Aydın Buluç, and John D. Owens. **Design Principles for Sparse Matrix Multiplication on the GPU**. In *Proceedings of the 24th International European Conference on Parallel and Distributed Computing*, Euro-Par, pages 672-687, August 2018. Distinguished Paper and Best Artifact Award. [[DOI](http://dx.doi.org/10.1007/978-3-319-96983-1_48) | [http](https://escholarship.org/uc/item/5h35w3b7) | [slides](http://www.ece.ucdavis.edu/~ctcyang/pub/europar-slides2018.pdf)]

2. Carl Yang, Aydın Buluç, John D. Owens. **Implementing Push-Pull Efficiently in GraphBLAS**. In *Proceedings of the International Conference on Parallel Processing*, ICPP, pages 89:1-89:11, August 2018. [[DOI](http://dx.doi.org/10.1145/3225058.3225122) | [http](https://escholarship.org/uc/item/021076bn) | [slides](http://www.ece.ucdavis.edu/~ctcyang/pub/icpp-slides2018.pdf)]

3. Carl Yang, Yangzihao Wang, and John D. Owens. **Fast Sparse Matrix and Sparse Vector Multiplication Algorithm on the GPU. In Graph Algorithms Building Blocks**, In *Graph Algorithm Building Blocks*, GABB, pages 841–847, May 2015. [[DOI](http://dx.doi.org/10.1109/IPDPSW.2015.77) | [http](http://www.escholarship.org/uc/item/1rq9t3j3) | [slides](http://www.ece.ucdavis.edu/~ctcyang/pub/ipdpsw-slides2015.pdf)]

## Presentations

* SIAM Minisymposium 2016, **Design Considerations for a GraphBLAS Compliant Graph Library on Clusters of GPUs**, July 2016. [[slides](http://www.ece.ucdavis.edu/~ctcyang/pub/siam-slides2016.pdf)]

## Acknowledgments

We would like to thank the following people: [Yangzihao Wang](https://yzhwang.github.io) for teaching me the basics of graph frameworks, [Yuechao Pan's](https://sites.google.com/site/panyuechao/home) for his valuable insights into BFS optimizations, [Scott McMillan](https://github.com/sei-smcmillan) for [his library](https://github.com/cmu-sei/gbtl) which inspired our code organization, [Ben Johnson](https://github.com/bkj) for helping me catch many bugs, and [John D. Owens](https://www.ece.ucdavis.edu/~jowens/) and [Aydın Buluç](https://people.eecs.berkeley.edu/~aydin/) for their guidance and believing in me.

This work was funded by the DARPA HIVE program under AFRL Contract FA8650-18-2-7835, the DARPA XDATA program under AFRL Contract FA8750-13-C-0002, by NSF awards OAC-1740333, CCF-1629657, OCI-1032859, and CCF-1017399, by DARPA STTR award D14PC00023, by DARPA SBIR award W911NF-16-C-0020, Applied Mathematics program of the DOE Office of Advanced Scientific Computing Research under Contract No. DE-AC02-05CH11231, and by the Exascale Computing Project (17-SC-20-SC), a collaborative effort of the U.S. Department of Energy Office of Science and the National Nuclear Security Administration. 

## Copyright and Software License

GunrockGrB is copyright under the Regents of the University of California, 2015–2018. The library, examples, and all source code are released under [Apache 2.0](LICENSE.md).
