# GraphBLAST

GraphBLAST is a GPU implementation of [GraphBLAS](https://graphblas.org), an open standard for building blocks of graph algorithms. It gives data scientists without GPU programming experience the power to implement graph algorithms on the GPU. We are the leading graph framework in the following metrics:

- **Performance**: The first high-performance GPU implementation of GraphBLAS
- **Composable**: A library with building blocks for expressing most graph algorithms
- **Concise**: Single-source shortest path (SSSP) on GPU can be expressed in a mere 25 lines of code gets 3.68 GTEPS on a single NVIDIA V100 GPU (which would place 2nd in [Graph500](https://graph500.org/?page_id=384) for SSSP as of Oct. 2018)
- **Portable**: Algorithms implemented using API can be run on any GraphBLAS implementation
- **Innovative**: Combines state-of-the-art [graph optimizations](https://escholarship.org/uc/item/021076bn) from [Gunrock](https://github.com/gunrock/gunrock) with the automatic direction-optimization heuristic of [Ligra](https://github.com/jshun/ligra)

## Prerequisites

This software has been tested on the following dependencies:

* CUDA 9.1, 9.2
* Boost 1.58
* g++ 4.9.3, 5.4.0

Optional:

* CMake 3.11.1

## Install

A step by step series of instructions that tell you how to get a development environment running GraphBLAST.

1. First, you must download the software:

```
git clone --recursive https://github.com/gunrock/graphblast.git
```

2. The current library is set up as a header-only library. To install this library, copy the graphblas directory, its subdirectories and the specific platform subdirectory (sans the platform's test directories) to a location in your include path. However, there are 2 source files that need to be compiled with your program (`ext/moderngpu/src/mgpucontext.cu` and `ext/moderngpu/src/mgpuutil.cpp`).

We provide two build toolchains using `Makefile` and `CMake`. The user can choose either of them.

### Option 1: Using Makefile

```
cd graphblast
make -j16
```

### Option 2: Using CMake

```
cd graphblast
mkdir build
cd build
cmake ..
make -j$(nproc)
```

## Usage

Single Source-Shortest Path (Bellman-Ford SSSP) example (see the [graphblas/algorithm]() directory for more examples):

```c++
#include "graphblas/graphblas.hpp"

// Single-source shortest-path on adjacency matrix A from source s
graphblas::Info ssspSimple( Vector<float>*       v,
                            const Matrix<float>* A,
                            Index                s,
                            Descriptor*          desc ) {
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

  do {
    succ_last = succ;
    
    // v = v + v * A^T (do relaxation on distance vector v)
    graphblas::vxm<float, float, float, float>(&w, GrB_NULL, GrB_NULL,
        graphblas::MinimumPlusSemiring<float>(), v, A, desc);
    graphblas::eWiseAdd<float, float, float, float>(v, GrB_NULL, GrB_NULL,
        graphblas::MinimumPlusSemiring<float>(), v, &w, desc);

    // w = v < FLT_MAX (get all reachable vertices)
    graphblas::eWiseMult<float, float, float, float>(&w, GrB_NULL, GrB_NULL,
        graphblas::PlusLessSemiring<float>(), v, &zero, desc);

    // succ = reduce(w) (do reduction on all reachable distances)
    graphblas::reduce<float, float>(&succ, GrB_NULL,
        graphblas::PlusMonoid<float>(), &w, desc);
    iter++;

    // Loop until total reachable distance has converged
  } while (succ_last != succ);

  return GrB_SUCCESS;
}
```

## Concepts

The idea behind GraphBLAS is that four concepts can be used to implement many graph algorithms: vector, matrix, operation and semiring.

### Vector

A vector is a subset of vertices of some graph.

### Matrix

A matrix is the adjacency matrix of some graph.

### Operation

An operation is the memory access pattern common to most graph algorithms (equivalent Ligra terminology is shown in brackets):

- `mxv`: matrix-vector multiply (EdgeMap)
- `vxm`: vector-matrix multiply (EdgeMap)
- `mxm`: matrix-matrix multiply (multi-frontier EdgeMap)
- `eWiseAdd`: elementwise addition (VertexMap)
- `eWiseMult`: elementwise multiplication (VertexMap)

See [graphblas/operations.hpp](https://github.com/gunrock/graphblast/blob/master/graphblas/operations.hpp) for a complete list of operations.

### Semiring

A semiring is the computation on vertex and edge of the graph. In standard matrix multiplication the semiring used is the `(+, x)` arithmetic semiring. In GraphBLAS, when the semiring is applied to this operation, it represents the transformation that is required to transform the input vector into the output vector. What the `(+, x)` represent are:

- `x`: computation per edge, generates up to `num_edges` intermediate elements
- `+`: computation in the reduction of intermediates back down to a subset of vertices, up to `num_verts` elements

The most frequently used semirings (with their common usage in brackets) are:

- `PlusMultipliesSemiring`: arithmetic semiring (classical linear algebra)
- `LogicalOrAndSemiring`: Boolean semiring (graph connectivity)
- `MinimumPlusSemiring`: tropical min-plus semiring (shortest path)
- `MaximumMultipliesSemiring`: tropical max-times semiring (maximal independent set)

See [graphblas/stddef.hpp](https://github.com/gunrock/graphblast/blob/master/graphblas/stddef.hpp) for a complete list of semirings.

## Publications

1. Carl Yang, Aydın Buluç, and John D. Owens. **Design Principles for Sparse Matrix Multiplication on the GPU**. In *Proceedings of the 24th International European Conference on Parallel and Distributed Computing*, Euro-Par, pages 672-687, August 2018. Distinguished Paper and Best Artifact Award. [[DOI](http://dx.doi.org/10.1007/978-3-319-96983-1_48) | [http](https://escholarship.org/uc/item/5h35w3b7) | [slides](http://www.ece.ucdavis.edu/~ctcyang/pub/europar-slides2018.pdf)]

2. Carl Yang, Aydın Buluç, John D. Owens. **Implementing Push-Pull Efficiently in GraphBLAS**. In *Proceedings of the International Conference on Parallel Processing*, ICPP, pages 89:1-89:11, August 2018. [[DOI](http://dx.doi.org/10.1145/3225058.3225122) | [http](https://escholarship.org/uc/item/021076bn) | [slides](http://www.ece.ucdavis.edu/~ctcyang/pub/icpp-slides2018.pdf)]

3. Carl Yang, Yangzihao Wang, and John D. Owens. **Fast Sparse Matrix and Sparse Vector Multiplication Algorithm on the GPU. In Graph Algorithms Building Blocks**, In *Graph Algorithm Building Blocks*, GABB, pages 841–847, May 2015. [[DOI](http://dx.doi.org/10.1109/IPDPSW.2015.77) | [http](http://www.escholarship.org/uc/item/1rq9t3j3) | [slides](http://www.ece.ucdavis.edu/~ctcyang/pub/ipdpsw-slides2015.pdf)]

## Presentations

1. SC Doctoral Showcase 2018, **Linear Algebra is the Right Way to Think About Graphs**, November 2018. [[slides](http://www.ece.ucdavis.edu/~ctcyang/pub/sc-slides2018.pdf) | [poster](http://www.ece.ucdavis.edu/~ctcyang/pub/sc-poster2018.pdf)]

2. SIAM Minisymposium 2016, **Design Considerations for a GraphBLAS Compliant Graph Library on Clusters of GPUs**, July 2016. [[slides](http://www.ece.ucdavis.edu/~ctcyang/pub/siam-slides2016.pdf)]

## Other GraphBLAS Backends

If you are interested in other GraphBLAS backends, please check out these high-quality open-source implementations of GraphBLAS:

- [GraphBLAS Template Library: GBTL](https://github.com/cmu-sei/gbtl)
- [SuiteSparse GraphBLAS](http://faculty.cse.tamu.edu/davis/suitesparse.html)
- [IBM GraphBLAS](https://github.com/IBM/ibmgraphblas)
- [PostgreSQL GraphBLAS: pggraphblas](https://github.com/michelp/pggraphblas)

## Acknowledgments

We would like to thank the following people: [Yangzihao Wang](https://yzhwang.github.io) for teaching me how to write high-performance graph frameworks, [Yuechao Pan's](https://sites.google.com/site/panyuechao/home) for his valuable insights into BFS optimizations, [Scott McMillan](https://github.com/sei-smcmillan) for [his library](https://github.com/cmu-sei/gbtl) which inspired our code organization and teaching me how to implement the semiring object using macros, [Ben Johnson](https://github.com/bkj) for helping me catch several bugs, and [John D. Owens](https://www.ece.ucdavis.edu/~jowens/) and [Aydın Buluç](https://people.eecs.berkeley.edu/~aydin/) for their guidance and belief in me.

This work was funded by the DARPA HIVE program under AFRL Contract FA8650-18-2-7835, the DARPA XDATA program under AFRL Contract FA8750-13-C-0002, by NSF awards OAC-1740333, CCF-1629657, OCI-1032859, and CCF-1017399, by DARPA STTR award D14PC00023, by DARPA SBIR award W911NF-16-C-0020, Applied Mathematics program of the DOE Office of Advanced Scientific Computing Research under Contract No. DE-AC02-05CH11231, and by the Exascale Computing Project (17-SC-20-SC), a collaborative effort of the U.S. Department of Energy Office of Science and the National Nuclear Security Administration. 

## Copyright and Software License

GraphBLAST is copyright under the Regents of the University of California, 2015–2019. The library, examples, and all source code are released under [Apache 2.0](LICENSE.md).
