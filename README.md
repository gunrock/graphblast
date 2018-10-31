# Gunrock GraphBLAS on GPU (GunrockGrB)

[GraphBLAS](https://graphblas.org) is an open standard for building blocks of graph algorithms in the language of linear algebra. By virtue of it being an open standard, it gives data scientists who have no GPU programming experience the power to implement graph algorithms on the GPU. [Gunrock](https://github.com/gunrock/gunrock) is the fastest GPU graph framework in the world. Gunrock is featured on NVIDIA's [list of GPU Accelerated Libraries](https://developer.nvidia.com/gpu-accelerated-libraries) as the only non-NVIDIA library for GPU graph analytics. 

Our project seeks to combine the elegance of the GraphBLAS interface with the state-of-the-art, breakneck speed Gunrock provides. Our goal is to be:

- The first high-performance GPU implementation of GraphBLAS
- A graph algorithm library containing commonly used graph algorithms implemented with the GraphBLAS primitive operations
- Capable of direction-optimization in general as first demonstrated by [Ligra](https://www.cs.cmu.edu/~jshun/ligra.pdf)

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

2. Then, you must compile the software.

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

```
TODO(@ctcyang):
```

## Publications

1. Carl Yang, Aydın Buluç, and John D. Owens. **Design Principles for Sparse Matrix Multiplication on the GPU**. In *Proceedings of the 24th International European Conference on Parallel and Distributed Computing*, Euro-Par, pages 672-687, August 2018. Distinguished Paper and Best Artifact Award. [[DOI](http://dx.doi.org/10.1007/978-3-319-96983-1_48) | [http](https://escholarship.org/uc/item/5h35w3b7) | [slides](http://www.ece.ucdavis.edu/~ctcyang/pub/europar-slides2018.pdf)]

2. Carl Yang, Aydın Buluç, John D. Owens. **Implementing Push-Pull Efficiently in GraphBLAS**. In *Proceedings of the International Conference on Parallel Processing*, ICPP, pages 89:1-89:11, August 2018. [[DOI](http://dx.doi.org/10.1145/3225058.3225122) | [http](https://escholarship.org/uc/item/021076bn) | [slides](http://www.ece.ucdavis.edu/~ctcyang/pub/icpp-slides2018.pdf)]

3. Carl Yang, Yangzihao Wang, and John D. Owens. **Fast Sparse Matrix and Sparse Vector Multiplication Algorithm on the GPU. In Graph Algorithms Building Blocks**, In *Graph Algorithm Building Blocks*, GABB, pages 841–847, May 2015. [[DOI](http://dx.doi.org/10.1109/IPDPSW.2015.77) | [http](http://www.escholarship.org/uc/item/1rq9t3j3) | [slides](http://www.ece.ucdavis.edu/~ctcyang/pub/ipdpsw-slides2015.pdf)]

## Presentations

* SIAM Minisymposium 2016, **Design Considerations for a GraphBLAS Compliant Graph Library on Clusters of GPUs**, July 2016. [[slides](http://www.ece.ucdavis.edu/~ctcyang/pub/siam-slides2016.pdf)]

## Acknowledgments

We would like to thank the following people: [Yangzihao Wang](https://yzhwang.github.io) for teaching me the basics of graph frameworks, [Yuechao Pan's](https://sites.google.com/site/panyuechao/home) for his valuable insights into BFS optimizations without which this library would not have been possible, [Scott McMillan](https://github.com/sei-smcmillan) for [his library](https://github.com/cmu-sei/gbtl) which inspired our code organization, and [Ben Johnson](https://github.com/bkj) for helping me catch many bugs.

## Copyright and Software License

GunrockGrB is copyright under the Regents of the University of California, 2013–2018. The library, examples, and all source code are released under [Apache 2.0](LICENSE.md).
