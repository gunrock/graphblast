# Gunrock GraphBLAS on GPU (GunrockGrB)

[GraphBLAS](https://graphblas.org) is an open standard for building blocks of graph algorithms in the language of linear algebra. By virtue of it being an open standard, it gives data scientists who have no GPU programming experience the power to implement graph algorithms on the GPU. [Gunrock](https://github.com/gunrock/gunrock) is the fastest GPU graph framework in the world. Gunrock is featured on NVIDIA's [list of GPU Accelerated Libraries](https://developer.nvidia.com/gpu-accelerated-libraries) as the only non-NVIDIA library for GPU graph analytics. 

Our project seeks to combine the elegance of the GraphBLAS interface with the state-of-the-art, breakneck speed Gunrock provides. Our goal is to be:

- The first high-performance GPU implementation of GraphBLAS
- A graph algorithm library containing commonly used graph algorithms implemented with the GraphBLAS primitive operations

## Prerequisites

This software has been tested on the following dependencies:

* CUDA 8.0, 9.2
* Boost 1.58
* CMake 3.11.1
* g++ 4.9.3, 5.4.0

## Installation

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

## Publications

Carl Yang, Aydın Buluç, and John D. Owens. Design Principles for Sparse Matrix Multiplication on the GPU. In Proceedings of the 24th International European Conference on Parallel and Distributed Computing, Euro-Par, pages 672-687, August 2018. Distinguished Paper and Best Artifact Award. [[DOI](http://dx.doi.org/10.1007/978-3-319-96983-1_48) | [http](https://escholarship.org/uc/item/5h35w3b7)]

Carl Yang, Aydın Buluç, John D. Owens. **Implementing Push-Pull Efficiently in GraphBLAS**. In *International Conference on Parallel Processing*, ICPP, pages 89:1-89:11, August 2018. [[DOI](http://dx.doi.org/10.1145/3225058.3225122) | [http](https://escholarship.org/uc/item/021076bn)]

Carl Yang, Yangzihao Wang, and John D. Owens. **Fast Sparse Matrix and Sparse Vector Multiplication Algorithm on the GPU. In Graph Algorithms Building Blocks**, *Graph Algorithm Building Blocks*, GABB, pages 841–847, May 2015. [[DOI](http://dx.doi.org/10.1109/IPDPSW.2015.77) | [http](http://www.escholarship.org/uc/item/1rq9t3j3)]
