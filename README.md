# Design Principles for Sparse Matrix Multiplication on the GPU: Sparse Matrix-Dense Matrix Multiplication

by Carl Yang, Aydin Buluc, John D. Owens

Accepted as Distinguished Paper at EuroPar 2018

## Abstract

We implement two novel algorithms for sparse-matrix dense- matrix multiplication (SpMM) on the GPU. Our algorithms expect 
the sparse input in the popular compressed-sparse-row (CSR) format and thus do not require expensive format conversion. While
previous SpMM work concentrates on thread-level parallelism, we additionally focus on latency hiding with instruction-level 
parallelism and load-balancing. We show, both theoretically and experimentally, that the proposed SpMM is a better fit for 
the GPU than previous approaches. We identify a key memory access pattern that allows efficient access into both input and 
output matrices that is crucial to getting excellent performance on SpMM. By combining these two ingredients—(i) merge-based 
load-balancing and (ii) row-major coalesced memory access—we demonstrate a 3.6× peak speedup and a 23.5% geomean speedup over 
state-of-the-art SpMM implementations on real-world datasets.


![](http://wwwimages.adobe.com/content/dam/acom/en/legal/images/badges/Adobe_PDF_file_icon_32x32.png) [spmm-europar18-preprint.pdf](https://github.com/owensgroup/GraphBLAS/raw/europar/spmm-europar18-preprint.pdf)


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing 
purposes.

### Prerequisites

This software has been tested on the following dependencies:

* CUDA 8.0
* Boost 1.58
* CMake 3.11.1
* g++ 4.9.3
* ModernGPU 1.1

---

### CUDA 8.0

If CUDA 8.0 is not already installed on your system, you will need to download CUDA 8.0 
[here](https://developer.nvidia.com/cuda-80-ga2-download-archive). Follow the onscreen instructions and select the operating system and vendor that suits your needs. Download the 1.4GB file. You do not need to download the optional Patch 2.

After generating a download link, the commands I typed were the following:

```
wget https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda_8.0.61_375.26_linux-run
chmod +x cuda_8.0.61_375.26_linux-run
sudo ./cuda_8.0.61_375.26_linux-run
```

You will need to select:
```
graphics driver: yes
OpenGL: yes
nvidia-xconfig: no
CUDA 8.0: yes
symbolic link: yes
CUDA samples: yes
```

Once installation has finished, check that your installation has completed by typing:

```
nvidia-smi
```

If installation was successful, you should be able to see information about your GPU printed onscreen. 
Check that the right information has been added to your system path by typing:

```
vi ~/.bashrc
```

If not already present, you should append to the bottom:

```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-8.0/lib64
export CUDA_HOME=/usr/local/cuda-8.0
```

Additional instructions on installing CUDA can be found 
[here](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#runfile).

### Boost 1.58

You will need to install or compile Boost 1.58 program options using the same compiler as you do our software. 
To only install Boost program options, type:

```
wget http://sourceforge.net/projects/boost/files/boost/1.58.0/boost_1_58_0.tar.gz
tar -xvzf boost_1_58_0.tar.gz
cd boost_1_58_0
./bootstrap.sh --prefix=path/to/installation/prefix
./b2 --with-program_options
```

### CMake 3.11.1

If not already installed, you will need to install CMake by typing:

```
sudo apt-get install cmake
```

### g++ 4.9.3

You will need g++-4.9. Install by typing:

```
sudo apt-get install gcc-4.9 g++-4.9
```

### ModernGPU 1.1

This excellent software by Sean Baxter will be automatically downloaded as a Git submodule.

## Installing

A step by step series of examples that tell you have to get a development env running.

1. First, we must download the software:

```
git clone --recursive https://github.com/owensgroup/GraphBLAS.git
git checkout europar
cd ext/moderngpu
git checkout europar
cd ../../
```

2. Also, we must compile a dependency.

```
cd ext/merge-spmv
make gpu_spmv sm=350
cd ../../
```

3. Then, we must compile the software.

```
cmake .
make -j16
```

4. Next, we must download the datasets. In order of increasing size, they are listed below and can be downloaded 
automatically.

Small - 10 small row length matrices (400MB)
```
cd dataset/europar/lowd
make
cd ../../../
```

Large - 10 large row length matrices (1650MB)
```
cd dataset/europar/highd
make
cd ../../../
```

Super large - 172 matrices (4000MB)
```
cd dataset/europar/large
sh DownloadFigure6.sh
sh Extract.sh
cd ../../../
```

5. The figures in the paper can be reconstructed by typing:

```
sh Figure1a-spmm-4.sh
sh Figure1a-spmv.sh
sh Figure1b.sh
sh Figure5a.sh
sh Figure5b.sh
sh Figure6.sh
```

## Authors

* **Carl Yang**
* **Aydin Buluc**
* **John D. Owens**

## License

This project is licensed under the Apache License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* A big thanks to Sean Baxter's [ModernGPU library](https://github.com/moderngpu/moderngpu) and Duane Merrill's 
[Merge-based SpMV paper](https://github.com/dumerrill/merge-spmv/raw/master/merge-based-spmv-sc16-preprint.pdf) and 
[code](https://github.com/dumerrill/merge-spmv) which inspired our work.
