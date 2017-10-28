#ifndef GRB_BACKEND_APSPIE_UTIL_HPP
#define GRB_BACKEND_APSPIE_UTIL_HPP

#include "graphblas/backend/apspie/apspie.hpp"

namespace graphblas
{

template <typename T>
void printDevice( const char* str, const T* array, int length=40 )
{
  //if( length>40 ) length=40;

  // Allocate array on host
  T *temp = (T*) malloc(length*sizeof(T));
  CUDA( cudaMemcpy( temp, array, length*sizeof(T), cudaMemcpyDeviceToHost ));
  printArray( str, temp, length );

  // Cleanup
  if( temp ) free( temp );
}

template <typename T>
void printCode( const char* str, const T* array, int length )
{
  // Allocate array on host
  T *temp = (T*) malloc(length*sizeof(T));
  CUDA( cudaMemcpy( temp, array, length*sizeof(T), cudaMemcpyDeviceToHost ));
  
  // Traverse array, printing out move
  // Followed by manual reordering:
  // 1) For each dst block, find final move to that block. Mark its src.
  // 2) For all moves to that dst block, change dst to src.
  for( Index i=length-1; i>=0; i-- )
	  if( temp[i]!=i )
      printf("  count += testMerge( state, %d, %d, true );\n", temp[i], i );

  // Cleanup
  if( temp ) free( temp );
}

struct GpuTimer
{
  cudaEvent_t start;
  cudaEvent_t stop;

  GpuTimer()
  {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
  }

  ~GpuTimer()
  {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  void Start()
  {
    cudaEventRecord(start, 0);
  }

  void Stop()
  {
    cudaEventRecord(stop, 0);
  }

  float ElapsedMillis()
  {
    float elapsed;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    return elapsed;
  }
};

}  // graphblas

#endif  // GRB_BACKEND_APSPIE_UTIL_HPP
