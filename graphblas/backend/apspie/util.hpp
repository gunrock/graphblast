#ifndef GRB_BACKEND_APSPIE_UTIL_HPP
#define GRB_BACKEND_APSPIE_UTIL_HPP

#include <cuda_runtime_api.h>
#include <cuda.h>

template <typename T>
void printArrayDevice( const char* str, const T* array, int length=40 )
{
  if( length>40 ) length=40;

  // Allocate array on host
  T *temp = (T*) malloc(length*sizeof(T));
  CUDA( cudaMemcpy( temp, array, length*sizeof(T), cudaMemcpyDeviceToHost ));
  printArray( str, temp, length );

  // Cleanup
  if( temp ) free( temp );
}

namespace graphblas
{
namespace backend
{
  void printMemory( const char* str )
  {
    size_t free, total;
    if( GrB_MEMORY )
    {
      CUDA( cudaMemGetInfo(&free, &total) );
      std::cout << str << ": " << free << " bytes left out of " << total << 
          " bytes\n";
    }
  }
}  // backend
	
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
}  // namespace graphblas

#endif  // GRB_BACKEND_APSPIE_UTIL_HPP
