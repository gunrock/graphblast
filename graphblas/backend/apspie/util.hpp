#ifndef GRB_BACKEND_APSPIE_UTIL_HPP
#define GRB_BACKEND_APSPIE_UTIL_HPP

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
