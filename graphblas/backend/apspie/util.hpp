#ifndef GRB_BACKEND_APSPIE_UTIL_HPP
#define GRB_BACKEND_APSPIE_UTIL_HPP

#define CUDA_SAFE_CALL_NO_SYNC(call) do {                               \
  cudaError err = call;                                                 \
  if( cudaSuccess != err) {                                             \
    fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",       \
                __FILE__, __LINE__, cudaGetErrorString( err) );         \
    exit(EXIT_FAILURE);                                                 \
    } } while (0)

#define CUDA_CALL(call) do {                                            \
  CUDA_SAFE_CALL_NO_SYNC(call);                                         \
  cudaError err = cudaThreadSynchronize();                              \
  if( cudaSuccess != err) {                                             \
     fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",      \
                 __FILE__, __LINE__, cudaGetErrorString( err) );        \
     exit(EXIT_FAILURE);                                                \
  } } while (0)

#include <cstdlib>

namespace graphblas
{
namespace backend
{
  void printMemory( const char* str )
  {
    size_t free, total;
    if( GrB_MEMORY )
    {
      CUDA_CALL( cudaMemGetInfo(&free, &total) );
      std::cout << str << ": " << free << " bytes left out of " << total << 
          " bytes\n";
    }
  }

  template <typename T>
  void printDevice( const char* str, const T* array, int length=40 )
  {
    //if( length>40 ) length=40;

    // Allocate array on host
    T *temp = (T*) malloc(length*sizeof(T));
    CUDA_CALL( cudaMemcpy( temp, array, length*sizeof(T), cudaMemcpyDeviceToHost ));
    printArray( str, temp, length );

    // Cleanup
    if( temp ) free( temp );
  }

  template <typename T>
  void printCode( const char* str, const T* array, int length )
  {
    // Allocate array on host
    T *temp = (T*) malloc(length*sizeof(T));
    CUDA_CALL( cudaMemcpy( temp, array, length*sizeof(T), cudaMemcpyDeviceToHost ));
    
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

  void printState( bool use_mask, bool use_accum, bool use_scmp, bool use_repl, 
      bool use_tran )
  {
    std::cout << "Mask: " << use_mask  << std::endl;
    std::cout << "Accum:" << use_accum << std::endl;
    std::cout << "SCMP: " << use_scmp  << std::endl;
    std::cout << "Repl: " << use_repl  << std::endl;
    std::cout << "Tran: " << use_tran  << std::endl;
  }

  template<typename T>
  inline T getEnv(const char *key, T default_val) {
    const char *val = std::getenv(key);
    if (val == NULL) {
      return default_val;
    } else {
      return static_cast<T>(atoi(val));
    }
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

}  // backend
}  // graphblas

#endif  // GRB_BACKEND_APSPIE_UTIL_HPP
