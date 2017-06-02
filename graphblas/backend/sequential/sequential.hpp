#ifndef GRB_BACKEND_SEQUENTIAL_SEQUENTIAL
#define GRB_BACKEND_SEQUENTIAL_SEQUENTIAL

// What is good replacement for CUDA_SAFE_CALL?
/*#define CUDA_SAFE_CALL_NO_SYNC(call) do {                               \
  cudaError err = call;                                                 \
  if( cudaSuccess != err) {                                             \
    fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",       \
                __FILE__, __LINE__, cudaGetErrorString( err) );         \
    exit(EXIT_FAILURE);                                                 \
    } } while (0)

#define CUDA_SAFE_CALL(call) do {                                       \
  CUDA_SAFE_CALL_NO_SYNC(call);                                         \
  cudaError err = cudaThreadSynchronize();                              \
  if( cudaSuccess != err) {                                             \
     fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",      \
                 __FILE__, __LINE__, cudaGetErrorString( err) );        \
     exit(EXIT_FAILURE);                                                \
  } } while (0)*/

#endif  // GRB_BACKEND_SEQUENTIAL_SEQUENTIAL
