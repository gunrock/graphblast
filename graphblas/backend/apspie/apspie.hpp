#ifndef GRB_BACKEND_APSPIE_APSPIE
#define GRB_BACKEND_APSPIE_APSPIE

#define CUDA_NO_SYNC(call) do {                               \
  cudaError err = call;                                                 \
  if( cudaSuccess != err) {                                             \
    fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",       \
                __FILE__, __LINE__, cudaGetErrorString( err) );         \
    exit(EXIT_FAILURE);                                                 \
    } } while (0)

#define CUDA(call) do {                                       \
  CUDA_NO_SYNC(call);                                         \
  cudaError err = cudaThreadSynchronize();                              \
  if( cudaSuccess != err) {                                             \
     fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",      \
                 __FILE__, __LINE__, cudaGetErrorString( err) );        \
     exit(EXIT_FAILURE);                                                \
  } } while (0)

#endif  // GRB_BACKEND_APSPIE_APSPIE
