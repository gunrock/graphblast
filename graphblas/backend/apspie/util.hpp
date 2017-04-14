#ifndef GRB_BACKEND_APSPIE_UTIL_HPP
#define GRB_BACKEND_APSPIE_UTIL_HPP

template <typename T>
void printArrayDevice( const char* str, const T* array, int length=40 )
{
  if( length>40 ) length=40;

	// Allocate array on host
	T *temp = (T*) malloc(length*sizeof(T));
	CUDA_SAFE_CALL( cudaMemcpy( temp, array, length*sizeof(T), cudaMemcpyDeviceToHost ));
	printArray( str, temp, length );

	// Cleanup
	if( temp ) free( temp );
}

#endif  // GRB_BACKEND_APSPIE_UTIL_HPP
