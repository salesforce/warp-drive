#include <assert.h>

#ifndef CUDA_INCLUDES_UTIL_H_
#define CUDA_INCLUDES_UTIL_H_

// helper function to index a flattened C array with multi-dim index
// for example:
//   int shape[] = {5, 5, 5, 5};
//   int index[] = {3, 1, 2, 4};
//   int value = Array[get_flattened_array_index(index, shape, 4)];

__device__ int get_flattened_array_index(const int* index_arr, const int* dim_arr, const int dimensionality);

#endif // CUDA_INCLUDES_UTIL_H_

