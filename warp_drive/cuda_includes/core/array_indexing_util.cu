#include "array_indexing_util.h"

__device__ int get_flattened_array_index(const int* index_arr, const int* dim_arr, const int dimensionality) {

  int res = 0;
  int multiplier = 1;

  for(int i=dimensionality-1; i>=0; i--){
      res += index_arr[i] * multiplier;
      multiplier *= dim_arr[i];
  }

  return res;
}