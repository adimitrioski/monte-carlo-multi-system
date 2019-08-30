#ifndef GPU_HISTOGRAM_CUH_
#define GPU_HISTOGRAM_CUH_


#include "../histogram.cuh"

#include <thrust/sort.h>
#include <thrust/inner_product.h>
#include <thrust/binary_search.h>
#include <thrust/adjacent_difference.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/system/cuda/vector.h>
#include <thrust/system/cuda/execution_policy.h>

template <typename DataType, typename IndexType>
inline void gpu_sparse_histogram(
		thrust::cuda::vector<DataType> &data,
		thrust::cuda::vector<DataType> &histogram_values,
		thrust::cuda::vector<IndexType> &histogram_counts) {


	// sort data to bring equal elements together
	thrust::sort(thrust::cuda::par, data.begin(), data.end());


	// number of histogram bins is equal to number of unique values (assumes data.size() > 0)
	IndexType num_bins = thrust::inner_product(
			thrust::cuda::par,
			data.begin(), data.end() - 1,
	        data.begin() + 1,
	        IndexType(1),
	        thrust::plus<IndexType>(),
	        thrust::not_equal_to<DataType>());

	// resize histogram storage
    histogram_values.resize(num_bins);
	histogram_counts.resize(num_bins);

	// compact find the end of each bin of values
	thrust::reduce_by_key(
			thrust::cuda::par,
			data.begin(),
			data.end(),
	        thrust::constant_iterator<IndexType>(1),
	        histogram_values.begin(),
	        histogram_counts.begin());
}

#endif /* GPU_HISTOGRAM_CUH_ */
