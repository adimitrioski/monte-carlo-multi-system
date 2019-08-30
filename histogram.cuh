#ifndef HISTOGRAM_CUH_
#define HISTOGRAM_CUH_

#include "cpu/histogram/cpu_cpp_histogram.cuh"
#include "cpu/histogram/cpu_omp_histogram.cuh"
#include "cpu/histogram/cpu_tbb_histogram.cuh"

#include "gpu/gpu_histogram.cuh"
#include <vector>
#include <thrust/system/cuda/vector.h>

// From: https://github.com/thrust/thrust/blob/master/examples/histogram.cu
// and slightly modified for this example

// This example illustrates several methods for computing a
// histogram [1] with Thrust.  We consider standard "dense"
// histograms, where some bins may have zero entries, as well
// as "sparse" histograms, where only the nonzero bins are
// stored.  For example, histograms for the data set
//    [2 1 0 0 2 2 1 1 1 1 4]
// which contains 2 zeros, 5 ones, and 3 twos and 1 four, is
//    [2 5 3 0 1]
// using the dense method and
//    [(0,2), (1,5), (2,3), (4,1)]
// using the sparse method. Since there are no threes, the
// sparse histogram representation does not contain a bin
// for that value.
//
// Note that we choose to store the sparse histogram in two
// separate arrays, one array of keys and one array of bin counts,
//    [0 1 2 4] - keys
//    [2 5 3 1] - bin counts
// This "structure of arrays" format is generally faster and
// more convenient to process than the alternative "array
// of structures" layout.
//
// The best histogramming methods depends on the application.
// If the number of bins is relatively small compared to the
// input size, then the binary search-based dense histogram
// method is probably best.  If the number of bins is comparable
// to the input size, then the reduce_by_key-based sparse method
// ought to be faster.  When in doubt, try both and see which
// is fastest.
//
// [1] http://en.wikipedia.org/wiki/Histogram


// sparse histogram using reduce_by_key

// CPU -------------------------------------- //

template <typename DataType, typename IndexType>
void cpu_cpp_sparse_histogram(
		std::vector<DataType> &data,
        std::vector<DataType> &histogram_values,
        std::vector<IndexType> &histogram_counts);

template <typename DataType, typename IndexType>
void cpu_omp_sparse_histogram(
		std::vector<DataType> &data,
        std::vector<DataType> &histogram_values,
        std::vector<IndexType> &histogram_counts);

template <typename DataType, typename IndexType>
void cpu_tbb_sparse_histogram(
		std::vector<DataType> &data,
        std::vector<DataType> &histogram_values,
        std::vector<IndexType> &histogram_counts);

// GPU -------------------------------------- //

template <typename DataType, typename IndexType>
void gpu_sparse_histogram(
		thrust::cuda::vector<DataType> &data,
        thrust::cuda::vector<DataType> &histogram_values,
        thrust::cuda::vector<IndexType> &histogram_counts);

#endif /* HISTOGRAM_CUH_ */
