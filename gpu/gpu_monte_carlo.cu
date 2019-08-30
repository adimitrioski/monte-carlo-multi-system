/*
 * gpu_monte_carlo.cu
 *
 *  Created on: Jun 27, 2019
 *      Author: andrej
 */

#include "../histogram.cuh"
#include "../monte_carlo.cuh"

#include <curand_kernel.h>
#include <cuda.h>
#include <vector>

#include <thrust/system/cuda/vector.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/copy.h>
#include <thrust/extrema.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/counting_iterator.h>



struct DeviceMonteCarloFunctor : MonteCarloFunctor, thrust::unary_function<unsigned long long int, float> {

	using MonteCarloFunctor::MonteCarloFunctor;

    __device__ inline float operator()(const unsigned long long int n) const {
		float sum = 0;
		float temp_value = starting_value;

		unsigned int seed = static_cast<unsigned int>(clock()) + n;
		curandState s;
		curand_init(seed, 0, 0, &s);

		for(unsigned int i = 0; i < time_horizon; i++)
		{
			float normal = expected_return + (volatility * curand_normal(&s));
			temp_value = lround(temp_value * normal + annual_investment);
			sum += temp_value;
		}
		return sum;
	}
};

float gpu_cuda_calculate_mean(
		thrust::cuda::vector<float> &d_vector,
		unsigned long long int num_iterations) {

	float sum = thrust::reduce(thrust::cuda::par, d_vector.begin(), d_vector.end());
	return sum / num_iterations;
};

float gpu_cuda_get_min(
		thrust::cuda::vector<float> &d_ending_values) {

	thrust::cuda::vector<float>::iterator min_iterator = thrust::min_element(thrust::cuda::par, d_ending_values.begin(), d_ending_values.end());
	return *min_iterator;
};

float gpu_cuda_get_max(
		thrust::cuda::vector<float> &d_ending_values) {

	thrust::cuda::vector<float>::iterator max_iterator = thrust::max_element(thrust::cuda::par, d_ending_values.begin(), d_ending_values.end());
    return *max_iterator;
};

float gpu_cuda_calculate_standard_deviation(
		thrust::cuda::vector<float> &d_ending_values,
		float mean,
		unsigned long long int num_iterations) {


	float sum = thrust::reduce(
			thrust::cuda::par,
			thrust::make_transform_iterator(d_ending_values.begin(), SubtractMeanAndSquareFromDataFunctor(mean)),
			thrust::make_transform_iterator(d_ending_values.end(), SubtractMeanAndSquareFromDataFunctor(mean)));

	return sqrt(sum / num_iterations);
};

MonteCarloResult gpu_run_monte_carlo_simulation(MonteCarloRequest monte_carlo_request) {

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	thrust::cuda::vector<float> d_ending_values(monte_carlo_request.num_iterations);

	cudaEventRecord(start);

	thrust::transform(
			thrust::cuda::par,
			thrust::counting_iterator<unsigned long long int>(0),
		    thrust::counting_iterator<unsigned long long int>(monte_carlo_request.num_iterations),
			d_ending_values.begin(),
			DeviceMonteCarloFunctor(
					monte_carlo_request.expected_return,
					monte_carlo_request.volatility,
					monte_carlo_request.time_horizon,
					monte_carlo_request.starting_value,
					monte_carlo_request.annual_investment));

	float mean = gpu_cuda_calculate_mean(d_ending_values, monte_carlo_request.num_iterations);
	float min = gpu_cuda_get_min(d_ending_values);
	float max = gpu_cuda_get_max(d_ending_values);
	float standard_deviation = gpu_cuda_calculate_standard_deviation(d_ending_values, mean, monte_carlo_request.num_iterations);

	thrust::cuda::vector<float> d_histogram_values;
	thrust::cuda::vector<unsigned int> d_histogram_counts;
	gpu_sparse_histogram(d_ending_values, d_histogram_values, d_histogram_counts);

	std::vector<float> histogram_values(d_histogram_values.size());
	std::vector<unsigned int> histogram_counts(d_histogram_counts.size());
	thrust::copy(d_histogram_values.begin(), d_histogram_values.end(), histogram_values.begin());
	thrust::copy(d_histogram_counts.begin(), d_histogram_counts.end(), histogram_counts.begin());

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float simulation_time = 0;
	cudaEventElapsedTime(&simulation_time, start, stop);

	return fillMonteCarloResult(
				histogram_values,
				histogram_counts,
				mean,
				min,
				max,
				standard_deviation,
				monte_carlo_request.num_iterations,
				monte_carlo_request.time_horizon,
				simulation_time);
};
