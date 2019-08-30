#include "../../histogram.cuh"
#include "../../monte_carlo.cuh"
#include "cpu_monte_carlo.cuh"
#include <tbb/tick_count.h>
#include <math.h>
#include <vector>
#include <thrust/extrema.h>
#include <thrust/reduce.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/system/tbb/execution_policy.h>


inline float calculate_mean(
		std::vector<float> &h_vector,
		unsigned long long int num_iterations) {

	float sum = thrust::reduce(thrust::tbb::par, h_vector.begin(), h_vector.end());
	return sum / num_iterations;
};

inline float get_min(
		std::vector<float> &h_ending_values) {

	std::vector<float>::iterator min_iterator = thrust::min_element(thrust::tbb::par, h_ending_values.begin(), h_ending_values.end());
	return *min_iterator;
}

inline float get_max(
		std::vector<float> &h_ending_values) {

	std::vector<float>::iterator max_iterator = thrust::max_element(thrust::tbb::par, h_ending_values.begin(), h_ending_values.end());
	return *max_iterator;
};

inline float calculate_standard_deviation(
		std::vector<float> &h_ending_values,
		float mean,
		unsigned long long int num_iterations) {

	float sum = thrust::reduce(
			thrust::tbb::par,
			thrust::make_transform_iterator(h_ending_values.begin(), SubtractMeanAndSquareFromDataFunctor(mean)),
			thrust::make_transform_iterator(h_ending_values.end(), SubtractMeanAndSquareFromDataFunctor(mean)));

	return sqrt(sum / num_iterations);
};

MonteCarloResult cpu_tbb_run_monte_carlo_simulation(
		MonteCarloRequest monte_carlo_request) {

	std::vector<float> h_ending_values(monte_carlo_request.num_iterations);

	tbb::tick_count start = tbb::tick_count::now();

	thrust::transform(
			thrust::tbb::par,
			thrust::counting_iterator<unsigned long long int>(0),
			thrust::counting_iterator<unsigned long long int>(monte_carlo_request.num_iterations),
			h_ending_values.begin(),
			HostMonteCarloFunctor(
					monte_carlo_request.expected_return,
					monte_carlo_request.volatility,
					monte_carlo_request.time_horizon,
					monte_carlo_request.starting_value,
					monte_carlo_request.annual_investment));

	float mean = calculate_mean(h_ending_values, monte_carlo_request.num_iterations);
	float min = get_min(h_ending_values);
	float max = get_max(h_ending_values);
	float standard_deviation = calculate_standard_deviation(h_ending_values, mean, monte_carlo_request.num_iterations);

	std::vector<float> h_histogram_values;
	std::vector<unsigned int> h_histogram_counts;
	cpu_tbb_sparse_histogram(h_ending_values, h_histogram_values, h_histogram_counts);

	tbb::tick_count end = tbb::tick_count::now();

	float simulation_time = (end - start).seconds() * 1000;

	return fillMonteCarloResult(
			h_histogram_values,
			h_histogram_counts,
			mean,
			min,
			max,
			standard_deviation,
			monte_carlo_request.num_iterations,
			monte_carlo_request.time_horizon,
			simulation_time);
};
