/*
 * monte_carlo.cuh
 *
 *  Created on: Jun 27, 2019
 *      Author: andrej
 */

#ifndef MONTE_CARLO_CUH_
#define MONTE_CARLO_CUH_

#include <thrust/functional.h>
#include <math.h>
#include <vector>
#include <string>

#include <thrust/system/cuda/vector.h>

struct MonteCarloRequest {

	unsigned long long int num_iterations;
	float expected_return;
	float volatility;
	unsigned int time_horizon;
	float starting_value;
	float annual_investment;
};

struct MonteCarloResult {

	std::vector<float> histogram_values;
	std::vector<unsigned int> histogram_counts;
	float mean;
	float standard_deviation;
	float max;
	float min;
	float simulation_time;
	unsigned long long int num_iterations;
	unsigned int time_horizon;
	std::string ran_on;
};

struct MonteCarloFunctor {

    const float expected_return;
    const float volatility;
    const unsigned int time_horizon;
    const float starting_value;
    const float annual_investment;

    MonteCarloFunctor(
    		float _expected_return,
			float _volatility,
			unsigned int _time_horizon,
			float _starting_value,
			float _annual_investment) :

				expected_return(_expected_return),
				volatility(_volatility),
				time_horizon(_time_horizon),
				starting_value(_starting_value),
				annual_investment(_annual_investment) {}
};

struct SubtractMeanAndSquareFromDataFunctor : thrust::unary_function<float, float> {

	const float mean;

	SubtractMeanAndSquareFromDataFunctor(float _mean) : mean(_mean) {}

	__host__ __device__ inline float operator()(const float value) const {
		return pow(value - mean, 2);
	}
};

inline MonteCarloResult fillMonteCarloResult(
		std::vector<float> &h_histogram_values,
		std::vector<unsigned int> &h_histogram_counts,
		float mean,
		float min,
		float max,
		float standard_deviation,
		unsigned long long int num_iterations,
		unsigned int time_horizon,
		float simulation_time) {

		MonteCarloResult monte_carlo_result = {};
		monte_carlo_result.histogram_values = h_histogram_values;
		monte_carlo_result.histogram_counts = h_histogram_counts;
		monte_carlo_result.standard_deviation = standard_deviation;
		monte_carlo_result.mean = mean;
		monte_carlo_result.max = max;
		monte_carlo_result.min = min;
		monte_carlo_result.simulation_time = simulation_time;
		monte_carlo_result.time_horizon = time_horizon;
		monte_carlo_result.num_iterations = num_iterations;

		return monte_carlo_result;
}

// CPU -------------------------------------- //

MonteCarloResult cpu_cpp_run_monte_carlo_simulation(MonteCarloRequest monte_carlo_request);
MonteCarloResult cpu_omp_run_monte_carlo_simulation(MonteCarloRequest monte_carlo_request);
MonteCarloResult cpu_tbb_run_monte_carlo_simulation(MonteCarloRequest monte_carlo_request);

// GPU -------------------------------------- //

MonteCarloResult gpu_run_monte_carlo_simulation(MonteCarloRequest monte_carlo_request);

#endif
