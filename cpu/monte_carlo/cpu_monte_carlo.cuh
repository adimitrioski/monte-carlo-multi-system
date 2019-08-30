/*
 * cpu_monte_carlo.cu
 *
 *  Created on: Jun 27, 2019
 *      Author: andrej
 */

#ifndef CPU_MONTE_CARLO_CUH_
#define CPU_MONTE_CARLO_CUH_


#include "../../monte_carlo.cuh"

#include <vector>
#include <time.h>
#include <cuda.h>
#include <thrust/functional.h>
#include <random>
#include <chrono>


struct HostMonteCarloFunctor : MonteCarloFunctor, thrust::unary_function<unsigned long long int, float> {

	using MonteCarloFunctor::MonteCarloFunctor;

	__host__ inline float operator()(const unsigned long long int n) const {
		float sum = 0;
		float temp_value = starting_value;

		auto seed = std::chrono::system_clock::now().time_since_epoch().count() + n;
		std::mt19937_64 gen{seed};
		std::normal_distribution<> d{expected_return,volatility};

		for(unsigned int i = 0; i < time_horizon; i++)
		{
			temp_value = lround(temp_value * d(gen) + annual_investment);
			sum += temp_value;
		}
		return sum;
	}
};

#endif /* CPU_MONTE_CARLO_CUH_ */
