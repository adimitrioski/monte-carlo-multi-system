/*
 * server.h
 *
 *  Created on: Aug 30, 2019
 *      Author: andrej
 */

#ifndef SERVER_H_
#define SERVER_H_

#include "monte_carlo.cuh"
#include "util.h"

#include <functional>
#include <string>
#include <type_traits>

#include <cuda_runtime_api.h>
#include <cuda.h>

#include <pistache/http.h>
#include <pistache/router.h>
#include <pistache/endpoint.h>

using namespace Pistache;
using namespace std;

enum Status { OK, NOK };

enum RunOn {
	cpu,
	cpu_omp,
	cpu_tbb,
	gpu,
	undefined
};

inline RunOn resolveRunOn(string run_on) {
	if (run_on == "cpu") return RunOn::cpu;
	else if (run_on == "cpu_omp") return RunOn::cpu_omp;
	else if (run_on == "cpu_tbb") return RunOn::cpu_tbb;
	else if (run_on == "gpu") return RunOn::gpu;
	return RunOn::undefined;
}

inline Status call_cpu(
		function<MonteCarloResult(MonteCarloRequest)> f,
		string ran_on,
		Http::ResponseWriter &response) {

	MonteCarloResult result { };
	try {
		typename remove_reference<MonteCarloRequest>::type request;
		result = f(request);
		result.ran_on = ran_on;
	} catch (std::exception &e) {
		response.send(Http::Code::Internal_Server_Error, e.what());
		return Status::NOK;
	}

	string str_result = dump_monte_carlo_result(result);
	response.send(Http::Code::Ok, str_result);
	return Status::OK;
}

inline Status call_gpu(
		function<MonteCarloResult(MonteCarloRequest)> f,
		string ran_on,
		Http::ResponseWriter &response) {

	MonteCarloResult result { };
	try {
		typename remove_reference<MonteCarloRequest>::type request;
		result = f(request);
		result.ran_on = ran_on;
	} catch (std::exception &e) {
		cudaGetLastError();
		cudaDeviceReset();
		cudaSetDevice(0);
		response.send(Http::Code::Internal_Server_Error, e.what());
		return Status::NOK;
	}

	string str_result = dump_monte_carlo_result(result);
	response.send(Http::Code::Ok, str_result);
	return Status::OK;
}

#endif /* SERVER_H_ */
