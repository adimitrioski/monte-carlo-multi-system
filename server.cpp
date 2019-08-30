/*
 * server.cpp
 *
 *  Created on: Jun 27, 2019
 *      Author: andrej
 */

#include "util.h"
#include "monte_carlo.cuh"
#include "server.h"

#include <functional>

#include <pistache/http.h>
#include <pistache/router.h>
#include <pistache/endpoint.h>

using namespace Pistache;
using namespace std;

class App: public Http::Endpoint {

private:
	Rest::Router router;

	string BASE_URL = "/api/monte_carlo/";

	string CPU_CPP_URL = BASE_URL + "/cpu";
	string CPU_OMP_URL = BASE_URL + "/cpu_omp";
	string CPU_TBB_URL = BASE_URL + "/cpu_tbb";

	string GPU_URL = BASE_URL + "/gpu";

public:
	App(Address addr) :
			Http::Endpoint(addr) {

		auto opts = Http::Endpoint::options().flags(Tcp::Options::ReuseAddr);

		init(opts);

		Rest::Routes::Post(router, CPU_CPP_URL,
				[=](const Rest::Request &request,
						Http::ResponseWriter response) {

					post_enable_cors(response);

					MonteCarloRequest monte_carlo_request =
							parse_monte_carlo_request_json(request);

					function<MonteCarloResult(MonteCarloRequest)> cpu_cpp_func = std::bind(cpu_cpp_run_monte_carlo_simulation, monte_carlo_request);
					Status status = call_cpu(cpu_cpp_func, "CPU", response);

					if (status == Status::NOK) {
						return Rest::Route::Result::Failure;
					}
					return Rest::Route::Result::Ok;
				});

		Rest::Routes::Post(router, CPU_OMP_URL,
				[=](const Rest::Request &request,
						Http::ResponseWriter response) {

					post_enable_cors(response);

					MonteCarloRequest monte_carlo_request =
							parse_monte_carlo_request_json(request);

					function<MonteCarloResult(MonteCarloRequest)> cpu_omp_func = std::bind(cpu_omp_run_monte_carlo_simulation, monte_carlo_request);
					Status status = call_cpu(cpu_omp_func, "CPU (OpenMP)", response);

					if (status == Status::NOK) {
						return Rest::Route::Result::Failure;
					}
					return Rest::Route::Result::Ok;
				});

		Rest::Routes::Post(router, CPU_TBB_URL,
				[=](const Rest::Request &request,
						Http::ResponseWriter response) {

					post_enable_cors(response);

					MonteCarloRequest monte_carlo_request =
							parse_monte_carlo_request_json(request);

					function<MonteCarloResult(MonteCarloRequest)> cpu_tbb_func = std::bind(cpu_tbb_run_monte_carlo_simulation, monte_carlo_request);
					Status status = call_cpu(cpu_tbb_func, "CPU (TBB)", response);

					if (status == Status::NOK) {
						return Rest::Route::Result::Failure;
					}
					return Rest::Route::Result::Ok;
				});

		Rest::Routes::Post(router, GPU_URL,
				[=](const Rest::Request &request,
						Http::ResponseWriter response) {

					post_enable_cors(response);

					MonteCarloRequest monte_carlo_request =
							parse_monte_carlo_request_json(request);

					function<MonteCarloResult(MonteCarloRequest)> gpu_func = std::bind(gpu_run_monte_carlo_simulation, monte_carlo_request);
					Status status = call_gpu(gpu_func, "GPU", response);

					if (status == Status::NOK) {
						return Rest::Route::Result::Failure;
					}
					return Rest::Route::Result::Ok;
				});

		options_enable_cors(router, CPU_CPP_URL);
		options_enable_cors(router, CPU_OMP_URL);
		options_enable_cors(router, CPU_TBB_URL);
		options_enable_cors(router, GPU_URL);

		setHandler(router.handler());
	}
};

int main() {
	App app( { Ipv4::any(), 9080 });
	app.serve();
}
;
