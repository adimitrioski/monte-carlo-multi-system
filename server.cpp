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
	string URL = "/api/monte_carlo/";

public:
	App(Address addr) :
			Http::Endpoint(addr) {

		auto opts = Http::Endpoint::options().flags(Tcp::Options::ReuseAddr);

		init(opts);

		Rest::Routes::Post(router, URL,
				[=](const Rest::Request &request,
						Http::ResponseWriter response) {

					post_enable_cors(response);

					Optional<string> runOnOptional = request.query().get("runOn");

					if (runOnOptional.isEmpty()) {
						response.send(Http::Code::Bad_Request, "Missing 'runOn' request parameter.");
						return Rest::Route::Result::Failure;
					}

					MonteCarloRequest monte_carlo_request = parse_monte_carlo_request_json(request);

					string run_on = runOnOptional.get();
					RunOn run_on_enum = resolveRunOn(run_on);
					Status status;

					switch(run_on_enum) {

					    case RunOn::cpu: {
					    	function<MonteCarloResult(MonteCarloRequest)> cpu_cpp_func = std::bind(cpu_cpp_run_monte_carlo_simulation, monte_carlo_request);
					    	status = call_cpu(cpu_cpp_func, "CPU", response);
					    	break;
					    }
					    case RunOn::cpu_omp: {
					    	function<MonteCarloResult(MonteCarloRequest)> cpu_omp_func = std::bind(cpu_omp_run_monte_carlo_simulation, monte_carlo_request);
					    	status = call_cpu(cpu_omp_func, "CPU (OpenMP)", response);
					    	break;
					    }
					    case RunOn::cpu_tbb: {
					    	function<MonteCarloResult(MonteCarloRequest)> cpu_tbb_func = std::bind(cpu_tbb_run_monte_carlo_simulation, monte_carlo_request);
					    	status = call_cpu(cpu_tbb_func, "CPU (TBB)", response);
					    	break;
					    }
					    case RunOn::gpu: {
					    	function<MonteCarloResult(MonteCarloRequest)> gpu_func = std::bind(gpu_run_monte_carlo_simulation, monte_carlo_request);
					    	status = call_gpu(gpu_func, "GPU", response);
					    	break;
					    }
					    default: {
					    	response.send(Http::Code::Bad_Request, "Invalid 'runOn' request parameter.");
					    	return Rest::Route::Result::Failure;
					    }
					}

					if (status == Status::NOK) {
						return Rest::Route::Result::Failure;
					}
					return Rest::Route::Result::Ok;
				});

		options_enable_cors(router, URL);

		setHandler(router.handler());
	}
};

int main() {
	App app( { Ipv4::any(), 9080 });
	app.serve();
}
