/*
 * util.h
 *
 *  Created on: Aug 30, 2019
 *      Author: andrej
 */

#ifndef UTIL_H_
#define UTIL_H_

#include "monte_carlo.cuh"

#include <iostream>
#include <string>
#include <json/json.h>
#include <json/writer.h>

#include <pistache/http.h>
#include <pistache/router.h>
#include <pistache/endpoint.h>

using namespace Pistache;
using namespace std;

inline string dump_monte_carlo_result(MonteCarloResult monte_carlo_result) {

	Json::Value data;
	Json::StreamWriterBuilder json_writer_builder;

//  HISTOGRAM IS VERY SLOW TO RENDER SINCE IT CAN GET QUITE HUGE!
//	for(unsigned int i = 0; i < monte_carlo_result.histogram_values.size(); i++) {
//		data["histogram"][i]["name"] = monte_carlo_result.histogram_values[i];
//		data["histogram"][i]["value"] = monte_carlo_result.histogram_counts[i];
//	}

	data["mean"] = monte_carlo_result.mean;
	data["max"] = monte_carlo_result.max;
	data["min"] = monte_carlo_result.min;
	data["standardDeviation"] = monte_carlo_result.standard_deviation;
	data["simulationTime"] = monte_carlo_result.simulation_time;
	data["numIterations"] = static_cast<Json::Value::UInt64>(monte_carlo_result.num_iterations);
	data["timeHorizon"] = monte_carlo_result.time_horizon;
	data["ranOn"] = monte_carlo_result.ran_on;

	return Json::writeString(json_writer_builder, data);
};

inline MonteCarloRequest parse_monte_carlo_request_json(const Rest::Request& request) {

	string monte_carlo_request_json = request.body();
	Json::Value root;
	Json::CharReaderBuilder builder;
	Json::CharReader *reader = builder.newCharReader();
	string errors;

	bool parsingSuccessful = reader->parse(monte_carlo_request_json.c_str(), monte_carlo_request_json.c_str() + monte_carlo_request_json.size(), &root, &errors);
	delete reader;

	if (!parsingSuccessful) {
		cout << "Error while parsing monte carlo request JSON." << endl;
		cout << errors << endl;
	}

	MonteCarloRequest monte_carlo_request = MonteCarloRequest();
	monte_carlo_request.num_iterations = root["numIterations"].asLargestUInt();
	monte_carlo_request.expected_return = root["expectedReturn"].asFloat();
	monte_carlo_request.volatility = root["volatility"].asFloat();
	monte_carlo_request.time_horizon = root["timeHorizon"].asUInt();
	monte_carlo_request.starting_value = root["startingValue"].asFloat();
	monte_carlo_request.annual_investment = root["annualInvestment"].asFloat();

	return monte_carlo_request;
};

inline void options_enable_cors(Rest::Router &router, string url) {

	Rest::Routes::Options(
			router,
			url,
			[=](const Rest::Request & request, Http::ResponseWriter response) {

		response.headers().add<Http::Header::AccessControlAllowOrigin>("*");
	    response.headers().add<Http::Header::AccessControlAllowHeaders>("*");
	    response.headers().add<Http::Header::AccessControlAllowMethods>("POST, GET, OPTIONS");

	    response.send(Http::Code::Ok, "CORS OK");

	    return Rest::Route::Result::Ok;
	});

}

inline void post_enable_cors(Http::ResponseWriter &response) {

	response.headers().add<Http::Header::AccessControlAllowOrigin>("*");
	response.headers().add<Http::Header::AccessControlAllowHeaders>("*");
	response.headers().add<Http::Header::AccessControlAllowMethods>("POST, GET, OPTIONS");
}

#endif /* UTIL_H_ */
