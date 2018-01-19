#include <string>
#include <fstream>
#include <iostream>
#include <chrono>
#include <omp.h>
#include <thread>

#include "model.h"
#include "data.h"

using namespace std;

// C:/Repository/KaggleFacebookV/generated_model100/ C:/Repository/KaggleFacebookV/input/valid09_100.csv 1
// C:/Repository/KaggleFacebookV/generated_model/ C:/Repository/KaggleFacebookV/input/test.csv 0

int main(int argc, char** argv)
{
	if (argc != 4)
	{
		cout << "Wrong number of parameters provided."  << endl
			 << "It should be \"FacebookV.exe <path to the model directory> <path to the data> <0/1>\"" << endl
			 << "The last parameter indicates if test data has true labels. The path to model dir should conrain \"/\" at the end" << endl;
		
		return 1;
	}

	std::string model_dir(argv[1]);
	std::string data_path(argv[2]);
	bool has_true_id(stoi(argv[3]));

	cout << "Loading data ..." << endl;

	std::unique_ptr<Model> model(new Model(model_dir));
	std::unique_ptr<Data> data(new Data(data_path, has_true_id));

	std::vector<std::string> result;
	result.resize(data->EventsCount());

	cout << "Computing ..." << endl;

	#pragma omp parallel for
	for (int idx = 0; idx < (int)data->EventsCount(); idx++)
	{					
		if (idx == 100)
		{
			//this brunch is only for time measurement
			auto start = std::chrono::high_resolution_clock::now();			
			result[idx] = process_event(*model.get(), *data.get(), (size_t)idx);
			auto elapsed = std::chrono::high_resolution_clock::now() - start;

			long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();			
			double speed = (1000000.0 * 60.0 * 60.0) / (double)(microseconds);
			int n_per_core = data->EventsCount() / omp_get_num_threads();
			double hours = (double)(n_per_core) / speed;
			cout << "Number of threads: " << omp_get_num_threads() << endl;
			cout << "Elapsed time " << hours << " hours" << endl;
		}
		else
		{
			result[idx] = process_event(*model.get(), *data.get(), (size_t)idx);			
		}
	}

	cout << "Saving results to \"result.csv\" ..." << endl;

	ofstream out("result.csv");
	out << "row_id,place_id" << endl;
	for (size_t i = 0; i < result.size(); i++)
	{
		out << i << "," << result.at(i) << endl;
	}
	out.close();

	cout << "Finish !!!" << endl;

	return 0;
}