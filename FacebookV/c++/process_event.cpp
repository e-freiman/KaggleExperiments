#define _USE_MATH_DEFINES
#include <math.h>
#include <memory>
#include <vector>
#include <algorithm>
#include <sstream>
#include <omp.h>

#include "model.h"
#include "data.h"

using namespace std;

struct ThreadLocalStorage
{
	void InitOnce(size_t size)
	{
		if (!was_inited)
		{
			probs.resize(size);
			idx.resize(size);
			was_inited = true;
		}
	}

	bool was_inited = false;

	std::vector<double> probs;
	std::vector<size_t> idx;
	std::stringstream ss;
};

const static int MAX_THREAD = 64;
ThreadLocalStorage tls[MAX_THREAD];

double gauss(double mu1, double mu2, double s1, double s12, double s2, double x1, double x2)
{
	double detS = s1*s2 - s12*s12;
	if (detS > 0) {
		return exp(-0.5*(mu2*mu2*s1-2.0*mu1*mu2*s12+mu1*mu1*s2+2.0*mu2*s12*x1-2.0*mu1*s2*x1+s2*x1*x1-2.0*mu2*s1*x2+2.0*mu1*s12*x2-2.0*s12*x1*x2+s1*x2*x2)/detS)/(2.0*M_PI*sqrt(detS));
	}
	else {
		return 0;
	}
};


std::string process_event(const Model& model, const Data& data, size_t idx)
{
	double x = data.m_coordinates->at(idx, Data::X);
	double y = data.m_coordinates->at(idx, Data::Y);
	int accuracy = data.m_accuracy->at(idx, Data::ACCURACY);
	int time = data.m_time->at(idx, Data::TIME);
		
	string result = compute_ids(model, x, y, accuracy, time);
	return result;
}


std::string compute_ids(const Model& model, double x, double y, int accuracy, int time)
{
	ThreadLocalStorage& tls_item = tls[omp_get_thread_num()];
	tls_item.InitOnce(model.IdsCount());

	const int MINUTES_IN_DAY = 60 * 24;
	
	int day_of_week = ((int)(time / MINUTES_IN_DAY)) % 7;
	int time_of_day = (time) % MINUTES_IN_DAY;

	for (size_t idx = 0; idx < model.IdsCount(); idx++)
	{
		double prior = model.m_prior->at(idx, Model::PRIOR_PROB);
		
		double mu1 = model.m_gauss_location->at(idx, Model::MU1);
		double mu2 = model.m_gauss_location->at(idx, Model::MU2);
		double s1 = model.m_gauss_location->at(idx, Model::S1);
		double s12 = model.m_gauss_location->at(idx, Model::S12);
		double s2 = model.m_gauss_location->at(idx, Model::S2);
		double prob_location = gauss(mu1, mu2, s1, s12, s2, x, y);

		double prob_day = model.m_hist_day->Prob(idx, (int)day_of_week);
		double prob_time = model.m_hist_time->Prob(idx, (double)time_of_day);
		double prob_accuracy = model.m_hist_accuracy->Prob(idx, (double)accuracy);
						
		tls_item.probs[idx] = prior * prob_location * prob_day * prob_time * prob_accuracy;
		tls_item.idx[idx] = idx;
	}

	const int K = 3;

	partial_sort(tls_item.idx.begin(), tls_item.idx.begin() + K, tls_item.idx.end(),
		[&](size_t i1, size_t i2) {return tls_item.probs[i1] > tls_item.probs[i2]; });

	
	tls_item.ss.str("");
	for (int i = 0; i < K; i++)
	{
		if (i > 0) tls_item.ss << " ";
		tls_item.ss << model.m_labels->at(tls_item.idx[i], Model::LABEL);
	}

	return tls_item.ss.str();
}