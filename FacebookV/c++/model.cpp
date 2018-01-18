#include <iostream>
#include <fstream>
#include <vector>

#include "model.h"

using namespace std;

Hist::Hist(std::string path_data, std::string path_meta)
{
	ifstream meta(path_meta);
	meta >> m_lo;
	meta >> m_hi;
	meta >> m_num;
	meta.close();

	vector<bool> mask({ false });
	for (int i = 0; i < m_num; i++) mask.push_back(true);

	m_data.reset(new Table<double>(
		path_data,
		mask,
		[](const std::string& x) { return stod(x); },
		true));
}

double Hist::Prob(size_t idx, double value)
{
	size_t pos = (size_t)((value - m_lo) / (m_hi - m_lo) * ((double)m_num - 1.0));
	return m_data->at(idx, pos);
}

double Hist::Prob(size_t idx, int value)
{
	return m_data->at(idx, value);
}

Model::Model(const std::string& dir)
{
	m_gauss_location.reset(new Table<double>(
		dir + "gauss_location.csv",
		{ false, false, true, true, true, true, true },
		[](const std::string& x) { return stod(x); },
		true));

	m_prior.reset(new Table<double>(
		dir + "prior_probabilities.csv",
		{ false, false, true },
		[](const std::string& x) { return stod(x); },
		true));

	m_labels.reset(new Table<long long>(
		dir + "prior_probabilities.csv",
		{ false, true, false },
		[](const std::string& x) { return stoll(x); },
		true));

	m_hist_day.reset(new Hist(dir + "hist_day_of_week.csv", dir + "hist_day_of_week.meta"));
	m_hist_time.reset(new Hist(dir + "hist_time_of_day.csv", dir + "hist_time_of_day.meta"));
	m_hist_accuracy.reset(new Hist(dir + "hist_accuracy.csv", dir + "hist_accuracy.meta"));
}

size_t Model::IdsCount() const
{
	return m_gauss_location->RowsCount();
}