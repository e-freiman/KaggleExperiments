#pragma once

#include <memory>
#include <string>
#include <vector>

#include "table.h"

class Hist
{
public:
	Hist(std::string path_data, std::string path_meta);
	double Prob(size_t idx, double value);
	double Prob(size_t idx, int value);
private:
	double m_lo, m_hi;
	int m_num;
	std::unique_ptr<Table<double>> m_data;
};

class Model
{
public:
	Model(const std::string& dir);
	size_t IdsCount() const;

private:
	enum GaussFields
	{
		MU1 = 0,
		MU2 = 1, 
		S1  = 2,
		S12 = 3,
		S2  = 4
	};

	enum PriorFields
	{
		PRIOR_PROB = 0
	};

	enum LabelFields
	{
		LABEL = 0
	};

	std::unique_ptr<Table<double>> m_gauss_location;
	std::unique_ptr<Table<double>> m_prior;
	std::unique_ptr<Table<long long>> m_labels;

	std::unique_ptr<Hist> m_hist_day;
	std::unique_ptr<Hist> m_hist_time;
	std::unique_ptr<Hist> m_hist_accuracy;

	friend std::string compute_ids(const Model& model, double x, double y, int accuracy, int time);
};