#pragma once

#include <memory>
#include <vector>

#include "table.h"

class Model;

class Data
{
public:
	Data(std::string path, bool has_true_id);
	size_t EventsCount() const;
private:
	enum CoordinatesField
	{
		X = 0,
		Y = 1,
	};

	enum AccuracyField
	{
		ACCURACY = 0,
	};

	enum TimeField
	{
		TIME = 0,
	};

	enum TrueIdField
	{
		ID = 0
	};

	std::unique_ptr<Table<double>> m_coordinates;
	std::unique_ptr<Table<int>> m_accuracy;
	std::unique_ptr<Table<int>> m_time;
	std::unique_ptr<Table<long long>> m_true_id;

	friend std::string process_event(const Model& model, const Data& data, size_t idx);
};