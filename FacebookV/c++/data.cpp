#include <string>

#include "data.h"

using namespace std;

Data::Data(std::string path, bool has_true_id)
{
	std::vector<bool> coordinates_mask({ false, true, true, false, false });
	if (has_true_id) coordinates_mask.push_back(false);
	m_coordinates.reset(new Table<double>(
		path,
		coordinates_mask,
		[](const std::string& x) { return stod(x); },
		true));

	std::vector<bool> accuracy_mask({ false, false, false, true, false });
	if (has_true_id) accuracy_mask.push_back(false);
	m_accuracy.reset(new Table<int>(
		path,
		accuracy_mask,
		[](const std::string& x) { return stoi(x); },
		true));

	std::vector<bool> time_mask({ false, false, false, false, true});
	if (has_true_id) time_mask.push_back(false);
	m_time.reset(new Table<int>(
		path,
		time_mask,
		[](const std::string& x) { return stoi(x); },
		true));
}

size_t Data::EventsCount() const
{
	return m_coordinates->RowsCount();
}