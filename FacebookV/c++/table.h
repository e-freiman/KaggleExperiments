#pragma once

#include <fstream>
#include <vector>
#include <algorithm>
#include <sstream>
#include <string>

template<typename TYPE> class Table
{
public:
	Table(const std::string& path, 
		const std::vector<bool>& columns_mask, 
		TYPE(*conversion)(const std::string&), 
		bool has_header)
	{
		std::string line;
		std::ifstream csv_file(path);
		size_t i = 0;

		if (csv_file.is_open())
		{
			if (has_header)
			{
				getline(csv_file, line);
			}

			while (getline(csv_file, line))
			{
				std::stringstream lineStream(line);
				std::string cell;

				while (getline(lineStream, cell, ','))
				{
					if (columns_mask[i++ % columns_mask.size()])
					{
						m_data.push_back(conversion(cell));
					}
				}
			}			
		}
		else
		{
			throw std::runtime_error("Problem with opening model's file");
		}

		m_row_length = 0;
		std::for_each(columns_mask.begin(), columns_mask.end(), [&](bool x) { if (x) m_row_length++; });
	}

	TYPE at(size_t row, size_t col)
	{
		return m_data[m_row_length * row + col];
	}

	size_t RowsCount() const
	{
		return m_data.size() / m_row_length;
	}

	size_t ColsCount() const
	{
		return m_row_length;
	}

private:
	std::vector<TYPE> m_data;
	size_t m_row_length;
};
