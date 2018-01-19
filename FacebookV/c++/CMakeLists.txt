cmake_minimum_required(VERSION 2.8)

set(HEADERS
	table.h
	model.h
	data.h)

set(SOURCES 
	main.cpp 
	model.cpp 
	data.cpp 
	process_event.cpp)

if (${CMAKE_CXX_COMPILER_ID} STREQUAL "MSVC")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp")
elseif (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -std=c++11 -ffloat-store")
else ()
	message(FATAL_ERROR "${CMAKE_CXX_COMPILER_ID} is not supported")
endif ()

add_executable(FacebookV ${SOURCES} ${HEADERS})
