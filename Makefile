CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall

mnist: src/main.cpp src/data.cpp src/net.cpp
	$(CXX) $(CXXFLAGS) src/main.cpp src/data.cpp src/net.cpp -o mnist
