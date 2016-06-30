PROJECT := biglearn

NVCC := nvcc
CXX := g++

ARCH := $(shell getconf LONG_BIT)

CUDA_PATH ?= /usr/local/cuda
LIB_FLAGS_32 := -L$(CUDA_PATH)/lib
LIB_FLAGS_64 := -L$(CUDA_PATH)/lib64
LIB_FLAGS := $(LIB_FLAGS_64)

CUDA_DIR := /usr/local/cuda

# Flags passed to the C++ compiler.
CXXFLAGS += -g -Wextra -pthread --std=c++11

# Flags passed to nvcc compiler.
NVCCFLAGS += -g -arch=sm_30 --std=c++11 --expt-extended-lambda


all: example clean
	
#-L/usr/local/cuda/lib64 -lcudart 
#-I/usr/local/cuda/include 
example: example.o vocab.o cudaPiece.o sparseMatrixData.o
	$(NVCC) -I/usr/local/cuda/include  -L/usr/local/cuda/lib64 -lcudart  -o example example.o vocab.o cudaPiece.o sparseMatrixData.o

example.o: example.cpp vocab.h cudaPiece.h
	$(CXX) -c $(CXXFLAGS) -I/usr/local/cuda/include  -L/usr/local/cuda/lib64 -lcudart  example.cpp

vocab.o: vocab.cpp vocab.h
	$(CXX) -c $(CXXFLAGS) -I/usr/local/cuda/include  -L/usr/local/cuda/lib64 -lcudart  vocab.cpp vocab.h

cudaPiece.o: cudaPiece.cpp cudaPiece.h
	$(CXX) -c $(CXXFLAGS) -I/usr/local/cuda/include  -L/usr/local/cuda/lib64 -lcudart  cudaPiece.cpp cudaPiece.h

sparseMatrixData.o : sparseMatrixData.cpp sparseMatrixData.h
	$(CXX) -c $(CXXFLAGS) -I/usr/local/cuda/include  -L/usr/local/cuda/lib64 -lcudart  sparseMatrixData.cpp sparseMatrixData.h

util.o : util.cpp
	$(CXX) -c $(CXXFLAGS) -I/usr/local/cuda/include  -L/usr/local/cuda/lib64 -lcudart  util.cpp


.PHONY: clean
clean:
	rm -rf *.o



#-o example

#example.o: example.cpp
#		$(CXX) -c $(CXXFLAGS) -o example
