# Location of the CUDA Toolkit binaries and libraries
CUDA_80_PATH   ?= /usr/local/cuda-8.0
CUDA_75_PATH   ?= /usr/local/cuda-7.5
CUDA_65_PATH   ?= /usr/local/cuda-6.5
CUDA_DRIVER_PATH ?= /usr/lib/x86_64-linux-gnu

ifeq ($(wildcard /usr/local/cuda-8.0/nvcc),)
	CUDA_PATH ?= /usr/local/cuda-8.0
else ifeq ($(wildcard /usr/local/cuda-7.5/nvcc),)
	CUDA_PATH ?= /usr/local/cuda-7.5
else ifeq ($(wildcard /usr/local/cuda-6.5/nvcc),)
	CUDA_PATH ?= /usr/local/cuda-6.5
endif

CUDA_VERSION  ?= $(shell $(CUDA_PATH)/nvcc --version  | grep "release" | sed 's/.*release \([0-9\.]*\),.*/\1/g')
	
CUDA_INC_PATH   ?= $(CUDA_PATH)/include
CUDA_BIN_PATH   ?= $(CUDA_PATH)/bin
CUDA_LIB_PATH  ?= $(CUDA_PATH)/lib64
CUDNN_PATH		?= /usr/local/cudnn-2.0

# Common binaries
NVCC            ?= $(CUDA_BIN_PATH)/nvcc
GPP             ?= g++
GCC				?= gcc

# Extra user flags
EXTRA_NVCCFLAGS ?=
EXTRA_LDFLAGS   ?=

# CUDA code generation flags
GENCODE_SM50    := -gencode arch=compute_50,code=sm_50
GENCODE_FLAGS   :=  $(GENCODE_SM50) 
CCFLAGS   += -m64 -g -std=c++11
NVCCFLAGS := -m64 -g 
#-Xcompiler 
#-fPIC
OBJD =	./object
TARGET_PATH = ./binary


# Common includes and paths for CUDA
INCLUDES      := -I$(CUDA_INC_PATH) -I. -I.. -I$(CUDA_PATH)/samples/common/inc -I./headers/cuda -I$(CUDNN_PATH)
LDFLAGS   :=  -L$(CUDNN_PATH) -lcudnn -L$(CUDA_LIB_PATH) -lcudadevrt -lcublas -L$(CUDA_DRIVER_PATH) -lcuda  -L/usr/lib64 -lstdc++ -ldl -lm -lpthread

LINK_OBJECTS = $(OBJD)/cudaCompute.o 
#$(OBJD)/activation.o $(OBJD)/bplda.o $(OBJD)/cudacal.o $(OBJD)/LBFGS.o $(OBJD)/lstm.o $(OBJD)/matrix.o $(OBJD)/optlstm.o $(OBJD)/rnn.o

shared_OBJECTS = $(OBJD)/main.o $(OBJD)/denseMatrixData.o $(OBJD)/PieceMem.o $(OBJD)/sparseMatrixData.o $(OBJD)/util.o $(OBJD)/vocab.o  $(LINK_OBJECTS)

# Target rules
all: $(TARGET_PATH)/DSSM
#libCudalib.so	

$(TARGET_PATH)/DSSM: $(shared_OBJECTS) 
	if ! (test -d $(TARGET_PATH)); then  mkdir $(TARGET_PATH); fi
	$(NVCC) -o $@ $^ $(LDFLAGS) 

#$(OBJD)/%.o: %.cpp $(OBJD)/link.o
$(OBJD)/%.o: %.cpp
	if ! (test  -d $(OBJD)); then mkdir $(OBJD); fi
	$(GCC) -o $@ $(INCLUDES) -O3 -c $<  $(CCFLAGS) 

#$(OBJD)/link.o: $(LINK_OBJECTS)
#	$(NVCC) -o $(OBJD)/link.o -dlink $(LINK_OBJECTS) $(GENCODE_SM50) $(LDFLAGS) $(NVCCFLAGS)

$(OBJD)/%.o: %.cu
	if ! (test  -d $(OBJD)); then mkdir $(OBJD); fi
	$(NVCC) -O3 -c -o $@ $< $(GENCODE_SM50)  $(NVCCFLAGS)

clean:
	test -d $(OBJD) && rm -r $(OBJD)
	test -d $(TARGET_PATH) && rm -r $(TARGET_PATH)
	#rm $(OBJD)/*.o $(TARGET_PATH)/*.so
