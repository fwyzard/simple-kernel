.PHONY: all clean

CUDA_BASE := /usr/local/cuda
ALPAKA_BASE := /usr/local/alpaka/alpaka

all: test

clean:
	rm -f test *.o

test.o: test.cc
	$(CUDA_BASE)/bin/nvcc -std=c++14 -O2 -g -w -I$(ALPAKA_BASE)/include -DALPAKA_ACC_GPU_CUDA_ENABLED -x cu -dc test.cc -o test.o

kernel.o: kernel.cc
	$(CUDA_BASE)/bin/nvcc -std=c++14 -O2 -g -w -I$(ALPAKA_BASE)/include -DALPAKA_ACC_GPU_CUDA_ENABLED -x cu -dc kernel.cc -o kernel.o

dlink.o: test.o kernel.o
	$(CUDA_BASE)/bin/nvcc -std=c++14 -O2 -g -w -dlink test.o kernel.o -o dlink.o

test: test.o kernel.o dlink.o
	g++ -std=c++14 -O2 -g test.o kernel.o dlink.o -L$(CUDA_BASE)/lib64 -lcudart -lcuda -o test
