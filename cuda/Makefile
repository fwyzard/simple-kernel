.PHONY: all clean

CUDA_BASE := /usr/local/cuda

all: test

clean:
	rm -f test *.o

test.o: test.cc kernel.h
	g++ -std=c++14 -O2 -g -I$(CUDA_BASE)/include -c test.cc -o test.o

kernel.o: kernel.cu kernel.h
	$(CUDA_BASE)/bin/nvcc -std=c++14 -O2 -g -c kernel.cu -o kernel.o

test: test.o kernel.o
	g++ -std=c++14 -O2 -g test.o kernel.o -L$(CUDA_BASE)/lib64 -lcudart -lcuda -o test
