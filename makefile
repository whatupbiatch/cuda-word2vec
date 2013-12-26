#simple makefile, make sure nvcc is available
CC=nvcc
CFLAGS= -arch=compute_30 -code=sm_30
SOURCE= src/language_network_kernel.cu src/language_network_main.cu src/language_network_utility.cu src/includes/optParser/getopt_pp.cpp
TESTSOURCE = src/language_network_kernel.cu src/language_network_test.cu src/language_network_utility.cu src/includes/optParser/getopt_pp.cpp
BIN= CBOWMODEL
TESTBIN = CBOWMODELTEST

all: 
	$(CC) -o $(BIN) $(CFLAGS) $(SOURCE)

test:
	$(CC) -o $(TESTBIN) $(CFLAGS) $(TESTSOURCE)
