SRCS=util.cc truth.cu main.cc
OBJS=$(SRCS:.cc=.o)

CXX=nvcc -std=c++11
CXXFLAGS=-g -Xcompiler -fopenmp -w -O3
CXXFLAGS+=$(FLAGS) -Wno-deprecated-gpu-targets

.PHONY: clean

all: ${OBJS}
	${CXX} ${CXXFLAGS} -o truth ${OBJS}

util.o: util.h 

truth.o: truth.h

main.o:

clean:
	-rm ${OBJS}
