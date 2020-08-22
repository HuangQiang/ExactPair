# ------------------------------------------------------------------------------
#  makefile with cuda
# ------------------------------------------------------------------------------
ALLOBJFILES = util.o truth.o main.o

all:${ALLOBJFILES}
	nvcc -std=c++11 -w -O3 -o truth -lm -lcudart -lcublas ${ALLOBJFILES}

%.o: %.cu 
	nvcc -std=c++11 -c -w -O3 -o $@ $<

%.o: %.cc 
	nvcc -std=c++11 -c -w -O3 -o $@ $<

clean:
	-rm *.o truth
