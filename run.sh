#!/bin/bash
make clean
make

# ------------------------------------------------------------------------------
#  Mnist
# ------------------------------------------------------------------------------
n=60000
d=50
k=100
dname=Mnist
dPath=data/${dname}
rPath=results/${dname}

./truth ${n} ${d} ${k} ${dPath}.bin ${rPath}

