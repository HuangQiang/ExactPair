# Pairs_Truth：Ground Truth for Closest/Furthest Pairs

## Introduction

This package provides a parallel program to find the ground truth of closest/furthest pairs search using GPU. 

## Compilation

The package requires at least ```cuda 8.0``` and  ```g++ with c++11``` support. To download and compile the code, type:

```bash
$ git clone https://github.com/HuangQiang/Pairs_Truth.git
$ cd Pairs_Truth
$ make
```

## Run Example

We provide a script ```run.sh``` to find the closest/furthest pairs of datasets. A quick example is shown as follows (find closest and furthest pairs on ```OptDigits.data```):

```bash
./truth 3823 64 OptDigits.data OptDigits.cp2.0 OptDigits.fp2.0
```

where `3823` is the `cardinality` of OptDigits.data and ``64`` is ``dimensionality`` of OptDigits.data.
