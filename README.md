# ExactPair：GPU-based Exact Closest/Furthest Pair Search

## Introduction

As is known, the computational cost to find the exact closest/furthest pair given a dataset of *n* data points in a *d*-dimensional space is very expensive, which is *O(n<sup>2</sup>d)*. This toolbox provides a GPU-based parallel program for the exact closest/furthest pair search. Users can quickly get the ground truth of the closest pair or furthest pair with this toolbox. 

In the remaining part, we will illustrate how to use this toolbox.

## Compilation

The toolbox requires at least **cuda 8.0** and  **g++ with c++11** support. We have enclosed a makefile. Users can download and compile the code with the following commands in bash:

```bash
git clone https://github.com/HuangQiang/ExactPair.git
cd ExactPair
make -j
```

## Running Example

We provide a script ```run.sh``` to find the closest/furthest pair of Mnist. A quick example is shown as follows:

```bash
./truth 60000 50 100 data/Mnist results/Mnist
```

where `60,000` and `50` are the `cardinality` and `dimensionality` of Mnist.bin, respectively; `100` is the top-k value for the exact pair search.
