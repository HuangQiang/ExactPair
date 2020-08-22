#ifndef __TRUTH_H
#define __TRUTH_H

// -----------------------------------------------------------------------------
void ground_truth(					// ground truth of closest/furthest pairs
	int   n,							// cardinality
	int   d,							// dimensionality
	int   k,							// top_k value
	const float *data,					// input data
	Pair  *closest_pair,				// closest  pairs (return)
	Pair  *furthest_pair);				// furthest pairs (return)

// -----------------------------------------------------------------------------
void partial_search(				// partial search
	int   n,							// cardinality	
	int   d,							// dimensionaity
	int   k,							// top-k value
	int   m,							// number of division
	int   id1,							// data slice id 1
	int   id2,							// data slide id 2
	const float *data,					// whole data
	Pair  *closest_pair,				// closest pairs (return)
	Pair  *furthest_pair);				// furthest pairs (return)

// -----------------------------------------------------------------------------
__global__ void euclidean(			// compute Euclidean distance
	int   n,							// cardinality of data
	int   qn,							// cardinality of query
	int   d,							// dimensionality
	float *data,						// data objects
	float *query,						// query objects
	float *results); 					// all-pair results (return)
	
#endif // __TRUTH_H