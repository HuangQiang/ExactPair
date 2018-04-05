#ifndef __TRUTH_H
#define __TRUTH_H

struct Pair;

// -----------------------------------------------------------------------------
void ground_truth(					// ground truth of closest/furthest pairs
	int    n,							// cardinality
	int    d,							// dimensionality
	int    m,							// number of division
	string data_set,					// file name of data set
	string closest_set,					// file name of closest set
	string furthest_set);				// file name of furthest set

// -----------------------------------------------------------------------------
void partial_search(				// partial search		
	int n,								// cardinality	
	int d,								// dimensionaity
	int k,								// top-k value
	int m, 								// number of division
	int id1,							// data slice id 1
	int id2,							// data slide id 2
	const vector<vector<float> > &data,	// whole data
	Pair *closest_pair,					// closest pairs (return)
	Pair *furthest_pair);				// furthest pairs (return)

// -----------------------------------------------------------------------------
__global__ void euclidean(			// compute Euclidean distance
	int   n,							// cardinality of data
	int   qn,							// cardinality of query
	int   d,							// dimensionality
	int   k,							// top-k value
	int   base1,						// start position of data 
	int   base2,						// start position of query
	float *data,						// data objects
	float *query,						// query objects
	float *cp_dist, 					// closest distance (return)
	int   *cp_index,					// closest index (return)
	float *fp_dist,						// furthest distance (return)
	int   *fp_index);					// furthest index (return)
	
#endif // __TRUTH_H