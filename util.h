#ifndef __UTIL_H
#define __UTIL_H

// -----------------------------------------------------------------------------
struct Pair {						// data structure for closest/furthest pair
	float key;							// distance for the pair
	int   id1;							// id of point 1
	int   id2;							// id of point 2

	void setto(Pair *p) {
		key = p->key;
		id1 = p->id1;
		id2 = p->id2;
	}
};

// -----------------------------------------------------------------------------
void create_dir(					// create dir if the path does not exist
	char *path);						// input path

// -----------------------------------------------------------------------------
void read_bin_data(					// read data_set from binary data
	int n,								// cardinality
	int d,								// dimensionality
	const char *fname,					// file name of data set
	float *data); 						// data (return)

// -----------------------------------------------------------------------------
float update_closest_pair(			// update closest pairs results
	int  k,								// number of pairs
	Pair *p,							// new pair
	Pair *pair);						// top-k pairs

// -----------------------------------------------------------------------------
float update_furthest_pair(			// update furthest pairs results
	int  k,								// number of pairs
	Pair *p,							// new pair
	Pair *pair);						// top-k pairs

// -----------------------------------------------------------------------------
void write_results(					// write results to disk
	int   k,							// top-k value
	const Pair *pairs,					// closest/furthest pairs
	const char *result_set);			// result set

#endif // __UTIL_H