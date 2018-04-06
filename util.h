#ifndef __UTIL_H
#define __UTIL_H

#include <vector>
using namespace std;

// -----------------------------------------------------------------------------
struct Pair {						// data structure for closest/furthest pair
	float key;							// distance for the pair
	int id1;							// id of point 1
	int id2;							// id of point 2

	void setto(Pair *p) {
		key = p->key;
		id1 = p->id1;
		id2 = p->id2;
	}
};

// -----------------------------------------------------------------------------
int read_set(						// read high-dimensional data from disk
	int n,								// cardinality
	int d,								// dimensionality
	string fname,						// file name of data set
	vector<vector<float> > &data);		// high-dimensional data (return)

// -----------------------------------------------------------------------------
float update_closest_pair(			// update closest pairs results
	int  k,								// number of pairs
	Pair *p,							// new pair
	Pair *pair);						// top-k pairs

// ---------------------------------------------------------------------------------------
float update_furthest_pair(			// update furthest pairs results
	int  k,								// number of pairs
	Pair *p,							// new pair
	Pair *pair);						// top-k pairs


#endif // __UTIL_H