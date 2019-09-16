#include "headers.h"

// -----------------------------------------------------------------------------
int read_set(						// read high-dimensional data from disk
	int n,								// cardinality
	int d,								// dimensionality
	string fname,						// file name of data set
	vector<vector<float> > &data)		// high-dimensional data (return)
{
	FILE *fp = fopen(fname.c_str(), "r");
	if (!fp) {
		printf("Could not open %s.\n", fname.c_str());
		return 1;
	}

	int i = 0;
	int j = 0;
	while (!feof(fp) && i < n) {
		fscanf(fp, "%d", &j);
		for (j = 0; j < d; ++j) {
			fscanf(fp, " %f", &data[i][j]);
		}
		fscanf(fp, "\n");

		++i;
	}
	fclose(fp);

	return 0;
}

// -----------------------------------------------------------------------------
float update_closest_pair(			// update closest pairs results
	int k,								// number of pair
	Pair *p,							// new pair
	Pair *pair)							// top-k pairs
{
	int i = -1;
	int pos = -1;

	for (i = 0; i < k; ++i) {
		if (fabs(p->key - pair[i].key) < 1e-6) {
			if ((p->id1 == pair[i].id1 && p->id2 == pair[i].id2) ||
				(p->id1 == pair[i].id2 && p->id2 == pair[i].id1))
				return pair[k - 1].key;
		}
		else if (p->key < pair[i].key) {
			break;
		}

	}
	pos = i;
	if (pos < k) {
		for (i = k - 1; i > pos; --i) {
			pair[i].setto(&pair[i - 1]);
		}
		pair[pos].setto(p);
	}
	return pair[k - 1].key;
}

// ---------------------------------------------------------------------------------------
float update_furthest_pair(			// update furthest pairs results
	int  k,								// number of pair
	Pair *p,							// new pair
	Pair *pair)							// top-k pairs
{
	int i = -1;
	int pos = -1;

	for (i = 0; i < k; ++i) {
		if (fabs(p->key - pair[i].key) < 1e-6) {
			if ((p->id1 == pair[i].id1 && p->id2 == pair[i].id2) ||
				(p->id1 == pair[i].id2 && p->id2 == pair[i].id1))
				return pair[k - 1].key;
		}
		else if (p->key > pair[i].key) {
			break;
		}

	}
	pos = i;
	if (pos < k) {
		for (i = k - 1; i > pos; --i) {
			pair[i].setto(&pair[i - 1]);
		}
		pair[pos].setto(p);
	}
	return pair[k - 1].key;
}
