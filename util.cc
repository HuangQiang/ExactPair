#include "headers.h"

// -----------------------------------------------------------------------------
void create_dir(					// create dir if the path does not exist
	char *path)							// input path
{
	int len = (int) strlen(path);
	for (int i = 0; i < len; ++i) {
		if (path[i] == '/') {
			char ch = path[i + 1];
			path[i + 1] = '\0';
									// check whether the directory exists
			int ret = access(path, F_OK);
			if (ret != 0) {			// create the directory
				ret = mkdir(path, 0755);
				if (ret != 0) {
					printf("Could not create directory %s\n", path);
				}
			}
			path[i + 1] = ch;
		}
	}
}

// -----------------------------------------------------------------------------
void read_bin_data(					// read data_set from binary data
	int n,								// cardinality
	int d,								// dimensionality
	const char *fname,					// file name of data set
	float *data) 						// data (return)
{
	timeval start_time, end_time;

	gettimeofday(&start_time, NULL);
	FILE *fin = fopen(fname, "rb");
	if (!fin) { printf("cannot read dataset from %s\n", fname); return; }

	for (int i = 0; i < n; ++i) {
		fread(&data[i * d], sizeof(float), d, fin);
		if ((i+1) % 100000 == 0) printf("i = %d, n = %d\n", i+1, n);
	}
	fclose(fin);

	gettimeofday(&end_time, NULL);
	float read_file_time = end_time.tv_sec - start_time.tv_sec + 
		(end_time.tv_usec - start_time.tv_usec) / 1000000.0f;
	printf("Read Dataset: %f Seconds\n\n", read_file_time);
}

// -----------------------------------------------------------------------------
float update_closest_pair(			// update closest pairs results
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
		else if (p->key < pair[i].key) break;
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
		else if (p->key > pair[i].key) break;
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

// -----------------------------------------------------------------------------
void write_results(					// write results to disk
	int   k,							// top-k value
	const Pair *pairs,					// closest/furthest pairs
	const char *result_set)				// result set
{
	FILE *fp = fopen(result_set, "w");
	fprintf(fp, "%d\n", k);
	for (int i = 0; i < k; ++i) {
		fprintf(fp, "%d %d %f\n", pairs[i].id1, pairs[i].id2, pairs[i].key);
	}
	fclose(fp);
}