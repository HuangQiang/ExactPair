#include "headers.h"

// -----------------------------------------------------------------------------
int main(int nargs, char **args)
{
	srand((unsigned)time(NULL));

	int  n = atoi(args[1]);
	int  d = atoi(args[2]);
	int  k = atoi(args[3]);
	char data_set[200]; strcpy(data_set, args[4]);
	char out_path[200]; strcpy(out_path, args[5]);

	create_dir(out_path);
	char closest_set[200];  sprintf(closest_set,  "%s.cp", out_path);
	char furthest_set[200]; sprintf(furthest_set, "%s.fp", out_path);

	printf("n_pts    = %d\n", n);
	printf("dim      = %d\n", d);
	printf("k      = %d\n", k);
	printf("data_set = %s\n", data_set);
	printf("out_path = %s\n", data_set);

	// -------------------------------------------------------------------------
	//  find the closest/furthest pairs ground truth results
	// -------------------------------------------------------------------------
	float *data = new float[n * d]; 
	Pair  *closest_pair  = new Pair[k];
	Pair  *furthest_pair = new Pair[k];

	read_bin_data(n, d, data_set, data);
	ground_truth(n, d, k, data, closest_pair, furthest_pair);

	// -------------------------------------------------------------------------
	//  write the closest/furthest pairs ground truth results to disk
	// -------------------------------------------------------------------------
	write_results(k, closest_pair,  closest_set);
	write_results(k, furthest_pair, furthest_set);

	// -------------------------------------------------------------------------
	//  release space
	// -------------------------------------------------------------------------
	delete[] data;          data          = NULL;
	delete[] closest_pair;  closest_pair  = NULL;
	delete[] furthest_pair; furthest_pair = NULL;

	return 0;
}
