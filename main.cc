#include "headers.h"


// -----------------------------------------------------------------------------
int main(int nargs, char **args)
{
	srand((unsigned)time(NULL));

	int n = atoi(args[1]);
	int d = atoi(args[2]);
	int m = (int) ceil((float) n / sqrt(1024.0f * 65535.0f));

	string data_set     = args[3];
	string closest_set  = args[4];
	string furthest_set = args[5];

	printf("n    = %d\n", n);
	printf("d    = %d\n", d);
	printf("m    = %d\n", m);
	printf("data = %s\n", data_set.c_str());
	
	ground_truth(n, d, m, data_set, closest_set, furthest_set);

	return 0;
}
