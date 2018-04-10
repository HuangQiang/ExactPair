#include "headers.h"


// -----------------------------------------------------------------------------
void ground_truth(					// ground truth of closest/furthest pairs
	int n,								// cardinality
	int d,								// dimensionality
	int m,								// number of division
	string data_set,					// file name of data set
	string closest_set,					// file name of closest set
	string furthest_set)				// file name of furthest set
{
	timeval start_time, end_time;

	// -------------------------------------------------------------------------
	//  step 1: read data set
	// -------------------------------------------------------------------------
	gettimeofday(&start_time, NULL);
	vector<vector<float> > data(n, vector<float>(d, 0.0f));
	if (read_set(n, d, data_set, data)) {
		printf("Reading Dataset Error!\n");
		exit(1);
	}

	gettimeofday(&end_time, NULL);
	float read_file_time = end_time.tv_sec - start_time.tv_sec + 
		(end_time.tv_usec - start_time.tv_usec) / 1000000.0f;
	printf("Read Dataset: %f Seconds\n\n", read_file_time);

	// -------------------------------------------------------------------------
	//  step 2: call kernel function
	// -------------------------------------------------------------------------
	gettimeofday(&start_time, NULL);
	int k = MAXK;
	float kernel_time = 0.0f;
	Pair  *closest_pair  = new Pair[k];
	Pair  *furthest_pair = new Pair[k];
	
	for (int i = 0; i < k; ++i) {
		closest_pair[i].key = MAXREAL;
		closest_pair[i].id1 = -1;
		closest_pair[i].id2 = -1;

		furthest_pair[i].key = MINREAL;
		furthest_pair[i].id1 = -1;
		furthest_pair[i].id2 = -1;
	}

	for (int i = 0; i < m; ++i) {
		for (int j = i; j < m; ++j) {
			partial_search(n, d, k, m, i, j, data, closest_pair, furthest_pair);

			gettimeofday(&end_time, NULL);
			kernel_time = end_time.tv_sec - start_time.tv_sec + 
				(end_time.tv_usec - start_time.tv_usec) / 1000000.0f;
			printf("Call Kernel Function: %f Seconds\n\n", kernel_time);
		}
	}

	// -------------------------------------------------------------------------
	//  step 3: write the closest/furthest pairs ground truth results to disk
	// -------------------------------------------------------------------------
	FILE *fp = fopen(closest_set.c_str(), "w");
	fprintf(fp, "%d\n", k);
	for (int i = 0; i < k; ++i) {
		fprintf(fp, "%d %d %f\n", closest_pair[i].id1, closest_pair[i].id2, 
			closest_pair[i].key);
	}
	fclose(fp);

	fp = fopen(furthest_set.c_str(), "w");
	fprintf(fp, "%d\n", k);
	for (int i = 0; i < k; ++i) {
		fprintf(fp, "%d %d %f\n", furthest_pair[i].id1, furthest_pair[i].id2, 
			furthest_pair[i].key);
	}
	fclose(fp);

	// -------------------------------------------------------------------------
	//  step 4: release space
	// -------------------------------------------------------------------------
	delete[] closest_pair;  closest_pair  = NULL;
	delete[] furthest_pair; furthest_pair = NULL;
}

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
	Pair *furthest_pair)				// furthest pairs (return)
{
	// -------------------------------------------------------------------------
	//  step 1: init parameters
	//  two conditions: num_threads <= 1024 & num_blocks <= 65535
	// -------------------------------------------------------------------------
	int num = (n + m - 1) / m;

	int n1 = num;
	int n2 = num;
	if (id1 == m - 1) n1 = n - num * (m - 1);
	if (id2 == m - 1) n2 = n - num * (m - 1);

	int base1 = id1 * num;
	int base2 = id2 * num;

	int result_size = n1 * n2;
	int num_threads = 1024;
	int num_blocks  = (result_size + num_threads - 1) / num_threads;

	printf("(%d, %d):\n", id1 + 1, id2 + 1);
	printf("n1          = %d\n", n1);
	printf("n2          = %d\n", n2);
	printf("result size = %d\n", result_size);
	printf("num threads = %d\n", num_threads);
	printf("num blocks  = %d\n", num_blocks);

	// -------------------------------------------------------------------------
	//  step 2: flatten and copy high-dimensional data
	// -------------------------------------------------------------------------
	vector<float> tmp_data;
	for (int j = base1; j < base1 + n1; ++j) {
		tmp_data.insert(end(tmp_data), begin(data[j]), end(data[j]));
	}

	vector<float> tmp_query;
	for (int j = base2; j < base2 + n2; ++j) {
		tmp_query.insert(end(tmp_query), begin(data[j]), end(data[j]));
	}
	
	thrust::device_vector<float> d_data(tmp_data.begin(), tmp_data.end());
	thrust::device_vector<float> d_query(tmp_query.begin(), tmp_query.end());
	thrust::device_vector<float> d_results(result_size, 0.0f);
	
	// -------------------------------------------------------------------------
	//  step 3: call kernel function
	// -------------------------------------------------------------------------
	euclidean<<<num_blocks, num_threads>>>(
		n1, n2, d, 
		thrust::raw_pointer_cast(d_data.data()),
		thrust::raw_pointer_cast(d_query.data()),		
		thrust::raw_pointer_cast(d_results.data()));

	cudaDeviceSynchronize();

	// -------------------------------------------------------------------------
	//  step 4: update closest/furthest pair results from GPU to CPU
	// -------------------------------------------------------------------------
	thrust::host_vector<float> h_results = d_results;
	thrust::host_vector<float> h_one_results(n2);
	thrust::host_vector<int>   h_one_ids(n2);

	Pair  pair;
	float min = MAXREAL;
	float max = MINREAL;

	for (int i = 0; i < n1; ++i) {
		// ---------------------------------------------------------------------
		//  step 4.1: get all-pairs results
		// ---------------------------------------------------------------------
		int base = i * n2;
		for (int j = 0; j < n2; ++j) {
			h_one_results[j] = h_results[base + j];
			h_one_ids[j] = base2 + j;

			if (base1 + i == base2 + j) h_one_results[j] = -1.0f;
		}

		// ---------------------------------------------------------------------
		//  step 4.2: sort results by dist (ascending order)
		// ---------------------------------------------------------------------
		thrust::sort_by_key(h_one_results.begin(), h_one_results.end(), 
			h_one_ids.begin());

		// ---------------------------------------------------------------------
		//  step 4.3: update closest-pair results
		// ---------------------------------------------------------------------
		int size = min(n2 - 1, k);
		for (int j = 0; j <= size; ++j) {
			float dist = h_one_results[j];

			if (dist < 0.0f) {
				continue;			// skip scanning the same id
			}
			else if (dist < min) {
				pair.key = dist;
				pair.id1 = base1 + i + 1;
				pair.id2 = h_one_ids[j] + 1;
				
				min = update_closest_pair(k, &pair, closest_pair);
			}
			else {
				break;				// no better results, stop early
			}
		}

		// ---------------------------------------------------------------------
		//  step 4.4: update furthest-pair results
		// ---------------------------------------------------------------------
		size = min(n2, k);
		for (int j = 0; j < size; ++j) {
			float dist = h_one_results[n2 - 1 - j];

			if (dist > max) {
				pair.key = dist;
				pair.id1 = base1 + i + 1;
				pair.id2 = h_one_ids[n2 - 1 - j] + 1;
				max = update_furthest_pair(k, &pair, furthest_pair);
			}
			else {
				break;				// no better results, stop early
			}
		}
	}

	// -------------------------------------------------------------------------
	//  step 5: release memory
	// -------------------------------------------------------------------------
	tmp_data.clear(); tmp_data.shrink_to_fit();
	tmp_query.clear(); tmp_query.shrink_to_fit();	
}

// -----------------------------------------------------------------------------
__global__ void euclidean(			// compute Euclidean distance
	int n,						// cardinality of data
	int qn,						// cardinality of query
	int d,						// dimensionality
	float *data,						// data objects
	float *query,						// query objects
	float *results) 					// all-pair results (return)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int size  = n * qn;

	if (index >= size) return;

	// -------------------------------------------------------------------------
	//  get source index
	// -------------------------------------------------------------------------
	int src_index  = (index / qn) * d;
	int dest_index = (index % qn) * d;

	float dist = 0.0f;
	for (int i = 0; i < d; ++i) {
		float diff = data[src_index + i] - query[dest_index + i]; 
		dist += diff * diff;
	}
	results[index] = sqrt(dist);
}
