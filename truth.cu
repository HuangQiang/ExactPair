#include "headers.h"


// -----------------------------------------------------------------------------
void ground_truth(					// ground truth of closest/furthest pairs
	int    n,							// cardinality
	int    d,							// dimensionality
	int    m,							// number of division
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
	Pair *closest_pair  = new Pair[k];
	Pair *furthest_pair = new Pair[k];
	
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
		}
	}
	gettimeofday(&end_time, NULL);
	float kernel_time = end_time.tv_sec - start_time.tv_sec + 
		(end_time.tv_usec - start_time.tv_usec) / 1000000.0f;
	printf("Call Kernel Function: %f Seconds\n\n", kernel_time);

	// -------------------------------------------------------------------------
	//  step 3: write the closest/furthest pairs ground truth results to disk
	// -------------------------------------------------------------------------
	FILE *fp = fopen(closest_set.c_str(), "w");
	fprintf(fp, "%d\n", k);
	for (int i = 0; i < k; ++i) {
		fprintf(fp, "%d %d %f\n", closest_pair[i].id1, 
			closest_pair[i].id2, closest_pair[i].key);
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
	// -------------------------------------------------------------------------
	int num = (int) ceil((double) n / (double) m);

	int n1 = num;
	int n2 = num;
	if (id1 == m - 1) n1 = n - num * (m - 1); 
	if (id2 == m - 1) n2 = n - num * (m - 1);

	int base1 = id1 * num;
	int base2 = id2 * num;

	int data_size   = n1 * d;
	int query_size  = n2 * d;
	int result_size = n1 * (k + 1);

	int num_threads = 1024;
	int num_blocks  = (n1 + num_threads - 1) / num_threads;

	printf("(%d, %d):\n", id1 + 1, id2 + 1);
	printf("data size   = %d\n", data_size);
	printf("query size  = %d\n", query_size);
	printf("result size = %d\n", result_size);
	printf("num threads = %d, num blocks  = %d\n\n", num_threads, num_blocks);

	// -------------------------------------------------------------------------
	//  step 2: flatten and copy high-dimensional data
	// -------------------------------------------------------------------------
	vector<float> tmp_data;
	for (int j = base1; j < base1 + n1; ++j) {
		tmp_data.insert(end(tmp_data), begin(data[j]), end(data[j]));
	}
	thrust::device_vector<float> d_data(data_size, 0.0f);
	thrust::copy(&(tmp_data[0]), &(tmp_data[data_size]), d_data.begin());

	vector<float> tmp_query;
	for (int j = base2; j < base2 + n2; ++j) {
		tmp_query.insert(end(tmp_query), begin(data[j]), end(data[j]));
	}
	thrust::device_vector<float> d_query(query_size, 0.0f);
	thrust::copy(&(tmp_query[0]), &(tmp_query[query_size]), d_query.begin());

	thrust::device_vector<float> d_cp_dist(result_size, MAXREAL);
	thrust::device_vector<int>   d_cp_index(result_size, -1);
	thrust::device_vector<float> d_fp_dist(result_size, MINREAL);
	thrust::device_vector<int>   d_fp_index(result_size, -1);

	// -------------------------------------------------------------------------
	//  step 3: call kernel function
	// -------------------------------------------------------------------------
	euclidean<<<num_blocks, num_threads>>>(
		n1, n2, d, k, base1, base2,
		thrust::raw_pointer_cast(d_data.data()),
		thrust::raw_pointer_cast(d_query.data()),		
		thrust::raw_pointer_cast(d_cp_dist.data()),
		thrust::raw_pointer_cast(d_cp_index.data()),
		thrust::raw_pointer_cast(d_fp_dist.data()),
		thrust::raw_pointer_cast(d_fp_index.data()));

	cudaDeviceSynchronize();

	// -------------------------------------------------------------------------
	//  step 3: update closest/furthest pair results from GPU to CPU
	// -------------------------------------------------------------------------
	thrust::host_vector<float> h_cp_dist  = d_cp_dist;
	thrust::host_vector<int>   h_cp_index = d_cp_index;
	thrust::host_vector<float> h_fp_dist  = d_fp_dist;
	thrust::host_vector<int>   h_fp_index = d_fp_index;

	Pair  pair;
	float truth_time = -1.0f;
	float min = MAXREAL;
	float max = MINREAL;

	for (int i = 0; i < n1; ++i) {
		int base = i * (k + 1);

		// ---------------------------------------------------------------------
		//  step 3.1: update closest pair results
		// ---------------------------------------------------------------------
		for (int j = 0; j < k; ++j) {
			float dist = h_cp_dist[base + j];

			if (dist < min) {
				pair.key = dist;
				pair.id1 = base1 + i + 1;
				pair.id2 = base2 + h_cp_index[base + j] + 1;
				min = update_closest_pair(k, &pair, closest_pair);
			}
			else break;
		}

		// ---------------------------------------------------------------------
		//  step 3.2: update furthest pair results
		// ---------------------------------------------------------------------
		for (int j = 0; j < k; ++j) {
			float dist = h_fp_dist[base + j];

			if (dist > max) {
				pair.key = dist;
				pair.id1 = base1 + i + 1;
				pair.id2 = base2 + h_fp_index[base + j] + 1;
				max = update_furthest_pair(k, &pair, furthest_pair);
			}
			else break;
		}
	}
}

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
	int   *fp_index)					// furthest index (return)
{
	int index = (int) blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= n) return;

	// -------------------------------------------------------------------------
	//  get source index
	// -------------------------------------------------------------------------
	int src_index  = index * d;
	int rslt_index = index * (k + 1);

	int i = -1;
	int j = -1;
	for (i = 0; i < qn; ++i) {
		if (base1 + index == base2 + i) return;

		// ---------------------------------------------------------------------
		//  calculate Euclidean distance
		// ---------------------------------------------------------------------
		int dest_index = i * d;
		float dist = 0.0f;
		for (j = 0; j < d; ++j) {
			float diff = data[src_index + j] - query[dest_index + j]; 
			dist += diff * diff;
		}
		dist = sqrt(dist);

		// ---------------------------------------------------------------------
		//  update closest pairs results
		// ---------------------------------------------------------------------
		j = 0;
		for (j = k; j > 0; --j) {
			if (cp_dist[rslt_index + j - 1] > dist) {
				cp_dist[rslt_index  + j] = cp_dist[rslt_index  + j - 1];
				cp_index[rslt_index + j] = cp_index[rslt_index + j - 1];
			}
			else break;
		}
		cp_dist[rslt_index + j]  = dist;
		cp_index[rslt_index + j] = i;

		// ---------------------------------------------------------------------
		//  update furthest pairs results
		// ---------------------------------------------------------------------
		j = 0;
		for (j = k; j > 0; --j) {
			if (fp_dist[rslt_index + j - 1] < dist) {
				fp_dist[rslt_index  + j] = fp_dist[rslt_index  + j - 1];
				fp_index[rslt_index + j] = fp_index[rslt_index + j - 1];
			}
			else break;
		}
		fp_dist[rslt_index + j]  = dist;
		fp_index[rslt_index + j] = i;
	}
}


// // -----------------------------------------------------------------------------
// void ground_truth(					// ground truth of closest/furthest pairs
// 	int n,								// cardinality
// 	int d,								// dimensionality
// 	string data_set,					// file name of data set
// 	string closest_set,					// file name of closest set
// 	string furthest_set)				// file name of furthest set
// {
// 	timeval start_time, end_time;

// 	// -------------------------------------------------------------------------
// 	//  step 1: read data set
// 	// -------------------------------------------------------------------------
// 	gettimeofday(&start_time, NULL);
// 	vector<vector<float> > data(n, vector<float>(d, 0.0));
// 	if (read_set(n, d, data_set, data)) {
// 		printf("Reading Dataset Error!\n");
// 		exit(1);
// 	}

// 	gettimeofday(&end_time, NULL);
// 	float read_file_time = end_time.tv_sec - start_time.tv_sec + 
// 		(end_time.tv_usec - start_time.tv_usec) / 1000000.0;
// 	printf("Read Dataset: %f Seconds\n\n", read_file_time);

// 	// -------------------------------------------------------------------------
// 	//  step 2.1: set up parameters for kernel function
// 	// -------------------------------------------------------------------------
// 	gettimeofday(&start_time, NULL);
// 	int data_size   = n * d;
// 	int result_size = n * n;

// 	int num_threads = 1024;
// 	int num_blocks  = (result_size + num_threads - 1) / num_threads;

// 	printf("data_size   = %d, result_size = %d\n", data_size, result_size);
// 	printf("num_threads = %d, num_blocks  = %d\n", num_threads, num_blocks);

// 	// -------------------------------------------------------------------------
// 	//  step 2.2: flatten and copy the time series data
// 	// -------------------------------------------------------------------------
// 	vector<float> tmp;
// 	for (int j = 0; j < n; ++j) {
// 		tmp.insert(end(tmp), begin(data[j]), end(data[j]));
// 	}

// 	thrust::device_vector<float> d_all_results(result_size, 0.0);
// 	thrust::device_vector<float> d_data(data_size, 0.0);
// 	thrust::copy(&(tmp[0]), &(tmp[data_size]), d_data.begin());

// 	// -------------------------------------------------------------------------
// 	//  step 2.3: call kernel function
// 	// -------------------------------------------------------------------------
// 	euclidean<<<num_blocks, num_threads>>>(
// 		n, d, 
// 		thrust::raw_pointer_cast(d_data.data()), 
// 		thrust::raw_pointer_cast(d_all_results.data()));

// 	cudaDeviceSynchronize();

// 	gettimeofday(&end_time, NULL);
// 	float kernel_time = end_time.tv_sec - start_time.tv_sec + 
// 		(end_time.tv_usec - start_time.tv_usec) / 1000000.0;
// 	printf("Call Kernel Function: %f Seconds\n\n", kernel_time);

// 	// -------------------------------------------------------------------------
// 	//  step 3: sort and copy results from GPU to CPU
// 	// -------------------------------------------------------------------------
// 	thrust::host_vector<float> h_all_results = d_all_results;
// 	thrust::host_vector<float> h_one_results(n - 1);
// 	thrust::host_vector<int> h_one_ids(n - 1);

// 	int k = MAXK;
// 	Pair *closest_pair  = new Pair[k];
// 	Pair *furthest_pair = new Pair[k];

// 	for (int i = 0; i < k; ++i) {
// 		closest_pair[i].key = MAXREAL;
// 		closest_pair[i].id1 = -1;
// 		closest_pair[i].id2 = -1;

// 		furthest_pair[i].key = MINREAL;
// 		furthest_pair[i].id1 = -1;
// 		furthest_pair[i].id2 = -1;
// 	}

// 	float truth_time = -1.0;
// 	float min = MAXREAL;
// 	float max = MINREAL;
// 	Pair   pair;
	
// 	for (int i = 0; i < n; ++i) {
// 		// ---------------------------------------------------------------------
// 		//  step 3.1: get all-pairs results
// 		// ---------------------------------------------------------------------
// 		int base = i * n;
// 		int count = 0;
// 		for (int j = 0; j < n; ++j) {
// 			if (i == j) continue;

// 			h_one_results[count] = h_all_results[base + j];
// 			h_one_ids[count] = j;
// 			count++;
// 		}
// 		assert(count == n - 1);

// 		// ---------------------------------------------------------------------
// 		//  step 3.2: sort results by dist (ascending order)
// 		// ---------------------------------------------------------------------
// 		thrust::sort_by_key(h_one_results.begin(), h_one_results.end(), 
// 			h_one_ids.begin());

// 		// ---------------------------------------------------------------------
// 		//  step 3.3: update closest pair results
// 		// ---------------------------------------------------------------------
// 		for (int j = 0; j < k; ++j) {
// 			float dist = h_one_results[j];

// 			if (dist < min) {
// 				pair.key = dist;
// 				pair.id1 = i + 1;
// 				pair.id2 = h_one_ids[j] + 1;
// 				min = update_closest_pair(k, &pair, closest_pair);
// 			}
// 			else break;
// 		}

// 		// ---------------------------------------------------------------------
// 		//  step 3.4: update furthest pair results
// 		// ---------------------------------------------------------------------
// 		for (int j = 0; j < k; ++j) {
// 			float dist = h_one_results[n - 2 - j];

// 			if (dist > max) {
// 				pair.key = dist;
// 				pair.id1 = i + 1;
// 				pair.id2 = h_one_ids[n - 2 - j] + 1;
// 				max = update_furthest_pair(k, &pair, furthest_pair);
// 			}
// 			else break;
// 		}
// 		printf("ok %d, min = %f, max = %f\n", i + 1, min, max);
// 	}

// 	// -------------------------------------------------------------------------
// 	//  step 4: write the closest/furthest pairs ground truth results to disk
// 	// -------------------------------------------------------------------------
// 	FILE *fp = fopen(closest_set.c_str(), "w");
// 	fprintf(fp, "%d\n", k);
// 	for (int i = 0; i < k; ++i) {
// 		fprintf(fp, "%d %d %f\n", closest_pair[i].id1, 
// 			closest_pair[i].id2, closest_pair[i].key);
// 	}
// 	fclose(fp);

// 	fp = fopen(furthest_set.c_str(), "w");
// 	fprintf(fp, "%d\n", k);
// 	for (int i = 0; i < k; ++i) {
// 		fprintf(fp, "%d %d %f\n", furthest_pair[i].id1, furthest_pair[i].id2, 
// 			furthest_pair[i].key);
// 	}
// 	fclose(fp);

// 	// -------------------------------------------------------------------------
// 	//  step 5: release space
// 	// -------------------------------------------------------------------------
// 	delete[] closest_pair;  closest_pair  = NULL;
// 	delete[] furthest_pair; furthest_pair = NULL;
// }

// // -----------------------------------------------------------------------------
// __global__ void euclidean(			// compute Euclidean distance
// 	int n, 								// cardinality
// 	int d,								// dimensionality
// 	float *data,						// data objects
// 	float *results) 					// all-pairs results (return)
// {
// 	int index = (int) blockIdx.x * blockDim.x + threadIdx.x;
// 	int size = n * n;
// 	if (index >= size) return;

// 	// -------------------------------------------------------------------------
// 	//  compute euclidean distance between data #src_index and #dest_index
// 	// -------------------------------------------------------------------------
// 	int src_index  = (index / n) * d;
// 	int dest_index = (index % n) * d;
// 	if (src_index == dest_index) return;

// 	float dist = 0.0;
// 	for (int i = 0; i < d; ++i) {
// 		float diff = data[src_index + i] - data[dest_index + i]; 
// 		dist += diff * diff;
// 	}
// 	results[index] = sqrt(dist);
// }
