#include "headers.h"

// -----------------------------------------------------------------------------
void ground_truth(					// ground truth of closest/furthest pairs
	int   n,							// cardinality
	int   d,							// dimensionality
	int   k,							// top_k value
	const float *data,					// input data
	Pair  *closest_pair,				// closest  pairs (return)
	Pair  *furthest_pair)				// furthest pairs (return)
{
	timeval start_time, end_time;

	gettimeofday(&start_time, NULL);
	int   m = (int) ceil((float) n / sqrt(1024.0f * 65535.0f));
	float kernel_time = 0.0f;
	
	for (int i = 0; i < k; ++i) {
		closest_pair[i].key = FLT_MAX;
		closest_pair[i].id1 = -1;
		closest_pair[i].id2 = -1;

		furthest_pair[i].key = -FLT_MAX;
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
}

// -----------------------------------------------------------------------------
void partial_search(				// partial search		
	int   n,							// cardinality	
	int   d,							// dimensionaity
	int   k,							// top-k value
	int   m, 							// number of division
	int   id1,							// data slice id 1
	int   id2,							// data slide id 2
	const float *data,					// whole data
	Pair  *closest_pair,				// closest pairs (return)
	Pair  *furthest_pair)				// furthest pairs (return)
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
	thrust::device_vector<float> d_data(data+base1*d,  data+(base1 + n1)*d);
	thrust::device_vector<float> d_query(data+base2*d, data+(base2 + n2)*d);
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
	float min_val = FLT_MAX;
	float max_val = -FLT_MAX;

	for (int i = 0; i < n1; ++i) {
		// ---------------------------------------------------------------------
		//  step 4.1: get all-pairs results
		// ---------------------------------------------------------------------
		int base = i * n2;
		for (int j = 0; j < n2; ++j) {
			h_one_results[j] = h_results[base + j];
			h_one_ids[j] = base2 + j;
		}

		// ---------------------------------------------------------------------
		//  step 4.2: sort results by dist (ascending order)
		// ---------------------------------------------------------------------
		thrust::sort_by_key(h_one_results.begin(), h_one_results.end(), 
			h_one_ids.begin());

		// ---------------------------------------------------------------------
		//  step 4.3: update closest-pair results
		// ---------------------------------------------------------------------
		int size = std::min(n2 - 1, k+1);
		for (int j = 0; j <= size; ++j) {
			float dist = h_one_results[j];

			if (dist < min_val) {
				pair.key = dist;
				pair.id1 = base1 + i + 1;
				pair.id2 = h_one_ids[j] + 1;
				
				if (pair.id1 != pair.id2) {
					min_val = update_closest_pair(k, &pair, closest_pair);
				}
			}
			else break;				// no better results, stop early
		}

		// ---------------------------------------------------------------------
		//  step 4.4: update furthest-pair results
		// ---------------------------------------------------------------------
		size = std::min(n2, k);
		for (int j = 0; j < size; ++j) {
			float dist = h_one_results[n2 - 1 - j];

			if (dist > max_val) {
				pair.key = dist;
				pair.id1 = base1 + i + 1;
				pair.id2 = h_one_ids[n2 - 1 - j] + 1;
				max_val = update_furthest_pair(k, &pair, furthest_pair);
			}
			else break;				// no better results, stop early
		}
	}
}

// -----------------------------------------------------------------------------
__global__ void euclidean(			// compute Euclidean distance
	int  n,								// cardinality of data
	int  qn,							// cardinality of query
	int  d,								// dimensionality
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
