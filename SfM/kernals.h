#pragma once

#include "common.h"
#include <vector>
#include <curand.h>
#include <cublas_v2.h>
#include "svd.h"
#include <CudaSift/cudaSift.h>
#include <vector>
using namespace std;
#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)
namespace Kernals {
	Common::PerformanceTimer& timer();
}

namespace SfM {
#define cuda_block_size 256
	class Image_pair {
		float *d_K;
		float *d_K_inv;
		int image_count;
		int num_points;
		float *d_E;
		int P_ind;
		// Canidate Transforms
		float *d_P; // 4 elements
		// Points
		vector<float *> U; // We only use 2, this can be "extended"
		vector<float *> X;
		// Handles
		cublasHandle_t handle;
		// Internal functions
		void batch_svd_device(float *A, float *U, float *S, float *V, int m, int n, const int batchSize, int *d_info);
		int * calculateInliers(float *d_E_canidate, int ransac_iter);
	public:
		Image_pair(float k[9], float k_inv[9], int image_count, int num_points);
		void fillXU(SiftPoint *data);
		void estimateE();
		void testSVD();
		void computePoseCanidates();
		void choosePose();
		void testBatchedmult();
		void testThrust_max();
		~Image_pair() {
			cudaFree(d_K);
			cudaFree(d_K_inv);
			// Free vector points
			for (auto x : X)
				cudaFree(x);
			for (auto x : U)
				cudaFree(x);
			// E
			cudaFree(d_P);
			cudaFree(d_E);
		}
	};
}