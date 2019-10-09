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
		// Points
		vector<float *> U;
		vector<float *> X;
		// Handles
		cublasHandle_t handle;
		// Internal functions
		void svd_device(float *A, float *U, float *S, float *V, int m, int n, const int batchSize, int *d_info);
	public:
		Image_pair(float k[9], float k_inv[9], int image_count, int num_points);
		void FillXU(SiftPoint *data);
		void estimateE();
		void testSVD();
		~Image_pair() {
			cudaFree(d_K);
			cudaFree(d_K_inv);
			// Free points
			for (auto x : X)
				cudaFree(x);
			for (auto x : U)
				cudaFree(x);
		}
	};
}