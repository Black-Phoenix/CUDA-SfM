#pragma once

#include "common.h"
#include <vector>
#include <curand.h>
#include <cublas_v2.h>
#include <CudaSift/cudaSift.h>
#include <vector>
#include <cusolverDn.h>

using namespace std;

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)
namespace kernels {
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
		// candidate Transforms
		float *d_P; // 4 elements
		// Points
		vector<float *> U; // We only use 2, this can be "extended"
		vector<float *> X;
		float *d_final_points;
		// svd handles
		cusolverDnHandle_t cusolverH = NULL;
		cudaStream_t stream = NULL;
		gesvdjInfo_t gesvdj_params = NULL;
		// Handles
		cublasHandle_t handle;
		// Internal functions
		void batch_svd_device(float *A, float *U, float *S, float *V, int m, int n, const int batchSize, int *d_info);
		int * calculateInliers(float *d_E_candidate, int ransac_iter);
	public:
		Image_pair(float k[9], float k_inv[9], int image_count, int num_points);
		void fillXU(SiftPoint *data);
		void estimateE();
		void testSVD();
		void computePosecandidates();
		void choosePose();
		void linear_triangulation();
		// Testing functions
		void testBatchedmult();
		void testThrust_max();
		void testInverse();
		void testBatchedmultTranspose();
		void testRow_extraction_kernel();
		void testVecnorm();
		///////
		void copyBoidsToVBO(float *vbodptr_positions, float *vbodptr_velocities);
		~Image_pair();
	};
}