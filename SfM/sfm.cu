#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "kernels.h"
#include "sfm.h"
#include <assert.h>
#include <cublas_v2.h>
#include <curand.h>
#include <memory>
#include <string>
#include <iostream>
#include <fstream>
#include <cublas_v2.h>
#include <CudaSift/cudaSift.h>
#include <curand.h>
#include <iomanip>      
#include <random>
#include <algorithm>
#include <thrust/extrema.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cusolverDn.h>
#include "svd.h"


namespace SfM {

	Image_pair::Image_pair(float k[9], float k_inv[9], int image_count, int num_points) :image_count(image_count), num_points(num_points) { // num_points should be a array if we want to deal with more than 2 images
		cudaMalloc((void**)&d_K, 3 * 3 * sizeof(float));
		checkCUDAErrorWithLine("Malloc failed!");
		cudaMalloc((void**)&d_K_inv, 3 * 3 * sizeof(float));
		checkCUDAErrorWithLine("Malloc failed!");
		cudaMemcpy(d_K, k, 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_K_inv, k_inv, 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);
		// Allocate Point space
		float *d_u, *d_x;
		for (int i = 0; i < image_count; i++) {
			cudaMalloc((void**)&d_u, 3 * num_points * sizeof(float));
			U.push_back(d_u);
			cudaMalloc((void**)&d_x, 3 * num_points * sizeof(float));
			X.push_back(d_x);
		}
		// E
		cudaMalloc((void **)&d_E, 3 * 3 * sizeof(float));
		// candidate R, T
		cudaMalloc((void **)&d_P, 4 * 4 * 4 * sizeof(float));
		// uniform svd handles
		float residual = 0;
		int executed_sweeps = 0;
		const float tol = 1.e-7;
		const int max_sweeps = 15;
		const int sort_svd = 1;
		using namespace std;
		cusolverDnCreate(&cusolverH);
		cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
		checkCUDAError("Could not create flags");
		cusolverDnSetStream(cusolverH, stream);
		checkCUDAError("Could not Set strea,");
		cusolverDnCreateGesvdjInfo(&gesvdj_params);
		checkCUDAError("Could not create GesvdjInfo");
		cusolverDnXgesvdjSetTolerance(
			gesvdj_params,
			tol);
		checkCUDAError("Could not SetTolerance");
		cusolverDnXgesvdjSetMaxSweeps(
			gesvdj_params,
			max_sweeps);
		checkCUDAError("Could not SetMaxSweeps");
		cusolverDnXgesvdjSetSortEig(
			gesvdj_params,
			sort_svd);
		checkCUDAError("Could not SetSortEigs");

		// cublas handle
		cublasCreate(&handle);
		// Space for final points
		cudaMalloc((void **)&d_final_points, 4 * num_points * sizeof(float));
	}
	
	void Image_pair::fillXU(SiftPoint *data) {
		kernels::printMatrix(d_K_inv, 3, 3, 3, "K inv");
		// Fill U
		dim3 grids(ceil((num_points + cuda_block_size - 1) / cuda_block_size), 1);
		dim3 blocks(cuda_block_size, 3);
		kernels::copy_point << <grids, blocks >> > (data, num_points, U[0], U[1]);
		kernels::printMatrix(U[0], 3, num_points, 5, "U[0]");
		// Fill X using X = inv(K) * U
		kernels::gpu_blas_mmul(d_K_inv, U[0], X[0], 3, 3, num_points, handle);
		kernels::gpu_blas_mmul(d_K_inv, U[1], X[1], 3, 3, num_points, handle);
		kernels::printMatrix(X[0], 3, num_points, 2, "X[0]");
		kernels::printMatrix(X[1], 3, num_points, 2, "X[1]");
	}

	void Image_pair::estimateE() {
		const int ransac_count = floor((num_points / 8));
		// Create random order of points (on cpu using std::shuffle)
		int *indices = new int[num_points];
		int *d_indices;
		cudaMalloc((void **)&d_indices, num_points * sizeof(int));
		for (int i = 0; i < num_points; indices[i] = i, i++);
		// Shufle data
		std::random_device rd;
		std::mt19937 g(rd());
		shuffle(indices, indices + num_points, g);
		// Copy data to gpu
		cudaMemcpy(d_indices, indices, num_points * sizeof(int), cudaMemcpyHostToDevice);
		// Calculate all kron products correctly
		float *d_A;
		cudaMalloc((void **)&d_A, 8 * 9 * ransac_count * sizeof(float));
		checkCUDAErrorWithLine("A malloc failed!");
		int grids = ceil((ransac_count + cuda_block_size - 1) / cuda_block_size);
		kernels::kernels << <grids, cuda_block_size >> > (X[0], X[1], d_A, d_indices, ransac_count, num_points);
		checkCUDAErrorWithLine("Kron failed!");

		float *d_E_candidate;
		cudaMalloc((void **)&d_E_candidate, 3 * 3 * ransac_count * sizeof(float));
		// Calculate batch SVD of d_A
		float *d_ut, *d_vt, *d_s;
		cudaMalloc((void **)&d_ut, 8 * 8 * ransac_count * sizeof(float));
		cudaMalloc((void **)&d_vt, 9 * 9 * ransac_count * sizeof(float));
		cudaMalloc((void **)&d_s, 8 * ransac_count * sizeof(float));
		int *d_info = NULL;
		cudaMalloc((void**)&d_info, 4 * sizeof(int));
		kernels::regular_svd(d_A, d_ut, d_s, d_vt, 8, 9, ransac_count, d_info, cusolverH, gesvdj_params);
		// Last column of V becomes E (row of v' in our case)
		int blocks = ceil((ransac_count + cuda_block_size - 1) / cuda_block_size);
		kernels::row_extraction_kernel << <blocks, cuda_block_size >> > (d_vt, d_E_candidate, ransac_count);
		// Calculate target E's
		kernels::normalizeE << <grids, cuda_block_size >> > (d_E_candidate, ransac_count);
		kernels::print3DSlice(d_E_candidate, 3, 3, 0, 3, "dE candidates[0]");
		// Calculate number of inliers for each E
		int *d_inliers = calculateInliers(d_E_candidate, ransac_count);
		kernels::printVector(d_inliers, ransac_count, "inliers");
		// Pick best E and allocate d_E and E using thrust
		thrust::device_ptr<int> dv_in(d_inliers);
		auto iter = thrust::max_element(dv_in, dv_in + ransac_count);
		int best_pos = (iter - dv_in) - 1;
		kernels::printVector(d_inliers + best_pos, 3, "best inliers");
		// Assigne d_E
		cudaMemcpy(d_E, &(d_E_candidate[access3(0, 0, best_pos, 3, 3)]), 3 * 3 * sizeof(float), cudaMemcpyDeviceToDevice);
		kernels::printMatrix(d_E, 3, 3, 3, "d_E!!!");
		// Free stuff
		cudaFree(d_A);
		// svd free
		cudaFree(d_ut);
		cudaFree(d_s);
		cudaFree(d_vt);
		cudaFree(d_info);
		cudaFree(d_indices);
		cudaFree(d_inliers);
		cudaFree(d_E_candidate);
		free(indices);
	}

	int * Image_pair::calculateInliers(float *d_E_candidate, int ransac_iter) {
		/// This function calculates n1, d1, n2, d2 and then finds the number of residuals per E candidate in X[0] and X[1]
		// Init E1
		float E1[9] = { 0, -1, 0, 1, 0, 0, 0, 0, 0 };
		float *d_E1;
		cudaMalloc((void **)&d_E1, 9 * sizeof(float));
		cudaMemcpy(d_E1, E1, 9 * sizeof(float), cudaMemcpyHostToDevice);
		// Allocs
		float *x1_transformed, *x2_transformed;
		cudaMalloc((void**)&x1_transformed, 3 * num_points * ransac_iter * sizeof(float));
		cudaMalloc((void**)&x2_transformed, 3 * num_points * ransac_iter * sizeof(float));
		float *d1, *d2;
		cudaMalloc((void**)&d1, 3 * num_points * ransac_iter * sizeof(float));
		cudaMalloc((void**)&d2, 3 * num_points * ransac_iter * sizeof(float));
		float *n1, *n2;
		cudaMalloc((void **)&n1, 3 * num_points * ransac_iter * sizeof(float));
		cudaMalloc((void **)&n2, 3 * num_points * ransac_iter * sizeof(float));
		// Calculate x1 (from matlab code) {
		int m = 3, k = 3, n = num_points;
		kernels::gpu_blas_mmul_batched(d_E_candidate, X[0], x1_transformed, m, k, n, m * k, 0, m * n, ransac_iter, handle);

		//Compute n1 
		m = num_points, k = 3, n = 3; // these probably need to change because we need to transpose X[1]
		kernels::gpu_blas_mmul_transpose_batched(X[1], d_E_candidate, n1, m, k, n, 0, 3 * 3, m * n, ransac_iter, handle); // transpose X[1]
		int blocks = ceil((3 * num_points + cuda_block_size - 1) / cuda_block_size); // BUG!!! we need to make this batched
		kernels::element_wise_mult << <blocks, cuda_block_size >> > (n1, X[0], 3 * num_points);
		// Compute d1
		// d1 = E1 * x1_transformed
		m = 3, k = 3, n = num_points;
		kernels::gpu_blas_mmul_batched(d_E_candidate, x1_transformed, d1, m, k, n, m*k, 0, m* n, ransac_iter, handle);
		// }
		// Now calculate x2_transformed, n2 and d2 {
		m = 3, k = 3, n = num_points;
		kernels::gpu_blas_mmul_batched(d_E_candidate, X[1], x2_transformed, m, k, n, m*k, 0, m* n, ransac_iter, handle);
		//Compute n2
		m = num_points, k = 3, n = 3; // these probably need to change because we need to transpose X[0]
		kernels::gpu_blas_mmul_transpose_batched(X[0], d_E_candidate, n2, m, k, n, 0, 3 * 3, m * n, ransac_iter, handle); // transpose X[0]
		blocks = ceil((3 * num_points + cuda_block_size - 1) / cuda_block_size);
		kernels::element_wise_mult << <blocks, cuda_block_size >> > (n2, X[1], 3 * num_points);
		// Compute d2
		m = 3, k = 3, n = num_points;
		kernels::gpu_blas_mmul_batched(d_E_candidate, x2_transformed, d2, m, k, n, m*k, 0, m* n, ransac_iter, handle);
		// }
		// Now calculate the residual per candidate E{
		float *norm_n1, *norm_n2, *norm_d1, *norm_d2;
		int *d_inliers;
		int size = num_points * ransac_iter;
		cudaMalloc((void**)&norm_n1, size * sizeof(float));
		cudaMalloc((void**)&norm_n2, size * sizeof(float));
		cudaMalloc((void**)&norm_d1, size * sizeof(float));
		cudaMalloc((void**)&norm_d2, size * sizeof(float));
		cudaMalloc((void**)&d_inliers, ransac_iter * sizeof(int));
		blocks = ceil((num_points * ransac_iter + cuda_block_size - 1) / cuda_block_size);
		kernels::vecnorm << <blocks, cuda_block_size >> > (n1, norm_n1, 3, size, 1, 2);
		kernels::vecnorm << <blocks, cuda_block_size >> > (n2, norm_n2, 3, size, 1, 2);

		kernels::vecnorm << <blocks, cuda_block_size >> > (d1, norm_d1, 3, size, 2, 2);
		kernels::vecnorm << <blocks, cuda_block_size >> > (d1, norm_d1, 3, size, 2, 2);

		kernels::element_wise_div << <blocks, cuda_block_size >> > (norm_n1, norm_d1, size);
		kernels::element_wise_div << <blocks, cuda_block_size >> > (norm_n2, norm_d2, size);
		// We now have the residuals in norm_n1
		kernels::element_wise_sum << <blocks, cuda_block_size >> > (norm_n1, norm_n2, size);
		// Calculate inliers per cell
		blocks = ceil((ransac_iter + cuda_block_size - 1) / cuda_block_size);
		kernels::threshold_count << <blocks, cuda_block_size >> > (norm_n1, d_inliers, num_points, ransac_iter, 1e-6); 
		//}
		// Not sure if we should free
		cudaFree(n1);
		cudaFree(n2);
		cudaFree(d1);
		cudaFree(d2);
		cudaFree(x1_transformed);
		cudaFree(x2_transformed);
		// Free the normies!!!
		cudaFree(norm_n1);
		cudaFree(norm_n2);
		cudaFree(norm_d1);
		cudaFree(norm_d2);
		cudaFree(d_E1);
		return d_inliers;
	}

	void Image_pair::computePosecandidates() {
		float E[9];
		cudaMemcpy(E, d_E, 3 * 3 * sizeof(float), cudaMemcpyDeviceToHost);
		float u[9], d[9], v[9], tmp[9];
		svd(E, u, d, v); // v is not transposed
		multABt(u, v, tmp); // u * v'
		if (det(tmp) < 0)
			neg(v);
		float *d_u, *d_v;
		d_u = kernels::cuda_alloc_copy(u, 3 * 3);
		d_v = kernels::cuda_alloc_copy(v, 3 * 3);
		kernels::candidate_kernels << <1, 32 >> > (d_P, d_u, d_v);
		cudaFree(d_u);
		cudaFree(d_v);
	}

	void Image_pair::choosePose() {
		// take 1 point and figure out if it is in front of the camera or behind
		float P1[16] = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 }; // I(4)
		float *d_P1 = kernels::cuda_alloc_copy(P1, 16);
		float *d_A, *d_u, *d_d, *d_vt;
		cudaMalloc((void **)&d_A, 4 * 4 * 4 * sizeof(float));
		cudaMalloc((void **)&d_u, 4 * 4 * 4 * sizeof(float));
		cudaMalloc((void **)&d_d, 4 * 4 * sizeof(float));
		cudaMalloc((void **)&d_vt, 4 * 4 * 4 * sizeof(float));
		// Create A

		dim3 blocks(1, 1);
		dim3 block_sizes(4, 2);
		kernels::compute_linear_triangulation_A << <blocks, block_sizes >> > (d_A, X[0], X[1], 4, num_points, d_P1, d_P, -1, true);
		kernels::print3DSlice(d_A, 4, 4, 0, 4, "d_A");
		// We only care about V
		float *d_d1, *d_d2; // 3x4 batched
		cudaMalloc((void **)&d_d1, 4 * 4 * sizeof(float));
		cudaMalloc((void **)&d_d2, 4 * 4 * sizeof(float));
		// Assumes V isnt transposed, we need to take the last row
		// svd(d_A, d_u, d_d, d_v, 4 batches)
		checkCUDAErrorWithLine("Before SVD");
		int *d_info = NULL;
		cudaMalloc((void**)&d_info, 4 * sizeof(int));
		kernels::svd_square(d_A, d_vt, d_d, d_u, 4, 4, 4, d_info, cusolverH, stream, gesvdj_params);
		checkCUDAErrorWithLine("SVD");
		kernels::normalize_pt_kernal << <1, 4 >> > (d_vt, d_d1, 4);
		kernels::printMatrix(d_d1, 4, 4, 4, "d1");

		float val_d1, val_d2;
		P_ind = 0;
		for (int i = 0; i < 4; i++) { // batched doesn't work for inverse + it is only 4, 4x4 matrices, should be easy
			kernels::invert(d_P + i * 4 * 4, d_P + i * 4 * 4, 4, 1, handle);
			int m = 4, k = 4, n = 4;
			kernels::gpu_blas_mmul(d_P + i * 4 * 4, d_d1, d_d2, m, k, n, handle);
			kernels::print3DSlice(d_P + i * 4 * 4, 4, 4, 0, 4, "d_P2_inv");
			kernels::printMatrix(d_d2, 4, 4, 4, "d2");
			// Do the final testing on the host
			cudaMemcpy(&val_d1, &(d_d1[access2(2, i, 4)]), sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(&val_d2, &(d_d2[access2(2, i, 4)]), sizeof(float), cudaMemcpyDeviceToHost);
			// Now we do the final check on the cpu as well, because it is the same ease
			if (val_d1 > 0 && val_d2 > 0)
				P_ind = i;
		}
		kernels::printMatrix(&(d_P[access3(0, 0, P_ind, 4, 4)]), 4, 4, 4, "Chosen Projection matrix");
		cudaFree(d_P1);
		cudaFree(d_A);
		cudaFree(d_u);
		cudaFree(d_d);
		cudaFree(d_vt);
		cudaFree(d_d1);
		cudaFree(d_d2);
		cudaFree(d_info);
	}

	void Image_pair::linear_triangulation() {
		// Similar to choosePose, except we know the pose we want so we don't need to do the later half of the computations
		// Tested
		float P1[16] = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 }; // I(4)
		float *d_P1 = kernels::cuda_alloc_copy(P1, 16);
		float *d_A, *d_u, *d_d, *d_vt;
		cudaMalloc((void **)&d_A, 4 * 4 * num_points * sizeof(float));
		cudaMalloc((void **)&d_u, 4 * 4 * num_points * sizeof(float));
		cudaMalloc((void **)&d_d, 4 * num_points * sizeof(float));
		cudaMalloc((void **)&d_vt, 4 * 4 * num_points * sizeof(float));
		// Create A

		dim3 grids(ceil((num_points * 2 + cuda_block_size - 1) / cuda_block_size), 1);
		dim3 block_sizes(cuda_block_size / 2, 2);
		kernels::compute_linear_triangulation_A << <grids, block_sizes >> > (d_A, X[0], X[1], num_points, num_points, d_P1, d_P, P_ind, false);
		checkCUDAError("A computation error");
		kernels::print3DSlice(d_A, 4, 4, 0, 4, "A[0]");
		// Assumes V isnt transposed, we need to take the last column
		int *d_info = NULL;
		cudaMalloc((void**)&d_info, 4 * sizeof(int));
		kernels::svd_square(d_A, d_vt, d_d, d_u, 4, 4, num_points, d_info, cusolverH, stream, gesvdj_params);
		checkCUDAError("SVD error");
		kernels::print3DSlice(d_d, 4, 4, 0, 4, "d_vt[0]");
		dim3 grids2(ceil((num_points + cuda_block_size - 1) / cuda_block_size), 1);
		dim3 block_sizes2(cuda_block_size, 4);
		// Normalize by using the last row of v'
		kernels::normalize_pt_kernal << <grids2, block_sizes2 >> > (d_vt, d_final_points, num_points);
		kernels::printMatrix(d_final_points, 3, num_points, 5, "Transformed points");
		cudaFree(d_P1);
		cudaFree(d_A);
		cudaFree(d_u);
		cudaFree(d_d);
		cudaFree(d_vt);
		cudaFree(d_info);

	}

	Image_pair::~Image_pair() {
		cudaFree(d_K);
		cudaFree(d_K_inv);
		// Free vector points
		for (auto x : X)
			cudaFree(x);
		for (auto x : U)
			cudaFree(x);
		cudaFree(d_final_points);
		// E
		cudaFree(d_P);
		cudaFree(d_E);
		cublasDestroy(handle);
	}
	////////////////////////////////////////
	///////			  VIZ         //////////
	////////////////////////////////////////
	__host__ __device__ unsigned int hash(unsigned int a) {
		a = (a + 0x7ed55d16) + (a << 12);
		a = (a ^ 0xc761c23c) ^ (a >> 19);
		a = (a + 0x165667b1) + (a << 5);
		a = (a + 0xd3a2646c) ^ (a << 9);
		a = (a + 0xfd7046c5) + (a << 3);
		a = (a ^ 0xb55a4f09) ^ (a >> 16);
		return a;
	}


	void Image_pair::copyBoidsToVBO(float *vbodptr_positions, float *vbodptr_velocities) {
		dim3 fullBlocksPerGrid((num_points + cuda_block_size - 1) / cuda_block_size);
		checkCUDAErrorWithLine("Not copyBoidsToVBO failed!");
		kernels::kernCopyPositionsToVBO << <fullBlocksPerGrid, cuda_block_size >> > (num_points, d_final_points, vbodptr_positions, 1);
		kernels::kernCopyVelocitiesToVBO << <fullBlocksPerGrid, cuda_block_size >> > (num_points, vbodptr_velocities, 1);

		checkCUDAErrorWithLine("copyBoidsToVBO failed!");

		cudaDeviceSynchronize();
	}


	////////////////////////////////////////
	///////			Testing       //////////
	////////////////////////////////////////
	void Image_pair::testBatchedmult() {
		// C(m,n) = A(m,k) * B(k,n)
		float A[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8 };
		float B[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
		float *d_A, *d_B, *d_C;
		// Alloc
		cudaMalloc((void**)&d_A, 9 * 2 * sizeof(float));
		cudaMalloc((void**)&d_B, 9 * sizeof(float));
		cudaMalloc((void**)&d_C, 6 * 3 * sizeof(float));
		// Copy
		cudaMemcpy(d_A, A, 9 * 2 * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_B, B, 9 * sizeof(float), cudaMemcpyHostToDevice);

		int m = 3, k = 1, n = 3;
		cublasHandle_t handle;
		cublasCreate(&handle);
		const float alf = 1; // gpu vs cpu
		const float bet = 0;
		const float *alpha = &alf;
		const float *beta = &bet;
		int sA = 3;
		int sB = 0;
		int sC = 3;
		int batches = 6;
		/*cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
			m, n, k, alpha, d_A, lda, 9, d_B, ldb, 0,
			beta, d_C, ldc, 3, 2);*/
		cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, alpha, d_B, n, sB, d_A, k, sA, beta, d_C, n, sC, batches);
		cublasDestroy(handle);
		kernels::printVector(d_C, 6 * 3, "C");
		// Free test stuff
		cudaFree(d_A);
		cudaFree(d_B);
		cudaFree(d_C);
	}
	void Image_pair::testSVD() {
		float b[4 * 4 * 2] = { 1,2,3,4,5,6,7,8,9,10,11,12,1,2,3,4,5,6,7,8,10,11,12,14 };
		float *d_b = kernels::cuda_alloc_copy(b, 4 * 4 * 2);
		kernels::printMatrix(d_b, 4, 4, 4, "b");
		float *d_VT = NULL;
		float *d_S = NULL;
		float *d_U = NULL;
		int *d_info = NULL;

		cudaMalloc((void**)&d_VT, sizeof(float) * 4 * 4 * 2);
		cudaMalloc((void**)&d_U, sizeof(float) * 4 * 4 * 2);
		cudaMalloc((void**)&d_S, sizeof(float) * 4 * 2);
		cudaMalloc((void**)&d_info, 4 * sizeof(int));
		kernels::svd_square(d_b, d_VT, d_S, d_U, 4, 4, 2, d_info, cusolverH, stream, gesvdj_params);
		kernels::printMatrix(d_VT, 4, 4, 4, "VT");
		kernels::printMatrix(d_S, 4, 2, 2, "S");
		kernels::printMatrix(d_U, 4, 4, 4, "U");
	}
	void Image_pair::testInverse() {
		// Conclusion, batched doesn't work
		float a[] = { 0.9649, 0.9572, 0.1419, 0.1576, 0.4854, 0.4218,0.9706, 0.8003, 0.9157,
						1, 2, 0, 0, 2, 0, 1, 2, 1 };
		float *d_A = kernels::cuda_alloc_copy(a, 18);
		float *d_b;
		cudaMalloc((void **)&d_b, 18 * sizeof(float));
		kernels::invert(d_A + 9, d_b, 3, 1, handle);
		kernels::printMatrix(d_b, 3, 3, 3, "b[0]");
		kernels::print3DSlice(d_b, 3, 3, 1, 3, "b[1]");
		cudaFree(d_A);
		cudaFree(d_b);
	}
	void Image_pair::testThrust_max() {
		int a[] = { 1,2,3,4,5,6, 4, 1, 3 };
		int *d_A = kernels::cuda_alloc_copy<int>(a, 7);

		thrust::device_ptr<int> dv_in(d_A);
		auto iter = thrust::max_element(dv_in, dv_in + 6);

		unsigned int position = iter - dv_in;
		int max_val = *iter;
		cudaFree(d_A);
		std::cout << "The maximum value is " << max_val << " at position " << position << std::endl;
	}
	void Image_pair::testBatchedmultTranspose() {
		// Verdict: Works!!!
		float A[] = { 1, 2, 3, 1,
					  4, 5, 6, 1,
					  7, 8, 9, 1,

					  0, 1, 2, 1,
					  3, 4, 5, 1,
					  6, 7, 8, 1 };
		float B[] = { 1, 2, 3,
					  4, 5, 6,
					  7, 8, 9 };
		float *d_A, *d_B, *d_C;
		cudaMalloc((void**)& d_C, 4 * 3 * 2 * sizeof(float));
		d_A = kernels::cuda_alloc_copy(A, 4 * 3 * 2);
		d_B = kernels::cuda_alloc_copy(B, 9);
		kernels::gpu_blas_mmul_transpose_batched(d_A, d_B, d_C, 4, 3, 3, 4 * 3, 0, 4 * 3, 2, handle);
		kernels::print3DSlice(d_C, 4, 3, 0, 3, "d_C[0]");
		kernels::print3DSlice(d_C, 4, 3, 1, 3, "d_C[1]");
		cudaFree(d_A);
		cudaFree(d_B);
		cudaFree(d_C);
	}
	void Image_pair::testRow_extraction_kernel() {
		float data[9 * 9 * 9];
		for (int i = 0; i < 9 * 9 * 9; i++)
			data[i] = i;
		float *d_d = kernels::cuda_alloc_copy(data, 9 * 9 * 9);
		float *res;
		cudaMalloc((void**)&res, 9 * 9 * sizeof(float));

		kernels::row_extraction_kernel << <1, 9 >> > (d_d, res, 9);
		kernels::printMatrix(res, 9, 9, 9, "res");
		cudaFree(res);
		cudaFree(d_d);
	}
	void Image_pair::testVecnorm() {
		float test[9] = { 1,2,3,4,5,6,7,8,9 };
		float *test_gpu = kernels::cuda_alloc_copy(test, 9 * sizeof(float));
		float *norm_n1;
		cudaMalloc((void**)&norm_n1, 3 * sizeof(float));
		kernels::vecnorm << <1, 3 >> > (test_gpu, norm_n1, 3, 3, 2, 1);
		kernels::printMatrix(norm_n1, 3, 1, 1, "norm");
	}
}