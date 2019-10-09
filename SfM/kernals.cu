#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "kernals.h"
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

namespace Kernals {
#define enable_debug true
#define esp 1e-5
#define x_pos 0
#define y_pos 1
#define z_pos 2
#define access2(i, j, col) i*col + j
#define access3(i, j, k, row, col) k * row * col + i*col + j
	using Common::PerformanceTimer;
	PerformanceTimer& timer()
	{
		static PerformanceTimer timer;
		return timer;
	}
	////////////////////////////////////////
	///////			Debugging	  //////////
	////////////////////////////////////////
	template<typename T>
	void printVector(const T *a1, int n, string name) {
		if (!enable_debug)
			return;
		T *print_a = new T[n];
		cout << name.c_str() << endl;
		cout << "{" << endl;
		cudaMemcpy(print_a, a1, n * sizeof(T), cudaMemcpyDeviceToHost);
		for (int i = 0; i < n; i++) {
			cout << "\t" << print_a[i] << endl;
		}
		cout << "}" << endl;
		delete[]print_a;
	}
	template<typename T>
	void printMatrix(const T*A, int row, int col, int print_col, const char* name)
	{
		/// Prints first and last print_col values of A if A is a 2d matrix
		if (!enable_debug)
			return;
		T *print_a = new T[col*row];
		cudaMemcpy(print_a, A, row* col * sizeof(T), cudaMemcpyDeviceToHost);
		cout << name << endl;
		cout << "{" << endl;
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				if (j < print_col || j > col - print_col - 1) {
					T Areg = print_a[access2(i, j, col)];
					cout << std::fixed << setprecision(3) << "\t" << Areg;
				}
				else if (j == print_col) {
					cout << "\t....";
				}
			}
			cout << endl;
		}
		cout << "}" << endl;
		delete[]print_a;
	}
	template<typename T>
	void print3DSlice(const T*A, int row, int col, int slice, int print_col, const char* name)
	{
		/// reads 
		if (!enable_debug)
			return;
		T *print_a = new T[col * row];
		cudaMemcpy(print_a, A + row * col * slice, row * col * sizeof(T), cudaMemcpyDeviceToHost);
		cout << name << endl;
		cout << "{" << endl;
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				if (j < print_col || j > col - print_col - 1) {
					T Areg = print_a[access2(i, j, col)];
					cout << std::setw(7) << setprecision(3) << "\t" << Areg;
				}
				else if (j == print_col) {
					cout << "\t....";
				}
			}
			cout << endl;
		}
		cout << "}" << endl;
		delete[]print_a;
	}
	//////////////////////////////
	/*		Kernals			*/
	//////////////////////////////
	// C(m,n) = A(m,k) * B(k,n)
	void gpu_blas_mmul(const float *A, const float *B, float *C, const int m, const int k, const int n) {
		const float alf = 1; // gpu vs cpu
		const float bet = 0;
		const float *alpha = &alf;
		const float *beta = &bet;
		// Do the actual multiplication
		cublasHandle_t handle;
		cublasCreate(&handle);
		cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, alpha, B, n, A, k, beta, C, n);
		cublasDestroy(handle);
	}
	// C(m,n) = A(m,k) * B(k,n)
	void gpu_blas_mmul_batched(const float *A, const float *B, float *C, const int m, const int k, const int n, const int stride_A, const int stride_B, const int stride_C, const int batches) {
		assert(stride_A == 0 || stride_A == m * k);
		assert(stride_B == 0 || stride_B == n * k);
		assert(stride_C == 0 || stride_C == m * n);
		cublasHandle_t handle;
		cublasCreate(&handle);
		const float alf = 1; // gpu vs cpu
		const float bet = 0;
		const float *alpha = &alf;
		const float *beta = &bet;
		cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, alpha, B, n, stride_B, A, k, stride_A, beta, C, n, stride_C, batches);
		cublasDestroy(handle);
	}

	__global__
		void kron_kernal(float*d1, float*d2, float *A, int *indices, const int ransac_iterations, int num_points) {
		const int index = blockIdx.x*blockDim.x + threadIdx.x;
		const int A_row = 8;
		const int A_col = 9;

		if (index > ransac_iterations)
			return;
#pragma unroll
		for (int i = 0; i < A_row; i++) {
			// begin
			A[access3(i, 0, index, A_row, A_col)] = d1[access2(x_pos, indices[index * A_row + i], num_points)] * d2[access2(x_pos, indices[index * A_row + i], num_points)];
			A[access3(i, 1, index, A_row, A_col)] = d1[access2(x_pos, indices[index * A_row + i], num_points)] * d2[access2(y_pos, indices[index * A_row + i], num_points)];
			A[access3(i, 2, index, A_row, A_col)] = d1[access2(x_pos, indices[index * A_row + i], num_points)] * d2[access2(z_pos, indices[index * A_row + i], num_points)];
			// second												  			    
			A[access3(i, 3, index, A_row, A_col)] = d1[access2(y_pos, indices[index * A_row + i], num_points)] * d2[access2(x_pos, indices[index * A_row + i], num_points)];
			A[access3(i, 4, index, A_row, A_col)] = d1[access2(y_pos, indices[index * A_row + i], num_points)] * d2[access2(y_pos, indices[index * A_row + i], num_points)];
			A[access3(i, 5, index, A_row, A_col)] = d1[access2(y_pos, indices[index * A_row + i], num_points)] * d2[access2(z_pos, indices[index * A_row + i], num_points)];
			//third													  			    
			A[access3(i, 6, index, A_row, A_col)] = d1[access2(z_pos, indices[index * A_row + i], num_points)] * d2[access2(x_pos, indices[index * A_row + i], num_points)];
			A[access3(i, 7, index, A_row, A_col)] = d1[access2(z_pos, indices[index * A_row + i], num_points)] * d2[access2(y_pos, indices[index * A_row + i], num_points)];
			A[access3(i, 8, index, A_row, A_col)] = d1[access2(z_pos, indices[index * A_row + i], num_points)] * d2[access2(z_pos, indices[index * A_row + i], num_points)];
		}
	}

	__global__
		void copy_point(SiftPoint* data, int numPoints, float *U1, float *U2) {
		const int index_col = blockIdx.x*blockDim.x + threadIdx.x; // col is x to prevent warp divergence as much as possible in this naive implementation
		const int index_row = blockIdx.y*blockDim.y + threadIdx.y;
		if (index_row >= 3 || index_col >= numPoints)
			return;
		if (!index_row) {
			U1[access2(index_row, index_col, numPoints)] = data[index_col].xpos;
			U2[access2(index_row, index_col, numPoints)] = data[index_col].match_xpos;
		}
		else if (index_row == 1) {
			U1[access2(index_row, index_col, numPoints)] = data[index_col].ypos;
			U2[access2(index_row, index_col, numPoints)] = data[index_col].match_ypos;
		}
		else {
			U1[access2(index_row, index_col, numPoints)] = 1;
			U2[access2(index_row, index_col, numPoints)] = 1;
		}
	}

	__global__
		void normalizeE(float *E, int ransac_iterations) {
		const int index = blockIdx.x*blockDim.x + threadIdx.x;
		if (index >= ransac_iterations)
			return;
		float u[9], d[9], v[9];
		svd(&(E[access3(0, 0, index, 3, 3)]), u, d, v); // find correct E
		d[access2(2, 2, 3)] = 0;
		d[access2(1, 1, 3)] = 1;
		d[access2(0, 0, 3)] = 1;
		// E = U * D * V'
		float tmp_u[9];
		multAB(u, d, tmp_u);
		multABt(tmp_u, v, &(E[access3(0, 0, index, 3, 3)]));
	}

	__global__
		void element_wise_mult(float *A, float *B, int size) {
		const int index = blockIdx.x*blockDim.x + threadIdx.x;
		if (index >= size)
			return;
		A[index] *= B[index];
	}

	__global__
		void element_wise_div(float *A, float *B, int size) {
		const int index = blockIdx.x*blockDim.x + threadIdx.x;
		if (index >= size)
			return;
		float val = B[index];
		if (val == 0)
			A[index] = 0;
		else
			A[index] /= val;
	}

	__global__
		void element_wise_sum(float *A, float *B, int size) {
		const int index = blockIdx.x*blockDim.x + threadIdx.x;
		if (index >= size)
			return;
		A[index] += B[index];
	}

	__global__ 
		void vecnorm(float *A, float *res, int row, int col, int exp,int final_pow) {
		int index = blockIdx.x*blockDim.x + threadIdx.x;
		if (index >= col)
			return;
		float tmp_vlaue = 0;
#pragma unroll
		for (int i = 0; i < row; i++) {
			tmp_vlaue += powf(A[access2(i, index, col)], exp);
		}
		// Now we can take the sqrt of exp and then rais to the final_pow
		if (exp == final_pow)
			return;
		res[index] = powf(tmp_vlaue, final_pow / exp);
	}

	__global__
		void threshold_count(float *A, int *count_res, int batch_size, int ransac_count, float threshold) {
		int index = blockIdx.x*blockDim.x + threadIdx.x;
		if (index >= ransac_count)
			return;
		int count = 0;
#pragma unroll
		for (int i = 0; i < batch_size; i++) {
			if (A[i + index * batch_size] < threshold)
				count++;
		}
		count_res[index] = count;
	}

	template<typename T>
	T* cuda_alloc_copy(const T* host, int size) {
		T* data;
		cudaMalloc((void**)&data, size * sizeof(T));
		cudaMemcpy(data, host, size * sizeof(T), cudaMemcpyHostToDevice);
		return data;
	}
}

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
		// Canidate R, T
		float *d_tmp;
		for (int i = 0; i < 2; i++) {
			cudaMalloc((void **)&d_tmp, 3 * 3 * sizeof(float));
			d_R.push_back(d_tmp);
			cudaMalloc((void **)&d_tmp, 3 * sizeof(float));
			d_T.push_back(d_tmp);
		}
	}

	void Image_pair::estimateE() {
		const int ransac_count = floor(num_points / 8);
		// Create random order of points (on cpu using std::shuffle)
		int *indices = new int[num_points];
		int *d_indices;
		cudaMalloc((void **)&d_indices, num_points * sizeof(int));
		for (int i = 0; i < num_points; indices[i] = i, i++);
		// Shufle data
		std::random_device rd;
		std::mt19937 g(rd());
		//shuffle(indices, indices + num_points, g);
		// Copy data to gpu
		cudaMemcpy(d_indices, indices, num_points * sizeof(int), cudaMemcpyHostToDevice);
		// Calculate all kron products correctly
		float *d_A;
		cudaMalloc((void **)&d_A, 8 * 9 * ransac_count * sizeof(float));
		checkCUDAErrorWithLine("A malloc failed!");
		int grids = ceil((ransac_count + cuda_block_size - 1) / cuda_block_size);
		Kernals::kron_kernal << <grids, cuda_block_size >> > (X[0], X[1], d_A, d_indices, ransac_count, num_points);
		checkCUDAErrorWithLine("Kron failed!");
		Kernals::printMatrix(X[0], 3, num_points, 9, "Kron X[0]");
		Kernals::printMatrix(X[1], 3, num_points, 9, "Kron X[1]");
		Kernals::print3DSlice(d_A, 8, 9, ransac_count - 1, 9, "Kron product");
		float *d_E_canidate;
		cudaMalloc((void **)&d_E_canidate, 3 * 3 * ransac_count * sizeof(float));
		// Calculate batch SVD
		// Last column of V becomes E
		// Calculate target E's
		
		Kernals::normalizeE << <grids, cuda_block_size >> > (d_E_canidate, ransac_count);
		// Calculate number of inliers for each E
		int *inliers = calculateInliers(d_E_canidate, ransac_count);
		// Pick best E and allocate d_E and E using thrust
		thrust::device_ptr<int> dv_in(inliers);
		auto iter = thrust::max_element(dv_in, dv_in + ransac_count);
		unsigned int best_pos = (iter - dv_in) - 1;
		// Assigne d_E
		cudaMemcpy(d_E, &(d_E_canidate[access3(0, 0, best_pos, 3, 3)]), 3 * 3 * sizeof(float), cudaMemcpyDeviceToDevice);
		// Free stuff
		cudaFree(inliers);
		cudaFree(d_A);
		cudaFree(d_indices);
		free(indices);
		cudaFree(d_E_canidate);
	}

	void Image_pair::fillXU(SiftPoint *data) {
		Kernals::printMatrix(d_K_inv, 3, 3, 3, "K inv");
		// Fill U
		dim3 grids(ceil((num_points + cuda_block_size - 1) / cuda_block_size), 1);
		dim3 blocks(cuda_block_size, 3);
		Kernals::copy_point << <grids, blocks >> > (data, num_points, U[0], U[1]);
		Kernals::printMatrix(U[0], 3, num_points, 5, "U[0]");
		// Fill X using X = inv(K) * U
		Kernals::gpu_blas_mmul(d_K_inv, U[0], X[0], 3, 3, num_points);
		Kernals::gpu_blas_mmul(d_K_inv, U[1], X[1], 3, 3, num_points);
		Kernals::printMatrix(X[0], 3, num_points, 5, "X[0]");
	}

	int * Image_pair::calculateInliers(float *d_E_canidate, int ransac_iter) {
		/// This function calculates n1, d1, n2, d2 and then finds the number of residuals per E canidate in X[0] and X[1]
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
			Kernals::gpu_blas_mmul_batched(d_E_canidate, X[0], x1_transformed, m, k, n, m*k, 0, m* n, ransac_iter); 
			//Compute n1 
			m = 3, k = num_points, n = 3; // these probably need to change because we need to transpose X[1]
			Kernals::gpu_blas_mmul_batched(X[1], d_E_canidate, n1, m, k, n, 0, 3 * 3, m * k, ransac_iter); // TODO transpose X[1]
			int blocks = ceil((3 * num_points + cuda_block_size - 1) / cuda_block_size); // BUG!!! we need to make this batched
			Kernals::element_wise_mult <<<blocks, cuda_block_size>>> (n1, X[0], 3 * num_points);
			// Compute d1
			// d1 = E1 * x1_transformed
			m = 3, k = 3, n = num_points;
			Kernals::gpu_blas_mmul_batched(d_E_canidate, x1_transformed, d1, m, k, n, m*k, 0, m* n, ransac_iter);
		// }
		// Now calculate x2_transformed, n2 and d2 {
			m = 3, k = 3, n = num_points;
			Kernals::gpu_blas_mmul_batched(d_E_canidate, X[1], x2_transformed, m, k, n, m*k, 0, m* n, ransac_iter);
			//Compute n2
			m = 3, k = num_points, n = 3; // these probably need to change because we need to transpose X[0]
			Kernals::gpu_blas_mmul_batched(X[0], d_E_canidate, n2, m, k, n, 0, 3 * 3, m * k, ransac_iter); // TODO transpose X[0]
			blocks = ceil((3 * num_points + cuda_block_size - 1) / cuda_block_size);
			Kernals::element_wise_mult << <blocks, cuda_block_size >> > (n2, X[1], 3 * num_points);
			// Compute d2
			m = 3, k = 3, n = num_points;
			Kernals::gpu_blas_mmul_batched(d_E_canidate, x2_transformed, d2, m, k, n, m*k, 0, m* n, ransac_iter);
		// }
		// Now calculate the residual per canidate E{
			float *norm_n1, *norm_n2, *norm_d1, *norm_d2;
			int *inliers;
			int size = num_points * ransac_iter;
			cudaMalloc((void**)&norm_n1, size * sizeof(float));
			cudaMalloc((void**)&norm_n2, size * sizeof(float));
			cudaMalloc((void**)&norm_d1, size * sizeof(float));
			cudaMalloc((void**)&norm_d2, size * sizeof(float));
			cudaMalloc((void**)&inliers, ransac_iter * sizeof(int));
			blocks = ceil((num_points * ransac_iter + cuda_block_size - 1) / cuda_block_size);
			Kernals::vecnorm<<<blocks, cuda_block_size>>>(n1, norm_n1, 3, size, 1, 2);
			Kernals::vecnorm<<<blocks, cuda_block_size>>>(n2, norm_n2, 3, size, 1, 2);
																		  
			Kernals::vecnorm<<<blocks, cuda_block_size>>>(d1, norm_d1, 3, size, 2, 2);
			Kernals::vecnorm<<<blocks, cuda_block_size>>>(d1, norm_d1, 3, size, 2, 2);

			Kernals::element_wise_div << <blocks, cuda_block_size >> > (norm_n1, norm_d1, size);
			Kernals::element_wise_div << <blocks, cuda_block_size >> > (norm_n2, norm_d2, size);
			// We now have the residuals in norm_n1
			Kernals::element_wise_sum << <blocks, cuda_block_size >> > (norm_n1, norm_n2, size);
			// Calculate inliers per cell
			blocks = ceil((ransac_iter + cuda_block_size - 1) / cuda_block_size);
			Kernals::threshold_count << <blocks, cuda_block_size >> > (norm_n1, inliers, num_points, ransac_iter, 1e-5);
		//}
		// Not sure if we should free
		cudaFree(n1);
		cudaFree(n2);
		cudaFree(d1);
		cudaFree(d2);
		cudaFree(x1_transformed);
		cudaFree(x2_transformed);
		// Free the norms!!!
		cudaFree(norm_n1);
		cudaFree(norm_n2);
		cudaFree(norm_d1);
		cudaFree(norm_d2);
		// 100% free
		cudaFree(d_E1);
		return inliers;
	}

	void Image_pair::computePoseCanidates() {
		// We will do all of this on the cpu because it is soo simple
		float E[9];
		cudaMemcpy(E, d_E, 3 * 3 * sizeof(float), cudaMemcpyDeviceToHost);
		float u[9], d[9], v[9];
		svd(E, u, d, v);
		float R_z[9] = { 0, -1, 0, 1, 0, 0, 0, 0, 1 };
		float canidate_T[2][3], canidate_R[2][9];
		// T1 = subset(U*R_z*d*U');
		float tmp_prod[9], tmp_prod2[9], T[9];
		for (int i = 0; i < 2; i++) {
			if (i) {// change signs for second iter
				R_z[1] = -R_z[1];
				R_z[3] = -R_z[3];
			}
			multAB(u, R_z, tmp_prod); //U * R_z 
			multAB(tmp_prod, d, tmp_prod2);// U * R_z * d
			multABt(tmp_prod2, u, T);
			canidate_T[i][0] = -T[access2(1, 2, 3)];
			canidate_T[i][1] = T[access2(0, 2, 3)];
			canidate_T[i][2] = -T[access2(0, 1, 3)];
			// R1
			multABt(u, R_z, tmp_prod);
			multABt(tmp_prod, v, canidate_R[i]);
			// Copy back to gpu
			cudaMemcpy(d_R[i], canidate_R[i], 9 * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_T[i], canidate_T[i], 3 * sizeof(float), cudaMemcpyHostToDevice);
		}
	}


	void Image_pair::testBatchedmult() {
		// C(m,n) = A(m,k) * B(k,n)
		float A[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8 };
		float B[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
		float *d_A, *d_B, *d_C;
		// Alloc
		cudaMalloc((void**)&d_A, 9 * 2 * sizeof(float));
		cudaMalloc((void**)&d_B, 9 * sizeof(float));
		cudaMalloc((void**)&d_C, 6 * 3 * sizeof(float));
		// Copy
		cudaMemcpy(d_A, A, 9 * 2 * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_B, B, 9 * sizeof(float), cudaMemcpyHostToDevice);

		int m = 3, k = 1, n = 3;
		int lda = m;
		int ldb = k;
		int ldc = m;
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
		Kernals::printVector(d_C, 6 * 3, "C");

		// Free test stuff
		cudaFree(d_A);
		cudaFree(d_B);
		cudaFree(d_C);
	}
	void Image_pair::testSVD() {
		float A[9] = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
		float u[9], d[9], v[9];
		/*cudaMalloc((void **)&d_A, 9 * sizeof(float));
		cudaMalloc((void **)&d_U, 9 * sizeof(float));
		cudaMalloc((void **)&d_D, 9 * sizeof(float));
		cudaMalloc((void **)&d_V, 9 * sizeof(float));
		cudaMemcpy(d_A, A, 9 * sizeof(float), cudaMemcpyHostToDevice);*/
		// Call svm
		/*svd(A[0], A[1], A[2], A[3], A[4], A[5], A[6], A[7], A[8],
			u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7], u[8],
			d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8],
			v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8]);*/
		svd(A, u, d, v);
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++)
				cout << d[access2(i, j, 3)] << "\t";
			cout << endl;
		}
	}
	void Image_pair::testThrust_max() {
		int a[] = { 1,2,3,4,5,6, 4, 1, 3 };
		int *d_A = Kernals::cuda_alloc_copy<int>(a, 7);
		
		thrust::device_ptr<int> dv_in(d_A);
		auto iter = thrust::max_element(dv_in, dv_in + 6);

		unsigned int position = iter - dv_in;
		int max_val = *iter;
		cudaFree(d_A);
		std::cout << "The maximum value is " << max_val << " at position " << position << std::endl;
	}
}