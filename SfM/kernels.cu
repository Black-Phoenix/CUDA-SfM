#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "kernels.h"
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

namespace kernels {
#define enable_debug true
#define esp 1e-5
#define x_pos 0
#define y_pos 1
#define z_pos 2
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
	/*		kernels			*/
	//////////////////////////////
	// C(m,n) = A(m,k) * B(k,n)
	void gpu_blas_mmul(const float *A, const float *B, float *C, const int m, const int k, const int n, cublasHandle_t handle) {
		const float alf = 1; // gpu vs cpu
		const float bet = 0;
		const float *alpha = &alf;
		const float *beta = &bet;
		// Do the actual multiplication
		cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, alpha, B, n, A, k, beta, C, n);
	}
	// C(m,n) = A(m,k) * B(k,n)
	void gpu_blas_mmul_batched(const float *A, const float *B, float *C, const int m, const int k, const int n, const int stride_A, const int stride_B, const int stride_C, const int batches, cublasHandle_t handle) {
		assert(stride_A == 0 || stride_A == m * k);
		assert(stride_B == 0 || stride_B == n * k);
		assert(stride_C == 0 || stride_C == m * n);

		const float alf = 1; // gpu vs cpu
		const float bet = 0;
		const float *alpha = &alf;
		const float *beta = &bet;
		cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, alpha, B, n, stride_B, A, k, stride_A, beta, C, n, stride_C, batches);
	}

	void gpu_blas_mmul_transpose_batched(const float *A, const float *B, float *C, const int m, const int k, const int n, const int stride_A, const int stride_B, const int stride_C, const int batches,
		cublasHandle_t handle) {
		const float alf = 1; // gpu vs cpu
		const float bet = 0;
		const float *alpha = &alf;
		const float *beta = &bet;
		cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T, n, m, k, alpha, B, n, stride_B, A, m, stride_A, beta, C, n, stride_C, batches);
	}

	void invert_device(float *src, float *dst, int n, int batchSize, cublasHandle_t handle) {
		int *P, *INFO;
		cudaMalloc<int>(&P, n * batchSize * sizeof(int));
		cudaMalloc<int>(&INFO, batchSize * sizeof(int));
		int lda = n;
		float *A[] = { src };
		float ** A_d;
		cudaMalloc<float*>(&A_d, sizeof(A));
		cudaMemcpy(A_d, A, sizeof(A), cudaMemcpyHostToDevice);
		cublasSgetrfBatched(handle, n, A_d, lda, P, INFO, batchSize);
		int INFOh = 0;
		cudaMemcpy(&INFOh, INFO, sizeof(int), cudaMemcpyDeviceToHost);
		if (INFOh == 17) {
			fprintf(stderr, "Factorization Failed: Matrix is singular\n");
			cudaDeviceReset();
			exit(EXIT_FAILURE);
		}
		float* C[] = { dst };
		float** C_d;
		cudaMalloc<float*>(&C_d, sizeof(C));
		cudaMemcpy(C_d, C, sizeof(C), cudaMemcpyHostToDevice);

		cublasSgetriBatched(handle, n, A_d, lda, P, C_d, n, INFO, batchSize);
		cudaMemcpy(&INFOh, INFO, sizeof(int), cudaMemcpyDeviceToHost);
		if (INFOh != 0)
		{
			fprintf(stderr, "Inversion Failed: Matrix is singular\n");
			cudaDeviceReset();
			exit(EXIT_FAILURE);
		}

		cudaFree(P), cudaFree(INFO);
	}

	void invert(float *s, float *d, int n, int batch, cublasHandle_t handle) {
		float *src;
		cudaMalloc<float>(&src, n * n * batch * sizeof(float));
		cudaMemcpy(src, s, n * n * batch * sizeof(float), cudaMemcpyHostToDevice);

		invert_device(src, d, n, batch, handle);
		cudaFree(src);
	}

	void svd_square(float *src, float *VT, float *S, float *U, int m, int n, const int batchSize, int *d_info, 
		cusolverDnHandle_t cusolverH, cudaStream_t stream, gesvdjInfo_t gesvdj_params) {
		assert(m == n);
		const int minmn = (m < n) ? m : n;
		const int lda = m;
		const int ldu = m;
		const int ldv = n;
		const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
		int lwork = 0;       /* size of workspace */
		float *d_work = NULL; /* device workspace for gesvdjBatched */
		cudaDeviceSynchronize();
		checkCUDAError("Could not Synchronize");
		cusolverDnSgesvdjBatched_bufferSize(cusolverH, jobz, m, n, src, lda, S, VT, ldu, U, ldv, &lwork, gesvdj_params, batchSize);
		checkCUDAError("Could not SgesvdjBatched_bufferSize");
		cudaMalloc((void**)&d_work, sizeof(float)*lwork);
		cusolverDnSgesvdjBatched(cusolverH, jobz, m, n, src, lda, S, VT, ldu, U, ldv, d_work, lwork, d_info, gesvdj_params, batchSize);
		checkCUDAError("Could not SgesvdjBatched");
		cudaDeviceSynchronize();

	}

	__global__
		void transpose(float *odata, float* idata, int width, int height)
	{
		// Naive transpose kernel because we will only be transposing 8x9 matrices
		unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
		unsigned int yIndex = blockDim.y * blockIdx.y + threadIdx.y;

		if (xIndex < width && yIndex < height)
		{
			unsigned int index_in = xIndex + width * yIndex;
			unsigned int index_out = yIndex + height * xIndex;
			odata[index_out] = idata[index_in];
		}
	}

	void svd_device_transpose(float *src, float *UT, float *S, float *VT, int m, int n, const int batchSize, int *d_info, cusolverDnHandle_t cusolverH, gesvdjInfo_t gesvdj_params) {
		float *d_A_trans = NULL;
		cudaMalloc((void **)&d_A_trans, 8 * 9 * batchSize * sizeof(float));
		for (int i = 0; i < batchSize; i++) {
			dim3 blocks(10, 10);
			dim3 fullBlocksPerGrid(1, 1);
			transpose <<< fullBlocksPerGrid, blocks >> > (d_A_trans + i * 8 * 9, src + i * 8 * 9, 9, 8);
		}
		const int minmn = (m < n) ? m : n;
		const int lda = m;
		const int ldu = m;
		const int ldv = n;
		const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
		int lwork = 0;       /* size of workspace */
		float *d_work = NULL; /* device workspace for gesvdjBatched */
		cudaDeviceSynchronize();
		checkCUDAError("Could not Synchronize");
		cusolverDnSgesvdjBatched_bufferSize(cusolverH, jobz, m, n, d_A_trans, lda, S, UT, ldu, VT, ldv, &lwork, gesvdj_params, batchSize);
		checkCUDAError("Could not SgesvdjBatched_bufferSize");
		cudaMalloc((void**)&d_work, sizeof(float)*lwork);
		cusolverDnSgesvdjBatched(cusolverH, jobz, m, n, d_A_trans, lda, S, UT, ldu, VT, ldv, d_work, lwork, d_info, gesvdj_params, batchSize);
		checkCUDAError("Could not SgesvdjBatched");
		cudaDeviceSynchronize();
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
		void vecnorm(float *A, float *res, int row, int col, int exp, int final_pow) {
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

	__global__
		void canidate_kernels(float *d_P, const float *u, const float *v) {
		int index = blockIdx.x*blockDim.x + threadIdx.x;
		if (index >= 4) // only 4 canidate positions exist so fixed value
			return;
		float W[9] = { 0, -1, 0, 1, 0, 0, 0, 0, 1 }; // rotation about z axis
		float Wt[9]; transpose_copy3x3(W, Wt, 3, 3);
		float canidate_P[4 * 4];

		float tmp_prod[9], tmp_prod2[9], T[9];
		// T
		canidate_P[access2(x_pos, 3, 4)] = ((!index || index == 2) ? -1 : 1) * u[access2(x_pos, 2, 3)];
		canidate_P[access2(y_pos, 3, 4)] = ((!index || index == 2) ? -1 : 1) * u[access2(y_pos, 2, 3)];
		canidate_P[access2(z_pos, 3, 4)] = ((!index || index == 2) ? -1 : 1) * u[access2(z_pos, 2, 3)];
		// R
		if (index < 2)
			multABt(W, v, tmp_prod);
		else
			multABt(Wt, v, tmp_prod);
		multAB(u, tmp_prod, tmp_prod2); // 3x3 transpose
		transpose_copy3x3(tmp_prod2, canidate_P, 3, 4);
		// Now we copy from 2d to 3d into d_P
		//d_P[index] = index;
		memcpy(&(d_P[access3(0, 0, index, 4, 4)]), canidate_P, 4 * 4 * sizeof(float));
		d_P[access3(3, 0, index, 4, 4)] = 0; // Set last row maually
		d_P[access3(3, 1, index, 4, 4)] = 0;
		d_P[access3(3, 2, index, 4, 4)] = 0;
		d_P[access3(3, 3, index, 4, 4)] = 1;
	}

	__global__
		void compute_linear_triangulation_A(float *A, const float *pt1, const float *pt2, const int count, const int num_points, const float *m1, const float *m2, int P_ind, bool canidate_m2) {
		// if canidate_m2,  we are computing 4 A's for different m2
		// Points are 3xN and Projection matrices are 4x4
		int index = blockIdx.x*blockDim.x + threadIdx.x;
		int row = blockIdx.y*blockDim.y + threadIdx.y; // 2 rows, x, y
		if (index >= count || row >= 2)
			return;
		float tmp_A[2 * 4], valx, valy;
		const float *correct_pt, *correct_m;
		if (canidate_m2) {
			assert(count == 4);

			if (!row) { // Slightly help with the warp divergence here
				correct_pt = pt1;
				correct_m = m1;
			}
			else {
				correct_pt = pt2;
				correct_m = &(m2[access3(0, 0, index, 4, 4)]);
			}
			valx = correct_pt[access2(x_pos, 0, num_points)]; // we only use the first point
			valy = correct_pt[access2(y_pos, 0, num_points)];
		}
		else {
			assert(P_ind < 4 && P_ind >= 0);
			if (!row) { // Slightly help with the warp divergence here
				correct_pt = pt1;
				correct_m = m1;
			}
			else {
				correct_pt = pt2;
				correct_m = &(m2[access3(0, 0, P_ind, 4, 4)]);;
			}
			valx = correct_pt[access2(x_pos, index, num_points)];
			valy = correct_pt[access2(y_pos, index, num_points)]; // Num points does not need to be the same as count
		}

#pragma unroll
		for (int i = 0; i < 4; i++) {
			tmp_A[access2(x_pos, i, 4)] = valx * correct_m[access2(2, i, 4)] - correct_m[access2(x_pos, i, 4)];
			tmp_A[access2(y_pos, i, 4)] = valy * correct_m[access2(2, i, 4)] - correct_m[access2(y_pos, i, 4)];
		}
		memcpy(&(A[access3(((!row) ? 0 : 2), 0, index, 4, 4)]), tmp_A, 4 * 2 * sizeof(float));
	}

	__global__
		void normalize_pt_kernal(float *v, float *converted_pt, int number_points) { // assumes size of converted_pt is 4xnum_points and v is 4x4xnum_points
		int index = blockIdx.x*blockDim.x + threadIdx.x; // one per num_points
		if (index >= number_points)
			return;
		float norm_value = v[access3(3, 3, index, 4, 4)];
		converted_pt[access2(x_pos, index, number_points)] = v[access3(3, x_pos, index, 4, 4)] / norm_value;
		converted_pt[access2(y_pos, index, number_points)] = v[access3(3, y_pos, index, 4, 4)] / norm_value;
		converted_pt[access2(z_pos, index, number_points)] = v[access3(3, z_pos, index, 4, 4)] / norm_value;
		converted_pt[access2(3, index, number_points)] = 1;
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
		kernels::kron_kernal << <grids, cuda_block_size >> > (X[0], X[1], d_A, d_indices, ransac_count, num_points);
		checkCUDAErrorWithLine("Kron failed!");
		kernels::printMatrix(X[0], 3, num_points, 9, "Kron X[0]");
		kernels::printMatrix(X[1], 3, num_points, 9, "Kron X[1]");
		kernels::print3DSlice(d_A, 8, 9, ransac_count - 1, 9, "Kron product");
		float *d_E_canidate;
		cudaMalloc((void **)&d_E_canidate, 3 * 3 * ransac_count * sizeof(float));
		// Calculate batch SVD

		// Last column of V becomes E

		// Calculate target E's
		kernels::normalizeE << <grids, cuda_block_size >> > (d_E_canidate, ransac_count);
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
		kernels::gpu_blas_mmul_batched(d_E_canidate, X[0], x1_transformed, m, k, n, m * k, 0, m * n, ransac_iter, handle);
		//Compute n1 
		m = num_points, k = 3, n = 3; // these probably need to change because we need to transpose X[1]
		kernels::gpu_blas_mmul_transpose_batched(X[1], d_E_canidate, n1, m, k, n, 0, 3 * 3, m * n, ransac_iter, handle); // transpose X[1]
		int blocks = ceil((3 * num_points + cuda_block_size - 1) / cuda_block_size); // BUG!!! we need to make this batched
		kernels::element_wise_mult << <blocks, cuda_block_size >> > (n1, X[0], 3 * num_points);
		// Compute d1
		// d1 = E1 * x1_transformed
		m = 3, k = 3, n = num_points;
		kernels::gpu_blas_mmul_batched(d_E_canidate, x1_transformed, d1, m, k, n, m*k, 0, m* n, ransac_iter, handle);
		// }
		// Now calculate x2_transformed, n2 and d2 {
		m = 3, k = 3, n = num_points;
		kernels::gpu_blas_mmul_batched(d_E_canidate, X[1], x2_transformed, m, k, n, m*k, 0, m* n, ransac_iter, handle);
		//Compute n2
		m = num_points, k = 3, n = 3; // these probably need to change because we need to transpose X[0]
		kernels::gpu_blas_mmul_transpose_batched(X[0], d_E_canidate, n2, m, k, n, 0, 3 * 3, m * n, ransac_iter, handle); // transpose X[0]
		blocks = ceil((3 * num_points + cuda_block_size - 1) / cuda_block_size);
		kernels::element_wise_mult << <blocks, cuda_block_size >> > (n2, X[1], 3 * num_points);
		// Compute d2
		m = 3, k = 3, n = num_points;
		kernels::gpu_blas_mmul_batched(d_E_canidate, x2_transformed, d2, m, k, n, m*k, 0, m* n, ransac_iter, handle);
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
		kernels::threshold_count << <blocks, cuda_block_size >> > (norm_n1, inliers, num_points, ransac_iter, 1e-5);
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
		// Tested
		float E[9] = { 1,2,3,4,5,6,7,8,9 }; // TODO remove this once testing is done
		//cudaMemcpy(E, d_E, 3 * 3 * sizeof(float), cudaMemcpyDeviceToHost);
		float u[9], d[9], v[9], tmp[9];
		svd(E, u, d, v); // v is not transposed
		multABt(u, v, tmp); // u * v'
		if (det(tmp) < 0)
			neg(v);
		float *d_u, *d_v;
		d_u = kernels::cuda_alloc_copy(u, 3 * 3);
		d_v = kernels::cuda_alloc_copy(v, 3 * 3);
		kernels::canidate_kernels << <1, 32 >> > (d_P, d_u, d_v);
		cudaFree(d_u);
		cudaFree(d_v);
	}

	void Image_pair::choosePose() {
		////Debugging{
		//	float *d_P_debugging;
		//	float x[] = { 1,2,1,2,
		//				2,1,2,1,
		//				1,1,2,2,
		//				1,2,2,1,
		//				
		//				1,1,2,2,
		//				1,2,1,2,
		//				2,1,2,1,
		//				1,2,2,1,
		//		
		//				1,2,1,2,
		//				1,1,2,2,
		//				2,1,2,1,
		//				1,2,2,1,
		//				
		//				1,2,2,1,
		//				1,2,1,2,
		//				2,1,2,1,
		//				1,1,2,2};
		//	d_P_debugging = kernels::cuda_alloc_copy(x, 4 * 4 * 4);
		////}
		// take 1 point and figure out if it is in front of the camera or behind
		float P1[16] = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 }; // I(4)
		float *d_P1 = kernels::cuda_alloc_copy(P1, 16);
		float *d_A, *d_u, *d_d, *d_vt;
		cudaMalloc((void **)&d_A, 4 * 4 * 4 * sizeof(float));
		cudaMalloc((void **)&d_u, 4 * 4 * 4 * sizeof(float));
		cudaMalloc((void **)&d_d, 4 * 4 * 4 * sizeof(float));
		cudaMalloc((void **)&d_vt, 4 * 4 * 4 * sizeof(float));
		// Create A

		dim3 blocks(1, 1);
		dim3 block_sizes(4, 2);
		kernels::compute_linear_triangulation_A << <blocks, block_sizes >> > (d_A, X[0], X[1], 4, num_points, d_P1, d_P_debugging, -1, true); // todo change d_P_debugging back to d_P
		
		// We only care about V
		float *d_d1, *d_d2; // 3x4 batched
		cudaMalloc((void **)&d_d1, 4 * 4 * sizeof(float));
		cudaMalloc((void **)&d_d2, 4 * 4 * sizeof(float));
		// Assumes V isnt transposed, we need to take the last row
		// svd(d_A, d_u, d_d, d_v, 4 batches)
		int *d_info = NULL;
		cudaMalloc((void**)&d_info, 4 * sizeof(int));
		kernels::svd_square(d_A, d_vt, d_d, d_u, 4, 4, 4, d_info, cusolverH, stream, gesvdj_params);
		kernels::normalize_pt_kernal <<<1, 4 >> > (d_vt, d_d1, 4);
		kernels::printMatrix(d_d1, 4, 4, 4, "d1");

		float *d_P2_inv;
		float val_d1, val_d2;
		cudaMalloc((void **)&d_P2_inv, 4 * 4 * sizeof(float));
		for (int i = 0; i < 4; i++) { // batched doesn't work for inverse + it is only 4, 4x4 matrices, should be easy
			kernels::invert(d_P_debugging + i * 4 * 4, d_P2_inv, 4, 1, handle); // todo change back d_P_debugging
			int m = 4, k = 4, n = 4;
			kernels::gpu_blas_mmul(d_P2_inv, d_d1, d_d2, m, k, n, handle);
			kernels::print3DSlice(d_P2_inv, 4, 4, 0, 4, "d_P2_inv");
			kernels::printMatrix(d_d2, 4, 4, 4, "d2");
			// Do the final testing on the host
			cudaMemcpy(&val_d1, &(d_d1[access2(2, i, 4)]), sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(&val_d2, &(d_d2[access2(2, i, 4)]), sizeof(float), cudaMemcpyDeviceToHost);
			// Now we do the final check on the cpu as well, because it is the same ease
			if (val_d1 > 0 && val_d2 > 0)
				P_ind = i;
		}
		
		cudaFree(d_P2_inv);
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
		cudaMalloc((void **)&d_d, 4 * 4 * num_points * sizeof(float));
		cudaMalloc((void **)&d_vt, 4 * 4 * num_points * sizeof(float));
		// Create A

		dim3 grids(ceil((num_points * 2 + cuda_block_size - 1) / cuda_block_size), 1);
		dim3 block_sizes(cuda_block_size/2, 2);
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
		kernels::normalize_pt_kernal <<<grids2, block_sizes2 >> > (d_vt, d_final_points, num_points);  
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
}