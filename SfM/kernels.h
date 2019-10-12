#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include <assert.h>
#include <cublas_v2.h>
#include <curand.h>
#include <memory>
#include <string>
#include <iostream>
#include <cublas_v2.h>
#include <CudaSift/cudaSift.h>
#include <curand.h>
#include <iomanip>      
#include <algorithm>
#include <cusolverDn.h>
#include "svd.h"

namespace kernels {
#define enable_debug false
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
					cout << std::setw(5) << setprecision(3) << "\t" << Areg;
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

	void regular_svd(float *src, float *UT, float *S, float *VT, int m, int n, const int batchSize, int *d_info, cusolverDnHandle_t cusolverH, gesvdjInfo_t gesvdj_params) {
		float *d_A_trans = NULL;
		cudaMalloc((void **)&d_A_trans, 8 * 9 * batchSize * sizeof(float));
		for (int i = 0; i < batchSize; i++) {
			dim3 blocks(10, 10);
			dim3 fullBlocksPerGrid(1, 1);
			transpose << < fullBlocksPerGrid, blocks >> > (d_A_trans + i * 8 * 9, src + i * 8 * 9, 9, 8);
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
		void kernels(float*d1, float*d2, float *A, int *indices, const int ransac_iterations, int num_points) {
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
		void vecnorm(float *A, float *res, int row, int col, float exp, float final_pow) {
		int index = blockIdx.x*blockDim.x + threadIdx.x;
		if (index >= col)
			return;
		float tmp_vlaue = 0;
#pragma unroll
		for (int i = 0; i < row; i++) {
			tmp_vlaue += powf(A[access2(i, index, col)], exp);
		}
		// Now we can take the sqrt of exp and then rais to the final_pow
		if (exp == final_pow) {
			res[index] = tmp_vlaue;
			return;
		}
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
		void candidate_kernels(float *d_P, const float *u, const float *v) {
		int index = blockIdx.x*blockDim.x + threadIdx.x;
		if (index >= 4) // only 4 candidate positions exist so fixed value
			return;
		float W[9] = { 0, -1, 0, 1, 0, 0, 0, 0, 1 }; // rotation about z axis
		float Wt[9]; transpose_copy3x3(W, Wt, 3, 3);
		float candidate_P[4 * 4];

		float tmp_prod[9], tmp_prod2[9], T[9];
		// T
		candidate_P[access2(x_pos, 3, 4)] = ((!index || index == 2) ? -1 : 1) * u[access2(x_pos, 2, 3)];
		candidate_P[access2(y_pos, 3, 4)] = ((!index || index == 2) ? -1 : 1) * u[access2(y_pos, 2, 3)];
		candidate_P[access2(z_pos, 3, 4)] = ((!index || index == 2) ? -1 : 1) * u[access2(z_pos, 2, 3)];
		// R
		if (index < 2)
			multABt(W, v, tmp_prod);
		else
			multABt(Wt, v, tmp_prod);
		multAB(u, tmp_prod, tmp_prod2); // 3x3 transpose
		transpose_copy3x3(tmp_prod2, candidate_P, 3, 4);
		// Now we copy from 2d to 3d into d_P
		//d_P[index] = index;
		memcpy(&(d_P[access3(0, 0, index, 4, 4)]), candidate_P, 4 * 4 * sizeof(float));
		d_P[access3(3, 0, index, 4, 4)] = 0; // Set last row maually
		d_P[access3(3, 1, index, 4, 4)] = 0;
		d_P[access3(3, 2, index, 4, 4)] = 0;
		d_P[access3(3, 3, index, 4, 4)] = 1;
	}

	__global__
		void compute_linear_triangulation_A(float *A, const float *pt1, const float *pt2, const int count, const int num_points, const float *m1, const float *m2, int P_ind, bool candidate_m2) {
		// if candidate_m2,  we are computing 4 A's for different m2
		// Points are 3xN and Projection matrices are 4x4
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		int row = blockIdx.y * blockDim.y + threadIdx.y; // 2 rows, x, y
		if (index >= count || row >= 2)
			return;
		float tmp_A[2 * 4], valx, valy;
		const float *correct_pt, *correct_m;
		if (candidate_m2) {
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
		if (norm_value == 0 || abs(norm_value) > 5) {
			converted_pt[access2(x_pos, index, number_points)] = 0;
			converted_pt[access2(y_pos, index, number_points)] = 0;
			converted_pt[access2(z_pos, index, number_points)] = 0;
		}
		else {
			converted_pt[access2(x_pos, index, number_points)] = v[access3(3, x_pos, index, 4, 4)] / norm_value;
			converted_pt[access2(y_pos, index, number_points)] = v[access3(3, y_pos, index, 4, 4)] / norm_value;
			converted_pt[access2(z_pos, index, number_points)] = v[access3(3, z_pos, index, 4, 4)] / norm_value;
		}
		converted_pt[access2(3, index, number_points)] = 1;
	}

	__global__
		void row_extraction_kernel(float *d_vt, float *d_E, int number_points) { // assumes size of converted_pt is 4xnum_points and v is 4x4xnum_points
		int index = blockIdx.x*blockDim.x + threadIdx.x; // one per num_points
		if (index >= number_points)
			return;
		memcpy(d_E + 9 * index, d_vt + 9 * 9 * index + 9 * 8, 3 * 3 * sizeof(float)); // the final 9 * 8 is because we want the last row
	}

	template<typename T>
	T* cuda_alloc_copy(const T* host, int size) {
		T* data;
		cudaMalloc((void**)&data, size * sizeof(T));
		cudaMemcpy(data, host, size * sizeof(T), cudaMemcpyHostToDevice);
		return data;
	}

	/////////////////////////////////////////////////////////////////////////
	////////////////////viz kernels///////////////////
	////////////////////////////////////////////////////////////////////////
	__global__
		void kernCopyPositionsToVBO(int N, float *pos, float *vbo, float s_scale) {
		int index = threadIdx.x + (blockIdx.x * blockDim.x);

		float c_scale = s_scale;

		if (index < N) {
			vbo[access2(index, x_pos, 4)] = pos[access2(x_pos, index, N)] * c_scale;
			vbo[access2(index, y_pos, 4)] = pos[access2(y_pos, index, N)] * c_scale;
			vbo[access2(index, z_pos, 4)] = pos[access2(z_pos, index, N)] * c_scale;
			vbo[access2(index, 3, 4)] = 1.0f;
		}
	}

	__global__
		void kernCopyVelocitiesToVBO(int N, float *vbo, float s_scale) {
		int index = threadIdx.x + (blockIdx.x * blockDim.x);

		if (index < N) {
			vbo[4 * index + 0] = 1;//vel[index].x + 0.3f;
			vbo[4 * index + 1] = 1;//vel[index].y + 0.3f;
			vbo[4 * index + 2] = 1;//vel[index].z + 0.3f;
			vbo[4 * index + 3] = 1.0f;
		}
	}
}
