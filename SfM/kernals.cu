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

namespace Kernals {
#define enable_debug false
#define esp 1e-5
	using Common::PerformanceTimer;
	PerformanceTimer& timer()
	{
		static PerformanceTimer timer;
		return timer;
	}
	//////////////////////////////
	/*		Kernals			*/
	//////////////////////////////
	void printCuda(double *a1, int n, string name) {
		if (!enable_debug)
			return;
		double *print_a = new double[n];
		cout << name.c_str() << endl;
		cout << "{" << endl;
		cudaMemcpy(print_a, a1, n * sizeof(double), cudaMemcpyDeviceToHost);
		for (int i = 0; i < n; i++) {
			cout << "\t" << print_a[i] << endl;
		}
		cout << "}" << endl;
		delete[]print_a;
	}
	// C(m,n) = A(m,k) * B(k,n)
	// lda = k (if transposed)
	// ldb = n (if we transpose)
	// ldb = n (if we transpose)
	__device__ __inline__
	void gpu_blas_mmul(const double *A, const double *B, double *C, const int m, const int k, const int n, bool trans_flag_a, bool trans_flag_b) {
		int lda, ldb, ldc;
		lda = (!trans_flag_a) ? m : k;
		ldb = (!trans_flag_b) ? k : n;
		ldc = m;
		const double alf = 1; // gpu vs cpu
		const double bet = 0;
		const double *alpha = &alf;
		const double *beta = &bet;
		// Do the actual multiplication
		cublasHandle_t handle;
		cublasCreate(&handle);
		cublasDgemm(handle, (cublasOperation_t)trans_flag_a, (cublasOperation_t)trans_flag_b, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
	}
}
