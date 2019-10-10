/**************************************************************************
**
**  svd3
**
** Quick singular value decomposition as described by:
** A. McAdams, A. Selle, R. Tamstorf, J. Teran and E. Sifakis,
** Computing the Singular Value Decomposition of 3x3 matrices
** with minimal branching and elementary floating point operations,
**  University of Wisconsin - Madison technical report TR1690, May 2011
** https://github.com/ericjang/svd3/blob/master/svd3_cuda/svd3.cu
**  OPTIMIZED GPU VERSION
**  Implementation by: Eric Jang
**
**  13 Apr 2014
**  Modified by Vaibhav Arcot, 2019
**************************************************************************/

#ifndef SVD3_CUDA_H
#define SVD3_CUDA_H

#define _gamma 5.828427124746190 // FOUR_GAMMA_SQUARED = sqrt(8)+3;
#define _cstar 0.923879532511287 // cos(pi/8)
#define _sstar 0.382683432365090 // sin(p/8)
#define EPSILON 1e-6

#include <cuda.h>
#include "math.h" // CUDA math library
#include "common.h"


// CUDA's 1.0/sqrtf seems to be faster than the inlined approximation?

__host__ __device__ __forceinline__
float accurateSqrt(float x)
{
	return x * 1.0 / sqrtf(x);
}

__host__ __device__ __forceinline__
void condSwap(bool c, float &X, float &Y)
{
	// used in step 2
	float Z = X;
	X = c ? Y : X;
	Y = c ? Z : Y;
}

__host__ __device__ __forceinline__
void condNegSwap(bool c, float &X, float &Y)
{
	// used in step 2 and 3
	float Z = -X;
	X = c ? Y : X;
	Y = c ? Z : Y;
}

// matrix multiplication M = A * B
__host__ __device__ __forceinline__
void multAB(const float *a,
			const float *b,float *m)
{
	m[0] = a[0] * b[0] + a[1] * b[3] + a[2] * b[6]; m[1] = a[0] * b[1] + a[1] * b[4] + a[2] * b[7]; m[2] = a[0] * b[2] + a[1] * b[5] + a[2] * b[8];
	m[3] = a[3] * b[0] + a[4] * b[3] + a[5] * b[6]; m[4] = a[3] * b[1] + a[4] * b[4] + a[5] * b[7]; m[5] = a[3] * b[2] + a[4] * b[5] + a[5] * b[8];
	m[6] = a[6] * b[0] + a[7] * b[3] + a[8] * b[6]; m[7] = a[6] * b[1] + a[7] * b[4] + a[8] * b[7]; m[8] = a[6] * b[2] + a[7] * b[5] + a[8] * b[8];
}

// matrix multiplication M = Transpose[A] * B
__host__ __device__ __forceinline__
void multAtB(const float *a, const float *b, float *m)
{
	m[0] = a[0] * b[0] + a[3] * b[3] + a[6] * b[6]; m[1] = a[0] * b[1] + a[3] * b[4] + a[6] * b[7]; m[2] = a[0] * b[2] + a[3] * b[5] + a[6] * b[8];
	m[3] = a[1] * b[0] + a[4] * b[3] + a[7] * b[6]; m[4] = a[1] * b[1] + a[4] * b[4] + a[7] * b[7]; m[5] = a[1] * b[2] + a[4] * b[5] + a[7] * b[8];
	m[6] = a[2] * b[0] + a[5] * b[3] + a[8] * b[6]; m[7] = a[2] * b[1] + a[5] * b[4] + a[8] * b[7]; m[8] = a[2] * b[2] + a[5] * b[5] + a[8] * b[8];
}

// matrix multiplication M = A * Transpose[B]
__host__ __device__ __forceinline__
void multABt(const float *a, const float *b, float *m)
{
	m[0] = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; m[1] = a[0] * b[3] + a[1] * b[4] + a[2] * b[5]; m[2] = a[0] * b[6] + a[1] * b[7] + a[2] * b[8];
	m[3] = a[3] * b[0] + a[4] * b[1] + a[5] * b[2]; m[4] = a[3] * b[3] + a[4] * b[4] + a[5] * b[5]; m[5] = a[3] * b[6] + a[4] * b[7] + a[5] * b[8];
	m[6] = a[6] * b[0] + a[7] * b[1] + a[8] * b[2]; m[7] = a[6] * b[3] + a[7] * b[4] + a[8] * b[5]; m[8] = a[6] * b[6] + a[7] * b[7] + a[8] * b[8];
}

__host__ __device__ __forceinline__
void neg(float *a) {
	a[0] = -a[0];
	a[1] = -a[1];
	a[2] = -a[2];
	a[3] = -a[3];
	a[4] = -a[4];
	a[5] = -a[5];
	a[6] = -a[6];
	a[7] = -a[7];
	a[8] = -a[8];
}
__host__ __device__ __forceinline__
void quatToMat3(const float * qV, float *m)
{
	float w = qV[3];
	float x = qV[0];
	float y = qV[1];
	float z = qV[2];

	float qxx = x * x;
	float qyy = y * y;
	float qzz = z * z;
	float qxz = x * z;
	float qxy = x * y;
	float qyz = y * z;
	float qwx = w * x;
	float qwy = w * y;
	float qwz = w * z;

	m[0] = 1 - 2 * (qyy + qzz); m[1] = 2 * (qxy - qwz); m[2] = 2 * (qxz + qwy);
	m[3] = 2 * (qxy + qwz); m[4] = 1 - 2 * (qxx + qzz); m[5] = 2 * (qyz - qwx);
	m[6] = 2 * (qxz - qwy); m[7] = 2 * (qyz + qwx); m[8] = 1 - 2 * (qxx + qyy);
}

__host__ __device__ __forceinline__
void approximateGivensQuaternion(float a11, float a12, float a22, float &ch, float &sh)
{
	/*
		 * Given givens angle computed by approximateGivensAngles,
		 * compute the corresponding rotation quaternion.
		 */
    ch = 2*(a11-a22);
    sh = a12;
    bool b = _gamma*sh*sh < ch*ch;
    float w = 1.0/sqrtf(ch*ch+sh*sh);
    ch=b?w*ch:_cstar;
    sh=b?w*sh:_sstar;
}

__host__ __device__ __forceinline__
void jacobiConjugation(const int x, const int y, const int z,
	float *s, float * qV)
{
	float ch, sh;
	approximateGivensQuaternion(s[0], s[3], s[4], ch, sh);

	float scale = ch * ch + sh * sh;
	float a = (ch*ch - sh * sh) / scale;
	float b = (2 * sh*ch) / scale;

	// make temp copy of S
	float _s[9];
	_s[0] = s[0];
	_s[3] = s[3];
	_s[4] = s[4];
	_s[6] = s[6];
	_s[7] = s[7];
	_s[8] = s[8];
	// perform conjugation S = Q'*S*Q
	// Q already implicitly solved from a, b
	s[0] = a * (a*_s[0] + b * _s[3]) + b * (a*_s[3] + b * _s[4]);
	s[3] = a * (-b * _s[0] + a * _s[3]) + b * (-b * _s[3] + a * _s[4]);   s[4] = -b * (-b * _s[0] + a * _s[3]) + a * (-b * _s[3] + a * _s[4]);
	s[6] = a * _s[6] + b * _s[7];                               s[7] = -b * _s[6] + a * _s[7]; s[8] = _s[8];

	// update cumulative rotation qV
	float tmp[3];
	tmp[0] = qV[0] * sh;
	tmp[1] = qV[1] * sh;
	tmp[2] = qV[2] * sh;
	sh *= qV[3];

	qV[0] *= ch;
	qV[1] *= ch;
	qV[2] *= ch;
	qV[3] *= ch;

	// (x,y,z) corresponds to ((0,1,2),(1,2,0),(2,0,1))
	// for (p,q) = ((0,1),(1,2),(0,2))
	qV[z] += sh;
	qV[3] -= tmp[z]; // w
	qV[x] += tmp[y];
	qV[y] -= tmp[x];

	// re-arrange matrix for next iteration
	_s[0] = s[4];
	_s[3] = s[7]; _s[4] = s[8];
	_s[6] = s[3]; _s[7] = s[6]; _s[8] = s[0];
	s[0] = _s[0];
	s[3] = _s[3]; s[4] = _s[4];
	s[6] = _s[6]; s[7] = _s[7]; s[8] = _s[8];

}

__host__ __device__ __forceinline__
float dist2(float x, float y, float z)
{
	return x * x + y * y + z * z;
}

// finds transformation that diagonalizes a symmetric matrix
__host__ __device__ __forceinline__
void jacobiEigenanlysis( // symmetric matrix
	float *s,	// quaternion representation of V
	float * qV)
{
	qV[3] = 1; qV[0] = 0; qV[1] = 0; qV[2] = 0; // follow same indexing convention as GLM
	for (int i = 0; i < 4; i++)
	{
		// we wish to eliminate the maximum off-diagonal element
		// on every iteration, but cycling over all 3 possible rotations
		// in fixed order (p,q) = (1,2) , (2,3), (1,3) still retains
		//  asymptotic convergence
		jacobiConjugation(0, 1, 2, s, qV); // p,q = 0,1
		jacobiConjugation(1, 2, 0, s, qV); // p,q = 1,2
		jacobiConjugation(2, 0, 1, s, qV); // p,q = 0,2
	}
}

__host__ __device__ __forceinline__
void sortSingularValues(// matrix that we want to decompose
	float *b, float *v)
{
	float rho1 = dist2(b[0], b[3], b[6]);
	float rho2 = dist2(b[1], b[4], b[7]);
	float rho3 = dist2(b[2], b[5], b[8]);
	bool c;
	c = rho1 < rho2;
	condNegSwap(c, b[0], b[1]); condNegSwap(c, v[0], v[1]);
	condNegSwap(c, b[3], b[4]); condNegSwap(c, v[3], v[4]);
	condNegSwap(c, b[6], b[7]); condNegSwap(c, v[6], v[7]);
	condSwap(c, rho1, rho2);
	c = rho1 < rho3;
	condNegSwap(c, b[0], b[2]); condNegSwap(c, v[0], v[2]);
	condNegSwap(c, b[3], b[5]); condNegSwap(c, v[3], v[5]);
	condNegSwap(c, b[6], b[8]); condNegSwap(c, v[6], v[8]);
	condSwap(c, rho1, rho3);
	c = rho2 < rho3;
	condNegSwap(c, b[1], b[2]); condNegSwap(c, v[1], v[2]);
	condNegSwap(c, b[4], b[5]); condNegSwap(c, v[4], v[5]);
	condNegSwap(c, b[7], b[8]); condNegSwap(c, v[7], v[8]);
}

__host__ __device__ __forceinline__
void QRGivensQuaternion(float a1, float a2, float &ch, float &sh)
{
	// a1 = pivot point on diagonal
	// a2 = lower triangular entry we want to annihilate
	float epsilon = EPSILON;
	float rho = accurateSqrt(a1*a1 + a2 * a2);

	sh = rho > epsilon ? a2 : 0;
	ch = fabs(a1) + fmax(rho, epsilon);
	bool b = a1 < 0;
	condSwap(b, sh, ch);
	float w = 1.0 / sqrtf(ch*ch + sh * sh);
	ch *= w;
	sh *= w;
}

__host__ __device__ __forceinline__
void QRDecomposition(// matrix that we want to decompose
	const float *b, float *q, float *r)
{
	float ch1, sh1, ch2, sh2, ch3, sh3;
	float a, i_b;

	// first givens rotation (ch,0,0,sh)
	QRGivensQuaternion(b[0], b[3], ch1, sh1);
	a = 1 - 2 * sh1*sh1;
	i_b = 2 * ch1*sh1;
	// apply B = Q' * B
	r[0] = a * b[0] + i_b * b[3];  r[1] = a * b[1] + i_b * b[4];  r[2] = a * b[2] + i_b * b[5];
	r[3] = -i_b * b[0] + a * b[3]; r[4] = -i_b * b[1] + a * b[4]; r[5] = -i_b * b[2] + a * b[5];
	r[6] = b[6];          r[7] = b[7];          r[8] = b[8];

	// second givens rotation (ch,0,-sh,0)
	QRGivensQuaternion(r[0], r[6], ch2, sh2);
	a = 1 - 2 * sh2*sh2;
	i_b = 2 * ch2*sh2;
	float X[9];
	// apply X = Q' * B;
	X[0] = a * r[0] + i_b * r[6];  X[1] = a * r[1] + i_b * r[7];  X[2] = a * r[2] + i_b * r[8];
	X[3] = r[3];				   X[4] = r[4];					  X[5] = r[5];
	X[6] = -i_b * r[0] + a * r[6]; X[7] = -i_b * r[1] + a * r[7]; X[8] = -i_b * r[2] + a * r[8];

	// third givens rotation (ch,sh,0,0)
	QRGivensQuaternion(X[4], X[7], ch3, sh3);
	a = 1 - 2 * sh3*sh3;
	i_b = 2 * ch3*sh3;
	// R is now set to desired value
	r[0] = X[0];             r[1] = X[1];           r[2] = X[2];
	r[3] = a * X[3] + i_b * X[6];     r[4] = a * X[4] + i_b * X[7];   r[5] = a * X[5] + i_b * X[8];
	r[6] = -i_b * X[3] + a * X[6];    r[7] = -i_b * X[4] + a * X[7];  r[8] = -i_b * X[5] + a * X[8];

	// construct the cumulative rotation Q=Q1 * Q2 * Q3
	// the number of floating point operations for three quaternion multiplications
	// is more or less comparable to the explicit form of the joined matrix.
	// certainly more memory-efficient!
	float sh11 = sh1 * sh1;
	float sh22 = sh2 * sh2;
	float sh33 = sh3 * sh3;

	q[0] = (-1 + 2 * sh11)*(-1 + 2 * sh22);
	q[1] = 4 * ch2*ch3*(-1 + 2 * sh11)*sh2*sh3 + 2 * ch1*sh1*(-1 + 2 * sh33);
	q[2] = 4 * ch1*ch3*sh1*sh3 - 2 * ch2*(-1 + 2 * sh11)*sh2*(-1 + 2 * sh33);

	q[3] = 2 * ch1*sh1*(1 - 2 * sh22);
	q[4] = -8 * ch1*ch2*ch3*sh1*sh2*sh3 + (-1 + 2 * sh11)*(-1 + 2 * sh33);
	q[5] = -2 * ch3*sh3 + 4 * sh1*(ch3*sh1*sh3 + ch1 * ch2*sh2*(-1 + 2 * sh33));

	q[6] = 2 * ch2*sh2;
	q[7] = 2 * ch3*(1 - 2 * sh22)*sh3;
	q[8] = (-1 + 2 * sh22)*(-1 + 2 * sh33);
}

__host__ __device__ __forceinline__
void svd(//output A
	const float *a,
	// output U
	float *u, float *s, float *v)
{
	// normal equations matrix
	float ATa[9];

	multAtB(a, a, ATa);

	// symmetric eigenalysis
	float qV[4];
	jacobiEigenanlysis(ATa, qV);
	quatToMat3(qV, v);

	float b[9];
	multAB(a, v, b);

	// sort singular values and find V
	sortSingularValues(b, v);

	// QR decomposition
	QRDecomposition(b, u, s);
}

__host__ __device__ __forceinline__
float det(const float *a) {
	return a[0] * a[4]*a[8] - a[0] * a[5]*a[7] - a[0] * a[3]*a[8] + a[1] * a[5] *a[6] + a[2] * a[3] *a[7] - a[2] * a[4]*a[6];

}

__host__ __device__ __forceinline__
void transpose_copy3x3(const float *a, float *b, int a_size, int b_size) { // transpose first 3x3 elements and put it in first 3x3 elements of b (regular transpose if a_size = b_size = 3)
	b[access2(0, 0, b_size)] = a[access2(0, 0, a_size)]; b[access2(1, 1, b_size)] = a[access2(1, 1, a_size)]; b[access2(2, 2, b_size)] = a[access2(2, 2, a_size)];
	b[access2(1, 0, b_size)] = a[access2(0, 1, a_size)]; b[access2(0, 1, b_size)] = a[access2(1, 0, a_size)];
	b[access2(2, 0, b_size)] = a[access2(0, 2, a_size)]; b[access2(0, 2, b_size)] = a[access2(2, 0, a_size)]; // 2
	b[access2(1, 2, b_size)] = a[access2(2, 1, a_size)]; b[access2(2, 1, b_size)] = a[access2(1, 2, a_size)]; // 5
}

__host__ __device__ __forceinline__
bool InvertMatrix4x4(const float m[16], float invOut[16])
{
	float inv[16], det;
	int i;

	inv[0] = m[5] * m[10] * m[15] -
		m[5] * m[11] * m[14] -
		m[9] * m[6] * m[15] +
		m[9] * m[7] * m[14] +
		m[13] * m[6] * m[11] -
		m[13] * m[7] * m[10];

	inv[4] = -m[4] * m[10] * m[15] +
		m[4] * m[11] * m[14] +
		m[8] * m[6] * m[15] -
		m[8] * m[7] * m[14] -
		m[12] * m[6] * m[11] +
		m[12] * m[7] * m[10];

	inv[8] = m[4] * m[9] * m[15] -
		m[4] * m[11] * m[13] -
		m[8] * m[5] * m[15] +
		m[8] * m[7] * m[13] +
		m[12] * m[5] * m[11] -
		m[12] * m[7] * m[9];

	inv[12] = -m[4] * m[9] * m[14] +
		m[4] * m[10] * m[13] +
		m[8] * m[5] * m[14] -
		m[8] * m[6] * m[13] -
		m[12] * m[5] * m[10] +
		m[12] * m[6] * m[9];

	inv[1] = -m[1] * m[10] * m[15] +
		m[1] * m[11] * m[14] +
		m[9] * m[2] * m[15] -
		m[9] * m[3] * m[14] -
		m[13] * m[2] * m[11] +
		m[13] * m[3] * m[10];

	inv[5] = m[0] * m[10] * m[15] -
		m[0] * m[11] * m[14] -
		m[8] * m[2] * m[15] +
		m[8] * m[3] * m[14] +
		m[12] * m[2] * m[11] -
		m[12] * m[3] * m[10];

	inv[9] = -m[0] * m[9] * m[15] +
		m[0] * m[11] * m[13] +
		m[8] * m[1] * m[15] -
		m[8] * m[3] * m[13] -
		m[12] * m[1] * m[11] +
		m[12] * m[3] * m[9];

	inv[13] = m[0] * m[9] * m[14] -
		m[0] * m[10] * m[13] -
		m[8] * m[1] * m[14] +
		m[8] * m[2] * m[13] +
		m[12] * m[1] * m[10] -
		m[12] * m[2] * m[9];

	inv[2] = m[1] * m[6] * m[15] -
		m[1] * m[7] * m[14] -
		m[5] * m[2] * m[15] +
		m[5] * m[3] * m[14] +
		m[13] * m[2] * m[7] -
		m[13] * m[3] * m[6];

	inv[6] = -m[0] * m[6] * m[15] +
		m[0] * m[7] * m[14] +
		m[4] * m[2] * m[15] -
		m[4] * m[3] * m[14] -
		m[12] * m[2] * m[7] +
		m[12] * m[3] * m[6];

	inv[10] = m[0] * m[5] * m[15] -
		m[0] * m[7] * m[13] -
		m[4] * m[1] * m[15] +
		m[4] * m[3] * m[13] +
		m[12] * m[1] * m[7] -
		m[12] * m[3] * m[5];

	inv[14] = -m[0] * m[5] * m[14] +
		m[0] * m[6] * m[13] +
		m[4] * m[1] * m[14] -
		m[4] * m[2] * m[13] -
		m[12] * m[1] * m[6] +
		m[12] * m[2] * m[5];

	inv[3] = -m[1] * m[6] * m[11] +
		m[1] * m[7] * m[10] +
		m[5] * m[2] * m[11] -
		m[5] * m[3] * m[10] -
		m[9] * m[2] * m[7] +
		m[9] * m[3] * m[6];

	inv[7] = m[0] * m[6] * m[11] -
		m[0] * m[7] * m[10] -
		m[4] * m[2] * m[11] +
		m[4] * m[3] * m[10] +
		m[8] * m[2] * m[7] -
		m[8] * m[3] * m[6];

	inv[11] = -m[0] * m[5] * m[11] +
		m[0] * m[7] * m[9] +
		m[4] * m[1] * m[11] -
		m[4] * m[3] * m[9] -
		m[8] * m[1] * m[7] +
		m[8] * m[3] * m[5];

	inv[15] = m[0] * m[5] * m[10] -
		m[0] * m[6] * m[9] -
		m[4] * m[1] * m[10] +
		m[4] * m[2] * m[9] +
		m[8] * m[1] * m[6] -
		m[8] * m[2] * m[5];

	det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];

	if (det == 0)
		return false;

	det = 1.0 / det;

	for (i = 0; i < 16; i++)
		invOut[i] = inv[i] * det;

	return true;
}
/// polar decomposition can be reconstructed trivially from SVD result
/// A = UP
__host__ __device__ __forceinline__
void pd(const float *a,
	float *u, float *p)
{
	float w[9];
	float s[9];
	float v[9];

	svd(a, w, s, v);

	// P = VSV'
	float t[9];
	multAB(v, s, t);

	multAB(t, v, p);

	// U = WV'
	multAB(w, v, u);
}

#endif
