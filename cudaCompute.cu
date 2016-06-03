#include <iostream> 
#include <vector> 
#include <cuda_runtime.h> 
#include <cublas.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_surface_types.h>
#include "device_launch_parameters.h" //device_launch_parameters.h"
#include <comutil.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <curand.h>
#include <curand_kernel.h>

#include "device_functions.h"

#include <windows.h>
#include <fstream>

#include "cublas_v2.h"
#include "cudaCompute.h"

#pragma comment(lib, "cudart") 
#pragma comment(lib,"cublas.lib")

using namespace std;
using namespace _com_util;


__global__ void cuda_SparseIndexForward(int * rowIdx, int * sparseColIndex, float * weight, int rowSize, int inputDim, int outputDim, float * output, float alpha, float beta)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;

	if (idx < rowSize && idy < outputDim)
	{
		int fea_end = rowIdx[idx];
		int fea_begin = idx > 0 ? rowIdx[idx - 1] : 0;

		float sum = 0;
		for (int i = fea_begin; i < fea_end; ++i)
		{
			int fea_idx = sparseColIndex[i];
			sum += weight[fea_idx * outputDim + idy];
		}
		output[idx * outputDim + idy] = alpha * output[idx * outputDim + idy] + beta * sum;
	}
}
void Cuda_SparseIndexForward(int * rowIdx, int * sparseColIndex, float * weight, int rowSize, int inputDim, int outputDim, float * output, float alpha, float beta)
{
	dim3 thread_tail(DEFAULT_THREAD_PER_DIM, DEFAULT_THREAD_PER_DIM);
	dim3 block_tail((rowSize - 1) / DEFAULT_THREAD_PER_DIM + 1, (outputDim - 1) / DEFAULT_THREAD_PER_DIM + 1);
	cuda_SparseIndexForward << <block_tail, thread_tail >> >(rowIdx, sparseColIndex, weight, rowSize, inputDim, outputDim, output, alpha, beta);
}


__global__ void cuda_SparseIndexBackward(int * rowIdx, int * sparseColIndex, float * doutput, int rowSize, int inputDim, int outputDim, float * weight, float beta)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < outputDim)
	{	
		for (int b = 0; b < rowSize; b++)
		{
			int col_end = rowIdx[b];
			int col_begin = b == 0 ? 0 : rowIdx[b - 1];	
			float dv = beta * doutput[b * outputDim + idx];
			for (int i = col_begin; i < col_end; i++)
			{
				int fea_idx = sparseColIndex[i];
				weight[fea_idx * outputDim + idx] += dv;
			}
		}
	}
}
void Cuda_SparseIndexBackward(int * rowIdx, int * sparseColIndex, float * doutput, int rowSize, int inputDim, int outputDim, float * weight, float beta)
{
	dim3 thread_tail(DEFAULT_THREAD_PER_BLOCK);
	dim3 block_tail((outputDim - 1) / DEFAULT_THREAD_PER_BLOCK + 1);

	cuda_SparseIndexBackward << <block_tail, thread_tail >> >(rowIdx, sparseColIndex, doutput, rowSize, inputDim, outputDim, weight, beta);
}

__global__ void cuda_Tanh(float * a, float * b, int size)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < size) b[idx] = tanhf(a[idx]);
}
void Cuda_Tanh(float * a, float * b, int size)
{
	int nThreadPerBlock = DEFAULT_THREAD_PER_BLOCK;
	int nBlockPerGrid = (size + DEFAULT_THREAD_PER_BLOCK - 1) / DEFAULT_THREAD_PER_BLOCK;
	cuda_Tanh << <nBlockPerGrid, nThreadPerBlock >> >(a, b, size);
}

__global__ void cuda_DerivTanh(float * doutput, float * output, float * dinput, float alpha, int size)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < size)
	{
		dinput[idx] = dinput[idx] * alpha + doutput[idx] * (1 - output[idx]) * (1 + output[idx]);
	}
}
// dinput[idx] = dinput[idx] * alpha + doutput[idx] * (1 - output[idx]) * (1 + output[idx]);
void Cuda_DerivTanh(float * doutput, float * output, float * dinput, float alpha, int size)
{
	int nThreadPerBlock = DEFAULT_THREAD_PER_BLOCK;
	int nBlockPerGrid = (size + DEFAULT_THREAD_PER_BLOCK - 1) / DEFAULT_THREAD_PER_BLOCK;
	cuda_DerivTanh <<<nBlockPerGrid, nThreadPerBlock >> >(doutput, output, dinput, alpha, size);
}

__global__ void cuda_VecMulVec(float * pLeft, float * pRight, float * pDst, int dim, int size, float alpha, float beta)
{
	int x = blockDim.x*blockIdx.x + threadIdx.x;
	if (x < size)
	{
		float sum = 0;

		int offset = x * dim;
		for (int i = 0; i < dim; i++)
		{
			sum += pLeft[offset + i] * pRight[offset + i];
		}
		pDst[x] = alpha * pDst[x] + beta * sum;
	}
}
// pDst = pDst * weiDst + pLeft @ pRight;
void Cuda_VecMulVec(float * pLeft, float * pRight, float * pDst, int dim, int size, float alpha, float beta)
{
	int nThreadPerBlock = DEFAULT_THREAD_PER_BLOCK;
	int nBlockPerGrid = (size + DEFAULT_THREAD_PER_BLOCK - 1) / DEFAULT_THREAD_PER_BLOCK;
	cuda_VecMulVec << <nBlockPerGrid, nThreadPerBlock >> >(pLeft, pRight, pDst, dim, size, alpha, beta);
}

__global__ void cuda_IVecMulVec(float * pLeft, int * leftIdxA, float * pRight, int * rightIdxB, float * pDst, int dim, int size, float alpha, float beta)
{
	int x = blockDim.x*blockIdx.x + threadIdx.x;
	if (x < size)
	{
		int leftIdx = leftIdxA[x] * dim;
		int rightIdx = rightIdxB[x] * dim;

		float sum = 0;
		for (int i = 0; i < dim; i++) sum += pLeft[leftIdx + i] * pRight[rightIdx + i];

		pDst[x] = alpha * pDst[x] + beta * sum;
	}
}
// pDst = pDst * weiDst + pLeft @ pRight;
void Cuda_IVecMulVec(float * pLeft, int * leftIdxA, float * pRight, int * rightIdxB, float * pDst, int dim, int size, float alpha, float beta)
{
	int nThreadPerBlock = DEFAULT_THREAD_PER_BLOCK;
	int nBlockPerGrid = (size + DEFAULT_THREAD_PER_BLOCK - 1) / DEFAULT_THREAD_PER_BLOCK;
	cuda_IVecMulVec << <nBlockPerGrid, nThreadPerBlock >> >(pLeft, leftIdxA, pRight, rightIdxB, pDst, dim, size, alpha, beta);
}


__global__ void cuda_CosineSimilarity(float * inputA, float * ASquare, int ASize, float * inputB, float * BSquare, int BSize,
	int dim, float * outputC, int * matchIdxA, int * matchIdxB, int matchSize, float eps)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < matchSize)
	{
		int aid = matchIdxA[idx];
		int bid = matchIdxB[idx];
		float sumxx = sqrtf(ASquare[aid]);
		float sumyy = sqrtf(BSquare[bid]);
		float sumxy = 0;
		float * ptrA = inputA + aid * dim;
		float * ptrB = inputB + bid * dim;
		
		for (int i = 0; i < dim; i++) sumxy += ptrA[i] * ptrB[i];
		outputC[idx] = (float)(sumxy * 1.0f / ((float)(sumxx * sumyy) + eps));
	}
}
void Cuda_CosineSimilarity(float * inputA, float * ASquare, int ASize, float * inputB, float * BSquare, int BSize,
	int dim, float * outputC, int * matchIdxA, int * matchIdxB, int matchSize, float eps)
{
	int nThreadPerBlock = DEFAULT_THREAD_PER_BLOCK;
	int nBlockPerGrid = (matchSize + DEFAULT_THREAD_PER_BLOCK - 1) / DEFAULT_THREAD_PER_BLOCK;
	cuda_CosineSimilarity << <nBlockPerGrid, nThreadPerBlock >> >(inputA, ASquare, ASize, inputB, BSquare, BSize, dim, outputC, matchIdxA, matchIdxB, matchSize, eps);
}


__global__ void cuda_Deriv_CosineSimilarity_partialMatching(float * src, float * tgt, float *srcSquare, float * tgtSquare, int dim,
		int * src2MatchIdx, int * src2MatchElement, int * tgtIdx, int srcSize, int matchSize, float * simi, float * derivSimi,
		float * dcSrc, float alpha, float eps)
{
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	//idx -> source/q Index.
	if (idx < srcSize && idy < dim)
	{
		int matchBeginIdx = idx == 0 ? 0 : src2MatchIdx[idx - 1];
		int matchEndIdx = src2MatchIdx[idx];

		float sum = 0;
		float qRoot = sqrtf(srcSquare[idx]);
		float qv = src[idx * dim + idy];
		float qSquare_qv = qv / (srcSquare[idx] + eps);
		for (int match = matchBeginIdx; match < matchEndIdx; match++)
		{
			int mIdx = src2MatchElement[match];
			int dIdx = tgtIdx[mIdx];
			float dRoot = sqrtf(tgtSquare[dIdx]);
			sum += derivSimi[mIdx] * (tgt[dIdx * dim + idy] / (qRoot * dRoot + eps) - qSquare_qv * simi[mIdx]); /// qSquare);
		}
		dcSrc[idx * dim + idy] = alpha * dcSrc[idx * dim + idy] + sum;
	}
}
void Cuda_DerivCosineSimilarity(float * q, float * d, float *qSquare, float * dSquare, int dim,
		int * src2MatchIdx, int * src2MatchElement, int * tgt2MatchIdx, int * tgt2MatchElement, int * srcIdx, int * tgtIdx,  int srcSize, int tgtSize, int matchSize, 
		float * simi, float * derivSimi, float * dcq, float * dcd, float alpha, float eps)
{
	dim3 srcThread_tail(DEFAULT_THREAD_PER_DIM, DEFAULT_THREAD_PER_DIM);
	dim3 srcBlock_tail((srcSize - 1) / DEFAULT_THREAD_PER_DIM + 1, (dim - 1) / DEFAULT_THREAD_PER_DIM + 1);

	cuda_Deriv_CosineSimilarity_partialMatching << <srcBlock_tail, srcThread_tail >> >(q, d, qSquare, dSquare, dim,
		src2MatchIdx, src2MatchElement, tgtIdx, srcSize, matchSize, simi, derivSimi, dcq, alpha, eps);

	dim3 tgtThread_tail(DEFAULT_THREAD_PER_DIM, DEFAULT_THREAD_PER_DIM);
	dim3 tgtBlock_tail((tgtSize - 1) / DEFAULT_THREAD_PER_DIM + 1, (dim - 1) / DEFAULT_THREAD_PER_DIM + 1);

	cuda_Deriv_CosineSimilarity_partialMatching << <tgtBlock_tail, tgtThread_tail >> >(d, q, dSquare, qSquare, dim,
		tgt2MatchIdx, tgt2MatchElement, srcIdx, tgtSize, matchSize, simi, derivSimi, dcd, alpha, eps);
}