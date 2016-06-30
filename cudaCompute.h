#ifndef CUDACOMPUTE_H
#define CUDACOMPUTE_H
#include <iostream> 
#include <vector> 
#include <cuda_runtime.h> 
#include <cublas.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_surface_types.h>
#include "device_launch_parameters.h" //device_launch_parameters.h"
//#include <comutil.h>
#include <stdint.h>
#include <stdio.h>

#include <cfloat>
#include <stdlib.h>

#include "cublas_v2.h"
#pragma comment(lib, "cudart") 
#pragma comment(lib,"cublas.lib")

#define DEFAULT_THREAD_PER_BLOCK    512    
#define DEFAULT_THREAD_PER_DIM		32
#define DEFAULT_THREAD_PER_DIM_3D	8
#define MAX_BATCH_SIZE              256
#define MAX_THREAD_NUM			1024
#define MAX_BLOCK_NUM			65536
#define COL_FOLD_NUM			4096

void Cuda_SparseIndexForward(int * rowIdx, int * sparseColIndex, float * weight, int rowSize, int inputDim, int outputDim, float * output, float alpha, float beta);
void Cuda_SparseIndexBackward(int * rowIdx, int * sparseColIndex, float * doutput, int rowSize, int inputDim, int outputDim, float * weight, float beta);

void Cuda_Tanh(float * a, float * b, int size);
void Cuda_DerivTanh(float * doutput, float * output, float * dinput, float alpha, int size);

void Cuda_VecMulVec(float * pLeft, float * pRight, float * pDst, int dim, int size, float alpha, float beta);
void Cuda_IVecMulVec(float * pLeft, int * leftIdxA, float * pRight, int * rightIdxB, float * pDst, int dim, int size, float alpha, float beta);

void Cuda_CosineSimilarity(float * inputA, float * ASquare, int ASize, float * inputB, float * BSquare, int BSize,
	int dim, float * outputC, int * matchIdxA, int * matchIdxB, int matchSize, float eps);

void Cuda_DerivCosineSimilarity(float * q, float * d, float *qSquare, float * dSquare, int dim,
		int * src2MatchIdx, int * src2MatchElement, int * tgt2MatchIdx, int * tgt2MatchElement, int * srcIdx, int * tgtIdx,  int srcSize, int tgtSize, int matchSize, 
		float * simi, float * derivSimi, float * dcq, float * dcd, float alpha, float eps);
#endif
