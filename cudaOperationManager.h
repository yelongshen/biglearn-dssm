#ifndef CUDAOPERATIONMANAGER_H
#define CUDAOPERATIONMANAGER_H

#include <math.h>
#include <cuda_runtime.h> 
#include <cublas.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_surface_types.h>
#include "device_launch_parameters.h" //device_launch_parameters.h"
#include <comutil.h>
#include <stdint.h>
#include <stdio.h>

#include <curand.h>
#include <curand_kernel.h>
#include <stdlib.h>
#include <cudnn.h>
#include <cusparse.h>

#include "cublas_v2.h"
#include "device_functions.h"

#include "PieceMem.h"
#include "cudaCompute.h"
#include "mathOperationManager.h"

#pragma comment(lib, "cudart") 
#pragma comment(lib, "cuda") 
#pragma comment(lib, "cudnn")
#pragma comment(lib,"cublas.lib")

using namespace std;
using namespace _com_util;


class CudaOperationManager : public IMathOperationManager
{
public: 

	// C = alpha C + beta op(A) op(B)
	virtual void Sgemm(PieceMem<float> * A, PieceMem<float> * B, PieceMem<float> * C, int rowA, int inDim, int outDim, float alpha, float beta, bool transA, bool transB)
	{
		if (!transA && !transB)
		{
			/// c = beta A * B + alpha C
			cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, outDim, rowA, inDim, &beta, B->Mem, outDim, A->Mem, inDim, &alpha, C->Mem, outDim);
		}
		else if (!transA && transB)
		{
			/// c = beta op(A) * B + alpha C
			cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, outDim, rowA, inDim, &beta, B->Mem, inDim, A->Mem, inDim, &alpha, C->Mem, outDim);
		}
		else if (transA && !transB)
		{
			/// c = beta A * op(B) + alpha C
			cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, outDim, inDim, rowA, &beta, B->Mem, outDim, A->Mem, inDim, &alpha, C->Mem, outDim);
		}
	}

	virtual void VecMulVec(PieceMem<float> * inputA, PieceMem<float> * inputB, PieceMem<float> * outputC, int dim, int size, float alpha, float beta)
	{
		Cuda_VecMulVec(inputA->Mem, inputB->Mem, outputC->Mem, dim, size, alpha, beta);
	}

	virtual void IVecMulVec(PieceMem<float> * pLeft, PieceMem<int> * leftIdxA, 
				PieceMem<float> * pRight, PieceMem<int> * rightIdxB, PieceMem<float> * pDst, int dim, int size, float alpha, float beta)
	{
		Cuda_IVecMulVec(pLeft->Mem, leftIdxA->Mem, pRight->Mem, rightIdxB->Mem, pDst->Mem, dim, size, alpha, beta);
	}

	virtual void SparseIndexForward(PieceMem<int> * rowIdx, PieceMem<int> * sparseColIndex, PieceMem<float> * weight, int rowSize, int inputDim, int outputDim, PieceMem<float> * output, float alpha, float beta)
	{
		Cuda_SparseIndexForward(rowIdx->Mem, sparseColIndex->Mem, weight->Mem, rowSize, inputDim, outputDim, output->Mem, alpha, beta);
	}

	virtual void SparseIndexBackward(PieceMem<int> * rowIdx, PieceMem<int> * sparseColIndex, PieceMem<float> * doutput, 
		int rowSize, int inputDim, int outputDim, PieceMem<float> * weight, float beta)
	{
		Cuda_SparseIndexBackward(rowIdx->Mem, sparseColIndex->Mem, doutput->Mem, rowSize, inputDim, outputDim, weight->Mem, beta);
	}

	virtual void Tanh(PieceMem<float> * input, PieceMem<float> * output, int size)
	{
		Cuda_Tanh(input->Mem, output->Mem, size);
	}

	virtual void DerivTanh(PieceMem<float> * doutput, PieceMem<float> * output, PieceMem<float> * dinput, float alpha, int size)
	{
		Cuda_DerivTanh(doutput->Mem, output->Mem, dinput->Mem, alpha, size);
	}

	virtual void CosineSimilarity(PieceMem<float> * inputA, PieceMem<float> * ASquare, int ASize, PieceMem<float> * inputB, PieceMem<float> * BSquare, int BSize,
		int dim, PieceMem<float> * outputC, PieceMem<int> * matchIdxA, PieceMem<int> * matchIdxB, int matchSize, float eps)
	{
		Cuda_CosineSimilarity(inputA->Mem, ASquare->Mem, ASize, inputB->Mem, BSquare->Mem, BSize, dim, outputC->Mem, matchIdxA->Mem, matchIdxB->Mem, matchSize, eps);
	}

	virtual void DerivCosineSimilarity(PieceMem<float> * q, PieceMem<float> * d, PieceMem<float> *qSquare, PieceMem<float> * dSquare, int dim,
		PieceMem<int> * src2MatchIdx, PieceMem<int> * src2MatchElement, PieceMem<int> * tgt2MatchIdx, PieceMem<int> * tgt2MatchElement, 
		PieceMem<int> * srcIdx, PieceMem<int> * tgtIdx,  int srcSize, int tgtSize, int matchSize, 
		PieceMem<float> * simi, PieceMem<float> * derivSimi, PieceMem<float> * dcq, PieceMem<float> * dcd, float alpha, float eps)
	{
		Cuda_DerivCosineSimilarity(q->Mem, d->Mem, qSquare->Mem, dSquare->Mem, dim, src2MatchIdx->Mem, src2MatchElement->Mem, 
			tgt2MatchIdx->Mem, tgt2MatchElement->Mem, srcIdx->Mem, tgtIdx->Mem, srcSize, tgtSize, matchSize,
			simi->Mem, derivSimi->Mem, dcq->Mem, dcd->Mem, alpha, eps);
	}

	virtual void SoftmaxForward(PieceMem<float> * input, PieceMem<float> * output, int batchSize, int dim)
	{
		cudnnTensorDescriptor_t pDesc;
		cudnnCreateTensorDescriptor(&pDesc);
		cudnnSetTensor4dDescriptor(pDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batchSize, dim, 1, 1);

		float alpha = 0;
		float beta = 1;
		cudnnSoftmaxForward(cuDnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, &beta, pDesc, input->Mem, &alpha, pDesc, output->Mem);
	}

	virtual void SoftmaxBackward(PieceMem<float> * y, PieceMem<float> * dy, PieceMem<float> * dx, int batchSize, int dim)
	{
		cudnnTensorDescriptor_t pDesc;
		cudnnCreateTensorDescriptor(&pDesc);
		cudnnSetTensor4dDescriptor(pDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batchSize, dim, 1, 1);

		float alpha = 0;
		float beta = 1;
		cudnnSoftmaxBackward(cuDnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, &beta, pDesc, y->Mem, pDesc, dy->Mem, &alpha, pDesc, dx->Mem);
	}

	virtual void VecAdd(PieceMem<float> * x, float alpha, PieceMem<float> * y, int size)
	{
		cublasSaxpy(cublasHandle, size, &alpha, x->Mem, 1, y->Mem, 1);
	}

	virtual void VecAdd(PieceMem<float> * x, float xScale, PieceMem<float> * y, float yScale, PieceMem<float> * dst, float alpha, int size)
	{
		cublasSscal(cublasHandle, size, &alpha, dst->Mem, 1);
		cublasSaxpy(cublasHandle, size, &xScale, x->Mem, 1, dst->Mem, 1);
		cublasSaxpy(cublasHandle, size, &yScale, y->Mem, 1, dst->Mem, 1);
	}

	virtual void Scale(PieceMem<float> * x, float scale, int size)
	{
		cublasSscal(cublasHandle, size, &scale, x->Mem, 1);
	}

	virtual void RandomVec(PieceMem<float> * A, int size, float upper, float lower)
	{
		float * host = new float[size];
		float range = upper - lower;
		for (int i = 0; i < size; i++) host[i] = ((float)rand() / RAND_MAX) * range + lower;
		A->SyncFromHost(0, host, size);
		free(host);
	}

	CudaOperationManager(bool isCuDNN, bool isCuBlas)
	{
		IsCuDNN = isCuDNN;
		IsCuBlas = isCuBlas;

		if (IsCuDNN) cublasCreate(&cublasHandle);
		if (IsCuBlas) cudnnCreate(&cuDnnHandle);
	}

	virtual ~CudaOperationManager()
	{
		if (IsCuDNN) cublasDestroy(cublasHandle);
		if (IsCuBlas) cudnnDestroy(cuDnnHandle);
	}

private:
	bool IsCuBlas;
	cublasHandle_t cublasHandle;
	bool IsCuDNN;
	cudnnHandle_t cuDnnHandle;
};

#endif