#ifndef MATHOPERATIONMANAGER_H
#define MATHOPERATIONMANAGER_H

#include <iostream>
#include <vector>
#include <map>
#include <tuple>
#include <stdio.h>
#include <cstring>
#include <cstdio>

#include "PieceMem.h"

using namespace std;

#define LARGEEPS 1.0e-6f
#define SMALLEPS 1.0e-9f

class IMathOperationManager
{
public:
	virtual ~IMathOperationManager(){}

	// C = alpha C + beta op(A) op(B)
	virtual void Sgemm(PieceMem<float> * A, PieceMem<float> * B, PieceMem<float> * C, int rowA, int inDim, int outDim, float alpha, float beta, bool transA, bool transB){}

	// random vec between (lower, upper)
	virtual void RandomVec(PieceMem<float> * A, int size, float upper, float lower){}

	// Sparse Index Forward.
	virtual void SparseIndexForward(PieceMem<int> * rowIdx, PieceMem<int> * sparseColIndex, PieceMem<float> * weight,
		int rowSize, int inputDim, int outputDim, PieceMem<float> * output, float alpha, float beta){}

	// Sparse Index Backward.
	virtual void SparseIndexBackward(PieceMem<int> * rowIdx, PieceMem<int> * sparseColIndex, PieceMem<float> * doutput, 
		int rowSize, int inputDim, int outputDim, PieceMem<float> * weight, float beta){}

	// output = Tanh(input).
	virtual void Tanh(PieceMem<float> * input, PieceMem<float> * output, int size){}

	// dinput = dinput[idx] * alpha + doutput[idx] * (1 - output[idx]) * (1 + output[idx]);
	virtual void DerivTanh(PieceMem<float> * doutput, PieceMem<float> * output, PieceMem<float> * dinput, float alpha, int size){}

	// outputC = alpha outputC + beta inputA @ inputB.
	virtual void VecMulVec(PieceMem<float> * inputA, PieceMem<float> * inputB, PieceMem<float> * outputC, int dim, int size, float alpha, float beta){}

	// outputC = alpha outputC + beta inputA[leftidx] @ inputB[rightidx].
	virtual void IVecMulVec(PieceMem<float> * pLeft, PieceMem<int> * leftIdxA, 
			PieceMem<float> * pRight, PieceMem<int> * rightIdxB, PieceMem<float> * pDst, int dim, int size, float alpha, float beta){}

	// outputC = cosineSimilarity(inputA, inputB)
	virtual void CosineSimilarity(PieceMem<float> * inputA, PieceMem<float> * ASquare, int ASize, PieceMem<float> * inputB, PieceMem<float> * BSquare, int BSize,
		int dim, PieceMem<float> * outputC, PieceMem<int> * matchIdxA, PieceMem<int> * matchIdxB, int matchSize, float eps){}

	virtual void DerivCosineSimilarity(PieceMem<float> * q, PieceMem<float> * d, PieceMem<float> *qSquare, PieceMem<float> * dSquare, int dim,
		PieceMem<int> * src2MatchIdx, PieceMem<int> * src2MatchElement, PieceMem<int> * tgt2MatchIdx, PieceMem<int> * tgt2MatchElement, 
		PieceMem<int> * srcIdx, PieceMem<int> * tgtIdx,  int srcSize, int tgtSize, int matchSize, 
		PieceMem<float> * simi, PieceMem<float> * derivSimi, PieceMem<float> * dcq, PieceMem<float> * dcd, float alpha, float eps){}

	virtual void SoftmaxForward(PieceMem<float> * input, PieceMem<float> * output, int batchSize, int dim){}

	virtual void SoftmaxBackward(PieceMem<float> * y, PieceMem<float> * dy, PieceMem<float> * dx, int batchSize, int dim){}

	// y = y + alpha x;
	virtual void VecAdd(PieceMem<float> * x, float alpha, PieceMem<float> * y, int size){}
	
	// dst = dst * alpha + xScale x + yScale y;
	virtual void VecAdd(PieceMem<float> * x, float xScale, PieceMem<float> * y, float yScale, PieceMem<float> * dst, float alpha, int size){}

	virtual void Scale(PieceMem<float> * x, float scale, int size){}
};
#endif