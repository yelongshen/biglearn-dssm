// sparseMatrixData.h


#ifndef SPARSEINDEXMATRIX_H
#define SPARSEINDEXMATRIX_H
#include <iostream>
#include <vector>
#include <map>

#include "PieceMem.h"

using namespace std;

class SparseIndexMatrixStat
{
public:
	int MAX_ROW_SIZE;
	int MAX_COL_SIZE;
	int MAX_ELEMENT_SIZE;

	int TOTAL_BATCH_NUM;
    int TOTAL_SAMPLE_NUM;
};

// row major sparse matrix
class SparseIndexMatrix
{
public:
	SparseIndexMatrixStat * Stat;
	int RowSize;
	int ElementSize;

	PieceMem<int> * SampleIdx; 
	PieceMem<int> * FeatureIdx;
    
    DEVICE Device;

	SparseIndexMatrix(SparseIndexMatrixStat * stat, DEVICE device)
	{
		Device = device;
		Stat = stat;
		Init();
	}

	void Init()
	{
		SampleIdx = new PieceMem<int>(Stat->MAX_ROW_SIZE, Device);
		FeatureIdx = new PieceMem<int>(Stat->MAX_ELEMENT_SIZE, Device);
	}

	void Refresh()
	{
		RowSize = 0;
		ElementSize = 0;
	}

	int PushSample(int * feaIdx, int size);
};
#endif