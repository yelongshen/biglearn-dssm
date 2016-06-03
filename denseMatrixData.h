// sparseMatrixData.h


#ifndef DENSEMATRIX_H
#define DENSEMATRIX_H
#include <iostream>
#include <vector>
#include <map>

#include "PieceMem.h"
#include "runnerBehavior.h"
using namespace std;


class DenseMatrixStat
{
public:
	int MAX_ROW_SIZE;
	int MAX_COL_SIZE;

	int TOTAL_BATCH_NUM;
	int TOTAL_SAMPLE_NUM;
};

// row major sparse matrix
class DenseMatrix
{
public:
	DenseMatrixStat * Stat;
	int RowSize;
	PieceMem<float> * Data;
	DEVICE Device;

	DenseMatrix(DenseMatrixStat * stat, DEVICE device)
	{
		Device = device;
		Stat = stat;
		Init();
	}

	void Init()
	{
		Data = new PieceMem<float>(Stat->MAX_ROW_SIZE * Stat->MAX_COL_SIZE, Device);
	}
};
#endif