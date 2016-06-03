// sparseMatrixData.h


#ifndef HIDDENDENSEMATRIX_H
#define HIDDENDENSEMATRIX_H
#include <iostream>
#include <vector>
#include <map>

#include "PieceMem.h"
#include "runnerBehavior.h"
#include "denseMatrixData.h"
using namespace std;

// row major sparse matrix
class HiddenDenseMatrix
{
public:
	DenseMatrixStat * Stat;
	DEVICE Device;

	DenseMatrix * Output;
	DenseMatrix * Deriv;

	HiddenDenseMatrix(DenseMatrixStat * stat, DEVICE device)
	{
		Output = new DenseMatrix(stat, device);
		Deriv = new DenseMatrix(stat, device);
		Device = device;
		Stat = stat;
	}
};
#endif