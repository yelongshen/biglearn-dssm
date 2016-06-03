#ifndef SIMILARITYRUNNER_H
#define SIMILARITYRUNNER_H

#include <math.h>

#include "PieceMem.h"
#include "util.h"
#include "denseMatrixData.h"
#include "mathOperationManager.h"
#include "runnerBehavior.h"

using namespace std;

class SoftmaxRunner
{
public:
	SoftmaxRunner(RunnerBehavior * rb)
	{
		RB = rb;
	}

	float Loss;

	void Forward(DenseMatrix * input, float gamma, DenseMatrix * output)
	{
		output->RowSize = input->RowSize;
		RB->ComputeLib->SoftmaxForward(input->Data, output->Data, input->RowSize, input->Stat->MAX_COL_SIZE);
	}

private:
	RunnerBehavior * RB;
};

#endif
