#ifndef FULLYCONNECTEDLAYER_H
#define FULLYCONNECTEDLAYER_H

#include <math.h>

#include "PieceMem.h"
#include "util.h"
#include "sparseMatrixData.h"
#include "denseMatrixData.h"
#include "mathOperationManager.h"


//fully connected layer.h
// simple version of fully connected layer. only support tanh activation function and no bias.

class FullyConnectedLayer
{
public:
	int InputDim;
	int OutputDim;
	PieceMem<float> * Weight;
	
	FullyConnectedLayer(int inputDim, int outputDim, RunnerBehavior * rb)
	{
		InputDim = inputDim;
		OutputDim = outputDim;
		RB = rb;
		Weight = new PieceMem<float>(InputDim * OutputDim, RB->Device);
		
		RB->ComputeLib->RandomVec(Weight, InputDim * OutputDim, sqrtf(6.0f / (inputDim + outputDim)), -sqrtf(6.0f / (inputDim + outputDim)));
	}
	
	void Forward(SparseIndexMatrix * input, DenseMatrix * output)
	{
		output->RowSize = input->RowSize;

		//input->SampleIdx->QuickWatch();
		//input->FeatureIdx->QuickWatch();
		//Weight->QuickWatch();

		RB->ComputeLib->SparseIndexForward(input->SampleIdx, input->FeatureIdx, Weight, input->RowSize, InputDim, OutputDim, output->Data, 0, 1);
		
		//output->Data->QuickWatch();

		RB->ComputeLib->Tanh(output->Data, output->Data, output->RowSize * OutputDim);

		//output->Data->QuickWatch();
	}

	void Forward(DenseMatrix * input, DenseMatrix * output)
	{
		output->RowSize = input->RowSize;
		RB->ComputeLib->Sgemm(input->Data, Weight, output->Data, input->RowSize, InputDim, OutputDim, 0, 1, false, false);
		RB->ComputeLib->Tanh(output->Data, output->Data, output->RowSize * OutputDim);
	}

	void Backward(DenseMatrix * doutput, DenseMatrix * output, DenseMatrix * dinput)
	{
		RB->ComputeLib->DerivTanh(doutput->Data, output->Data, doutput->Data, 0,  output->RowSize * OutputDim);
		RB->ComputeLib->Sgemm(doutput->Data, Weight, dinput->Data, output->RowSize, OutputDim, InputDim, 0, 1, false, true);
	}

	void Backward(DenseMatrix * doutput, DenseMatrix * output)
	{
		RB->ComputeLib->DerivTanh(doutput->Data, output->Data, doutput->Data, 0,  output->RowSize * OutputDim);
	}

	void Update(DenseMatrix * doutput, DenseMatrix * input)
	{
		RB->ComputeLib->Sgemm(input->Data, doutput->Data, Weight, input->RowSize, InputDim, OutputDim, 1, 0.0005f, true, false);
	}

	void Update(DenseMatrix * doutput, SparseIndexMatrix * input)
	{
		RB->ComputeLib->SparseIndexBackward(input->SampleIdx, input->FeatureIdx, doutput->Data,
			 input->RowSize, InputDim, OutputDim, Weight, 0.0005f);
	}

	~FullyConnectedLayer()
	{
		free(Weight);
	}
private:
	RunnerBehavior * RB;
};

#endif
