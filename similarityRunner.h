#ifndef SIMILARITYRUNNER_H
#define SIMILARITYRUNNER_H

#include <math.h>

#include "PieceMem.h"
#include "util.h"
#include "denseMatrixData.h"
#include "BiMatchData.h"
#include "mathOperationManager.h"
#include "runnerBehavior.h"

using namespace std;

class SimilarityRunner
{
public:
	SimilarityRunner(float gamma, RunnerBehavior * rb)
	{
		Gamma = gamma;
		RB = rb;
	}
	
	void Forward(DenseMatrix * inputA, DenseMatrix * inputB, BiMatchData * matchData, DenseMatrix * output)
	{
		if (ASquare == NULL) ASquare = new PieceMem<float>(inputA->RowSize, RB->Device);
		if (BSquare == NULL) BSquare = new PieceMem<float>(inputB->RowSize, RB->Device);

		if (ASquare->Size < inputA->RowSize) ASquare->Resize(inputA->RowSize);
		if (BSquare->Size < inputB->RowSize) BSquare->Resize(inputB->RowSize);

		InputA = inputA;
		InputB = inputB;
		MatchData = matchData;
		Output = output;

		Output->RowSize = MatchData->SrcSize;

		RB->ComputeLib->VecMulVec(InputA->Data, InputA->Data, ASquare, InputA->Stat->MAX_COL_SIZE, InputA->RowSize, 0, 1);
		RB->ComputeLib->VecMulVec(InputB->Data, InputB->Data, BSquare, InputB->Stat->MAX_COL_SIZE, InputB->RowSize, 0, 1);

		RB->ComputeLib->CosineSimilarity(InputA->Data, ASquare, InputA->RowSize, 
									     InputB->Data, BSquare, InputB->RowSize, InputA->Stat->MAX_COL_SIZE,
									     Output->Data, MatchData->SrcIdx, MatchData->TgtIdx, MatchData->MatchSize, LARGEEPS);

		RB->ComputeLib->Scale(Output->Data, Gamma, MatchData->MatchSize);
	}

	void Backward(DenseMatrix * doutput, DenseMatrix * dinputA, DenseMatrix * dinputB)
	{
		RB->ComputeLib->Scale(doutput->Data, Gamma, MatchData->MatchSize);
		RB->ComputeLib->Scale(Output->Data, 1.0f/Gamma, MatchData->MatchSize);
		
		RB->ComputeLib->DerivCosineSimilarity(InputA->Data, InputB->Data, ASquare, BSquare, InputA->Stat->MAX_COL_SIZE,
                                              MatchData->Src2MatchIdx, MatchData->Src2MatchElement, MatchData->Tgt2MatchIdx, MatchData->Tgt2MatchElement,
                                              MatchData->SrcIdx, MatchData->TgtIdx, MatchData->SrcSize, MatchData->TgtSize, MatchData->MatchSize, 
											  Output->Data, doutput->Data, dinputA->Data, dinputB->Data, 0, LARGEEPS);
	}

	~SimilarityRunner()
	{
		if (ASquare != NULL) free(ASquare);
		if (BSquare != NULL) free(BSquare);
	}
private:
	PieceMem<float> * ASquare;
	PieceMem<float> * BSquare;

	DenseMatrix * InputA;
	DenseMatrix * InputB;
	BiMatchData * MatchData;
	DenseMatrix * Output;

	float Gamma;
	RunnerBehavior * RB;
};

#endif
