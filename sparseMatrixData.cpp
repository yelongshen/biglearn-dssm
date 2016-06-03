//sparseMatrixData.cpp
#include "PieceMem.h"
#include "sparseMatrixData.h"

int SparseIndexMatrix::PushSample(int * feaIdx, int size)
{
	int incr = 1024;

	if (SampleIdx->Size < RowSize + 1) 
		SampleIdx->Resize(SampleIdx->Size + 1 + incr);
	if (FeatureIdx->Size < ElementSize + size) 
		FeatureIdx->Resize(FeatureIdx->Size + size + incr);

	FeatureIdx->SyncFromHost(ElementSize, feaIdx, size);
	//FeatureIdx->SyncToHost(ElementSize, size);
	ElementSize += size;

	SampleIdx->SyncFromHost(RowSize, &ElementSize, 1);
	//SampleIdx->SyncToHost(RowSize, 1);
	RowSize += 1;


	if (Stat->MAX_ROW_SIZE < RowSize) Stat->MAX_ROW_SIZE = RowSize;
	if (Stat->MAX_ELEMENT_SIZE < ElementSize) Stat->MAX_ELEMENT_SIZE = ElementSize;

	return size;
}