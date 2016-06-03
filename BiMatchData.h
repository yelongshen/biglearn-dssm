// sparseMatrixData.h


#ifndef BIMATCHDATA_H
#define BIMATCHDATA_H
#include <iostream>
#include <vector>
#include <map>

#include "PieceMem.h"
#include "util.h"

using namespace std;


class BiMatchDataStat
{
public:
	int MAX_SRC_BATCHSIZE ;
	int MAX_TGT_BATCHSIZE ;
	int MAX_MATCH_BATCHSIZE;
	int TOTAL_BATCHNUM ;
	int TOTAL_MATCHNUM ;
};

class BiMatchData
{
public:
	BiMatchDataStat * Stat;
	int SrcSize;
	int TgtSize;
	int MatchSize;

	int NTrial;
	PieceMem<int> * SrcIdx;
	PieceMem<int> * TgtIdx;
	PieceMem<float> * MatchInfo;

	/// <summary>
	/// Src and Tgt Match Info.
	/// </summary>
	PieceMem<int> * Src2MatchIdx;
	PieceMem<int> * Src2MatchElement;
	PieceMem<int> * Tgt2MatchIdx;
	PieceMem<int> * Tgt2MatchElement;

	DEVICE Device;

	BiMatchData(BiMatchDataStat * stat, DEVICE device)
	{
		Device = device;
		Stat = stat;
		NTrial = -1;
		Init();
	}

	BiMatchData(int maxBatchSize, int nTrial, DEVICE device)
	{
		Device = device;

		Stat = new BiMatchDataStat();
		Stat->MAX_SRC_BATCHSIZE = maxBatchSize;
		Stat->MAX_TGT_BATCHSIZE = maxBatchSize;
		Stat->MAX_MATCH_BATCHSIZE = (nTrial + 1) * maxBatchSize;

		NTrial = nTrial;
		Init();
	}

	void GenerateMatch(int batchSize)
	{
		SrcSize = batchSize;
		TgtSize = batchSize;
		MatchSize = batchSize * (NTrial + 1);
		BasicUtil::MatchSampling(batchSize, NTrial, SrcIdx->HostMem, TgtIdx->HostMem, MatchInfo->HostMem);
		BasicUtil::InverseMatchIdx(SrcIdx->HostMem, MatchSize, Src2MatchIdx->HostMem, Src2MatchElement->HostMem, SrcSize);
		BasicUtil::InverseMatchIdx(TgtIdx->HostMem, MatchSize, Tgt2MatchIdx->HostMem, Tgt2MatchElement->HostMem, TgtSize);

		SrcIdx->SyncFromHost(0, MatchSize);
		TgtIdx->SyncFromHost(0, MatchSize);
		MatchInfo->SyncFromHost(0, MatchSize);

		Src2MatchIdx->SyncFromHost(0, SrcSize);
		Src2MatchElement->SyncFromHost(0, MatchSize);

		Tgt2MatchIdx->SyncFromHost(0, TgtSize);
		Tgt2MatchElement->SyncFromHost(0, MatchSize);
	}

	void Init()
	{
		SrcIdx = new PieceMem<int>(Stat->MAX_MATCH_BATCHSIZE, Device);
		TgtIdx = new PieceMem<int>(Stat->MAX_MATCH_BATCHSIZE, Device);
		MatchInfo = new PieceMem<float>(Stat->MAX_MATCH_BATCHSIZE, Device);

		Src2MatchIdx = new PieceMem<int>(Stat->MAX_SRC_BATCHSIZE, Device);
		Src2MatchElement = new PieceMem<int>(Stat->MAX_MATCH_BATCHSIZE, Device);

		Tgt2MatchIdx = new PieceMem<int>(Stat->MAX_TGT_BATCHSIZE, Device);
		Tgt2MatchElement = new PieceMem<int>(Stat->MAX_MATCH_BATCHSIZE, Device);
	}
};

#endif