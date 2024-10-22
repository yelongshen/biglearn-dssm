#include <stdio.h>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <tuple>
#include <cstdlib>
#include <time.h>
#include <sys/time.h>

#include "util.h"
#include "vocab.h"

#include "PieceMem.h"
#include "sparseMatrixData.h"
#include "denseMatrixData.h"
#include "fullyConnectedLayer.h"
#include "cudaOperationManager.h"
#include "BiMatchData.h"
#include "similarityRunner.h"
#include "hiddenDenseMatrixData.h"

using namespace std;

void extractBinaryfromStream(const char * inputStream, Vocab & textHash,
		vector < tuple <int *, int > > & src_batch, vector < tuple <int *, int > > & tgt_batch, int isFilter, int debugLines)
{
	ifstream infile;
	infile.open(inputStream, ifstream::in);
	string line;
	int lineIdx = 0;
	while (getline(infile, line))
	{
		stringstream linestream(line);
		string src, tgt;
		getline(linestream, src, '\t');
		getline(linestream, tgt, '\t');

		int src_token_num = 0;
		int tgt_token_num = 0;
		char** src_tokens = BasicUtil::TokenizeString(src, src_token_num, MAX_TOKEN_NUM, MAX_TOKEN_LEN);
		char** tgt_tokens = BasicUtil::TokenizeString(tgt, tgt_token_num, MAX_TOKEN_NUM, MAX_TOKEN_LEN);

		int * src_fea = new int[MAX_TOKEN_LEN * MAX_TOKEN_NUM];
		int * src_seg = new int[MAX_TOKEN_NUM];

		int * tgt_fea = new int[MAX_TOKEN_LEN * MAX_TOKEN_NUM];
		int * tgt_seg = new int[MAX_TOKEN_NUM];

		int src_seg_num = textHash.FeatureExtract((const char **)src_tokens, src_token_num, src_seg, src_fea);
		int tgt_seg_num = textHash.FeatureExtract((const char **)tgt_tokens, tgt_token_num, tgt_seg, tgt_fea);
		
		int src_feature_num = 0; //src_seg[src_seg_num - 1];
		int tgt_feature_num = 0; //tgt_seg[tgt_seg_num - 1];
		
		if(src_seg_num >= 1)
		{
		    src_feature_num = src_seg[src_seg_num - 1];
		}
		
		if(tgt_seg_num >= 1)
		{
		    tgt_feature_num = tgt_seg[tgt_seg_num - 1];
		}
		
		if(isFilter == 1)
		{
		    if(src_feature_num <= 0 || tgt_feature_num <= 0) continue;
		}
		
		src_batch.push_back(tuple<int*, int>(src_fea, src_feature_num));
		tgt_batch.push_back(tuple<int*, int>(tgt_fea, tgt_feature_num));

		lineIdx += 1;
		if(lineIdx == debugLines) break;
	}
}

void ModelTrain()
{
	Vocab vocab;
	vocab.LoadVocab("l3g.txt");
	cout << "vocab Size " << vocab.VocabSize << endl;
	vector < tuple <int *, int > > src_batch, tgt_batch;
	extractBinaryfromStream("data//train_data_40k.tsv", vocab, src_batch, tgt_batch, 1, 0);

	int sampleSize = src_batch.size();
	cout << "train sample size" << sampleSize << endl;

	int iteration = 30;
	int miniBatchSize = 1024;
	int featureDim = vocab.VocabSize;
	int batchNum = sampleSize / miniBatchSize;
	int nTrial = 4;

	vector <int> shuff(sampleSize);

	RunnerBehavior rb;
	rb.RunMode = RUNMODE_TRAIN;
	rb.Device = DEVICE_GPU;
	cout<<"init cuda computation ...."<<endl;
	rb.ComputeLib = new CudaOperationManager(true, true);
	
	cout<<"init cuda computation done"<<endl;
	
	int hiddenDim1 = 128;
	int hiddenDim2 = 128;

	SparseIndexMatrixStat srcMiniBatchInfo;
	srcMiniBatchInfo.MAX_ROW_SIZE = miniBatchSize;
	srcMiniBatchInfo.MAX_COL_SIZE = featureDim;
	srcMiniBatchInfo.TOTAL_BATCH_NUM = batchNum;
	srcMiniBatchInfo.TOTAL_SAMPLE_NUM = sampleSize;
	srcMiniBatchInfo.MAX_ELEMENT_SIZE = miniBatchSize * 256;

	SparseIndexMatrixStat tgtMiniBatchInfo;
	tgtMiniBatchInfo.MAX_ROW_SIZE = miniBatchSize;
	tgtMiniBatchInfo.MAX_COL_SIZE = featureDim;
	tgtMiniBatchInfo.TOTAL_BATCH_NUM = batchNum;
	tgtMiniBatchInfo.TOTAL_SAMPLE_NUM = sampleSize;
	tgtMiniBatchInfo.MAX_ELEMENT_SIZE = miniBatchSize * 256;

	DenseMatrixStat OutputLayer1Info;
	OutputLayer1Info.MAX_ROW_SIZE = miniBatchSize;
	OutputLayer1Info.MAX_COL_SIZE = hiddenDim1;
	OutputLayer1Info.TOTAL_BATCH_NUM = batchNum;
	OutputLayer1Info.TOTAL_SAMPLE_NUM = sampleSize;


	DenseMatrixStat OutputLayer2Info;
	OutputLayer2Info.MAX_ROW_SIZE = miniBatchSize;
	OutputLayer2Info.MAX_COL_SIZE = hiddenDim2;
	OutputLayer2Info.TOTAL_BATCH_NUM = batchNum;
	OutputLayer2Info.TOTAL_SAMPLE_NUM = sampleSize;


	FullyConnectedLayer srcLayer1(featureDim, hiddenDim1, &rb);
	FullyConnectedLayer srcLayer2(hiddenDim1, hiddenDim2, &rb);

	FullyConnectedLayer tgtLayer1(featureDim, hiddenDim1, &rb);
	FullyConnectedLayer tgtLayer2(hiddenDim1, hiddenDim2, &rb);

	DenseMatrixStat OutputSimInfo;
	OutputSimInfo.MAX_ROW_SIZE = miniBatchSize;
	OutputSimInfo.MAX_COL_SIZE = 1 + nTrial;
	OutputSimInfo.TOTAL_BATCH_NUM = batchNum;
	OutputSimInfo.TOTAL_SAMPLE_NUM = sampleSize;

	SparseIndexMatrix srcBatch(&srcMiniBatchInfo, rb.Device);	
	HiddenDenseMatrix srcLayer1Data(&OutputLayer1Info, rb.Device);
	HiddenDenseMatrix srcLayer2Data(&OutputLayer2Info, rb.Device);

	SparseIndexMatrix tgtBatch(&tgtMiniBatchInfo, rb.Device);
	HiddenDenseMatrix tgtLayer1Data(&OutputLayer1Info, rb.Device);
	HiddenDenseMatrix tgtLayer2Data(&OutputLayer2Info, rb.Device);

	BiMatchData biMatchData(miniBatchSize, nTrial, rb.Device);

	SimilarityRunner similarityRunner(10, &rb);
	HiddenDenseMatrix simOutput(&OutputSimInfo, rb.Device);
	HiddenDenseMatrix probOutput(&OutputSimInfo, rb.Device);

	probOutput.Deriv->Data->Zero();
	
	//iteration = 1;
	cout<<"start training iteration"<<endl;
	
	double train_time = 0;
	double io_time = 0;
		
	struct timeval train_start, train_end;
	struct timeval io_start, io_end;
	
	gettimeofday(&train_start, 0);
	
	for (int iter = 0; iter<iteration; iter++)
	{
		for (int i = 0; i<sampleSize; i++) shuff[i] = i;

		int shuffIdx = 0;

		float avgLoss = 0;
		for (int b = 0; b<batchNum; b++)
		{
			gettimeofday(&io_start, 0);

			srcBatch.Refresh();
			tgtBatch.Refresh();

			while (shuffIdx < sampleSize - 1 && srcBatch.RowSize < miniBatchSize && tgtBatch.RowSize < miniBatchSize)
			{
				int p = shuffIdx + rand() % (sampleSize - shuffIdx);
				int smpIdx = shuff[p];
				shuff[p] = shuff[shuffIdx];
				shuff[shuffIdx] = smpIdx;
				shuffIdx += 1;

				srcBatch.PushSample(get<0>(src_batch[smpIdx]), get<1>(src_batch[smpIdx]));
				tgtBatch.PushSample(get<0>(tgt_batch[smpIdx]), get<1>(tgt_batch[smpIdx]));
			}
			
			gettimeofday(&io_end, 0);
			
			io_time += io_end.tv_sec - io_start.tv_sec;
			
			
			//cout<<"src batch row "<< srcBatch.RowSize<<endl;
			//cout<<"src element size " <<srcBatch.ElementSize<<endl; 
			//cout<<"tgt batch row "<< tgtBatch.RowSize<<endl;
			//cout<<"tgt element size " <<tgtBatch.ElementSize<<endl; 
			
			//srcLayer1.Weight->SyncToHost(0, 100);
			//tgtLayer1.Weight->SyncToHost(0, 100);
			
			//for(int i=0;i<100;i++)
			//{
			 //   cout<<"smpIdx "<< src.Weight->HostMem[i]<<endl;
			//}
			
			
			
			//cout<<"src weight "<<srcLayer1.Weight->HostMem[0]<<endl;
			//cout<<"tgt weight "<<tgtLayer1.Weight->HostMem[0]<<endl;
			
			//for(int i = 0; i< srcBatch.ElementSize; i++)
			//{
			//	srcBatch.SampleIdx
			//}
			//if( cudaSuccess != cudaGetLastError())
			//	cout <<"error 1"<<endl;
			
			
			srcLayer1.Forward(&srcBatch, srcLayer1Data.Output);
			//if( cudaSuccess != cudaGetLastError())
			//	cout <<"fdsfasdf"<<endl;
			//srcLayer1Data.Output->Data->SyncToHost(0,100);
			//cout<<"src 1 output"<<srcLayer1Data.Output->Data->HostMem[0]<<endl;
			
			srcLayer2.Forward(srcLayer1Data.Output, srcLayer2Data.Output);

			tgtLayer1.Forward(&tgtBatch, tgtLayer1Data.Output);
			tgtLayer2.Forward(tgtLayer1Data.Output, tgtLayer2Data.Output);
			
			biMatchData.GenerateMatch(srcBatch.RowSize);
			
			//srcLayer2Data.Output->Data->SyncToHost(0, srcLayer2Data.Stat->MAX_COL_SIZE * srcBatch.RowSize);
			//tgtLayer2Data.Output->Data->SyncToHost(0, tgtLayer2Data.Stat->MAX_COL_SIZE * tgtBatch.RowSize);
			
			//cout<<"src output"<<srcLayer2Data.Output->Data->HostMem[0]<<endl;
			//cout<<"tgt output"<<tgtLayer2Data.Output->Data->HostMem[0]<<endl;
			
			similarityRunner.Forward(srcLayer2Data.Output, tgtLayer2Data.Output, &biMatchData, simOutput.Output);

			//simOutput.Output->Data->SyncToHost(0, srcBatch.RowSize * 5);
			//for(int i=0;i<srcBatch.RowSize;i++)
			//{
			//	cout<<"sim"<< simOutput.Output->Data->HostMem[i]<<endl;
			//	break;
			//}
			//break;				
			rb.ComputeLib->SoftmaxForward(simOutput.Output->Data, probOutput.Output->Data, srcBatch.RowSize, simOutput.Stat->MAX_COL_SIZE);
			/// log softmax backward.  probOutput.Deriv->Data  --> biMatchData.MatchInfo
			rb.ComputeLib->VecAdd(probOutput.Output->Data, -1, biMatchData.MatchInfo, 1, simOutput.Deriv->Data, 0, biMatchData.MatchSize);

			//rb.ComputeLib->SoftmaxBackward(probOutput.Output->Data, probOutput.Deriv->Data, simOutput.Deriv->Data, srcBatch.RowSize, probOutput.Stat->MAX_COL_SIZE);
			/// output Loss.
			float loss = 0;
			//simOutput.Output->Data->QuickWatch();
			//simOutput.Deriv->Data->QuickWatch();
			probOutput.Output->Data->SyncToHost(0, srcBatch.RowSize * probOutput.Stat->MAX_COL_SIZE); //  ->QuickWatch();
			//probOutput.Deriv->Data->QuickWatch();
			for(int i=0;i< srcBatch.RowSize; i++)
			{
				//cout<< probOutput.Output->Data->HostMem[i * probOutput.Stat->MAX_COL_SIZE]<<endl;

				loss += logf(probOutput.Output->Data->HostMem[i * probOutput.Stat->MAX_COL_SIZE] + LARGEEPS);
			}
			loss = loss / srcBatch.RowSize;
			avgLoss = b * 1.0f / (b + 1) * avgLoss + 1.0f / (b + 1) * loss;

			if((b+1) % 10 == 0) cout<<"mini batch : "<<b+1<<"\t avg loss :"<<avgLoss<<endl;
			//cout<<"current loss "<<loss<<endl;
			similarityRunner.Backward(simOutput.Deriv, srcLayer2Data.Deriv, tgtLayer2Data.Deriv);


			tgtLayer2.Backward(tgtLayer2Data.Deriv, tgtLayer2Data.Output, tgtLayer1Data.Deriv);
			tgtLayer1.Backward(tgtLayer1Data.Deriv, tgtLayer1Data.Output);

			srcLayer2.Backward(srcLayer2Data.Deriv, srcLayer2Data.Output, srcLayer1Data.Deriv);
			srcLayer1.Backward(srcLayer1Data.Deriv, srcLayer1Data.Output);

			/// update.
			tgtLayer2.Update(tgtLayer2Data.Deriv, tgtLayer1Data.Output);
			tgtLayer1.Update(tgtLayer1Data.Deriv, &tgtBatch);

			srcLayer2.Update(srcLayer2Data.Deriv, srcLayer1Data.Output);
			srcLayer1.Update(srcLayer1Data.Deriv, &srcBatch);
		}
		cout<<"iteration : "<<iter + 1<<"\t avg loss :"<<avgLoss<<endl;

	}
	
	gettimeofday(&train_end, 0);
	
	train_time = (train_end.tv_sec - train_start.tv_sec);
			
	cout<<"train overall time elipsed (sec):"<<train_time<<endl;
	cout<<"io time elipsed (sec):"<<io_time<<endl;
	cout<<"gpu time elipsed (sec):"<<train_time - io_time<<endl;
	ofstream modelWriter;
	modelWriter.open("model//dssm.v2.model", ofstream::binary);
	srcLayer1.Serialize(modelWriter);
	srcLayer2.Serialize(modelWriter);
	tgtLayer1.Serialize(modelWriter);
	tgtLayer2.Serialize(modelWriter);
	modelWriter.close();
}

void ModelPredict()
{
	Vocab vocab;
	vocab.LoadVocab("l3g.txt");
	cout << "vocab Size " << vocab.VocabSize << endl;
	vector < tuple <int *, int > > src_batch, tgt_batch;
	extractBinaryfromStream("data//test_data_clean.tsv", vocab, src_batch, tgt_batch, 0, 0);

	int sampleSize = src_batch.size();
	cout << "test sample size" << sampleSize << endl;

	int miniBatchSize = 1024;
	int featureDim = vocab.VocabSize;
	int batchNum = (sampleSize - 1) / miniBatchSize + 1;

	RunnerBehavior rb;
	rb.RunMode = RUNMODE_PREDICT;
	rb.Device = DEVICE_GPU;

	rb.ComputeLib = new CudaOperationManager(true, true);
	int hiddenDim1 = 128;
	int hiddenDim2 = 128;

	SparseIndexMatrixStat srcMiniBatchInfo;
	srcMiniBatchInfo.MAX_ROW_SIZE = miniBatchSize;
	srcMiniBatchInfo.MAX_COL_SIZE = featureDim;
	srcMiniBatchInfo.TOTAL_BATCH_NUM = batchNum;
	srcMiniBatchInfo.TOTAL_SAMPLE_NUM = sampleSize;
	srcMiniBatchInfo.MAX_ELEMENT_SIZE = miniBatchSize * 256;

	SparseIndexMatrixStat tgtMiniBatchInfo;
	tgtMiniBatchInfo.MAX_ROW_SIZE = miniBatchSize;
	tgtMiniBatchInfo.MAX_COL_SIZE = featureDim;
	tgtMiniBatchInfo.TOTAL_BATCH_NUM = batchNum;
	tgtMiniBatchInfo.TOTAL_SAMPLE_NUM = sampleSize;
	tgtMiniBatchInfo.MAX_ELEMENT_SIZE = miniBatchSize * 256;

	DenseMatrixStat OutputLayer1Info;
	OutputLayer1Info.MAX_ROW_SIZE = miniBatchSize;
	OutputLayer1Info.MAX_COL_SIZE = hiddenDim1;
	OutputLayer1Info.TOTAL_BATCH_NUM = batchNum;
	OutputLayer1Info.TOTAL_SAMPLE_NUM = sampleSize;

	DenseMatrixStat OutputLayer2Info;
	OutputLayer2Info.MAX_ROW_SIZE = miniBatchSize;
	OutputLayer2Info.MAX_COL_SIZE = hiddenDim2;
	OutputLayer2Info.TOTAL_BATCH_NUM = batchNum;
	OutputLayer2Info.TOTAL_SAMPLE_NUM = sampleSize;

	ifstream modelReader;
	modelReader.open("model//dssm.v2.model", ofstream::binary);
	FullyConnectedLayer srcLayer1(modelReader, &rb);
	FullyConnectedLayer srcLayer2(modelReader, &rb);
	FullyConnectedLayer tgtLayer1(modelReader, &rb);
	FullyConnectedLayer tgtLayer2(modelReader, &rb);
	modelReader.close();

	DenseMatrixStat OutputSimInfo;
	OutputSimInfo.MAX_ROW_SIZE = miniBatchSize;
	OutputSimInfo.MAX_COL_SIZE = 1;
	OutputSimInfo.TOTAL_BATCH_NUM = batchNum;
	OutputSimInfo.TOTAL_SAMPLE_NUM = sampleSize;

	SparseIndexMatrix srcBatch(&srcMiniBatchInfo, rb.Device);	
	HiddenDenseMatrix srcLayer1Data(&OutputLayer1Info, rb.Device);
	HiddenDenseMatrix srcLayer2Data(&OutputLayer2Info, rb.Device);

	SparseIndexMatrix tgtBatch(&tgtMiniBatchInfo, rb.Device);
	HiddenDenseMatrix tgtLayer1Data(&OutputLayer1Info, rb.Device);
	HiddenDenseMatrix tgtLayer2Data(&OutputLayer2Info, rb.Device);

	BiMatchData biMatchData(miniBatchSize, 0, rb.Device);

	SimilarityRunner similarityRunner(10, &rb);
	HiddenDenseMatrix simOutput(&OutputSimInfo, rb.Device);
	HiddenDenseMatrix probOutput(&OutputSimInfo, rb.Device);

	ofstream outfile;
	outfile.open("data//test_data.v2.result", ofstream::out);

	int smpIdx = 0;

	for (int b = 0; b<batchNum; b++)
	{
		srcBatch.Refresh();
		tgtBatch.Refresh();

		while (smpIdx < sampleSize && srcBatch.RowSize < miniBatchSize && tgtBatch.RowSize < miniBatchSize)
		{
			srcBatch.PushSample(get<0>(src_batch[smpIdx]), get<1>(src_batch[smpIdx]));
			tgtBatch.PushSample(get<0>(tgt_batch[smpIdx]), get<1>(tgt_batch[smpIdx]));
			smpIdx++;
		}

		srcLayer1.Forward(&srcBatch, srcLayer1Data.Output);
		srcLayer2.Forward(srcLayer1Data.Output, srcLayer2Data.Output);

		tgtLayer1.Forward(&tgtBatch, tgtLayer1Data.Output);
		tgtLayer2.Forward(tgtLayer1Data.Output, tgtLayer2Data.Output);

		biMatchData.GenerateMatch(srcBatch.RowSize);

		similarityRunner.Forward(srcLayer2Data.Output, tgtLayer2Data.Output, &biMatchData, simOutput.Output);

		simOutput.Output->Data->QuickWatch();

		//probOutput.Deriv->Data->QuickWatch();
		for(int i=0;i< srcBatch.RowSize; i++)
			outfile<<simOutput.Output->Data->HostMem[i]<<endl;
		//cout<<srcBatch.RowSize<<"\t"<<smpIdx<<endl;
		
		if((b+1) % 10 == 0) cout<<"mini batch : "<<b+1<<" sample number "<<smpIdx<<endl;
	}
	outfile.close();
}

void TestCuda()
{
	PieceMem<float> * a = new PieceMem<float>(100, DEVICE_GPU);
	PieceMem<float> * b = new PieceMem<float>(100, DEVICE_GPU);
	for(int i=0;i<100;i++)
	{
		a->HostMem[i] = 0; //i * 0.1f;
		b->HostMem[i] = 0;	
	}
	a->SyncFromHost(0, 100);
	b->SyncFromHost(0, 100);
	
	CudaOperationManager ComputeLib(true, true);
	
	ComputeLib.Scale(a, 0.1f, 100);
	
	ComputeLib.SoftmaxForward(a, b, 10, 10);
	a->SyncToHost(0, 100);
	b->SyncToHost(0, 100);
	for(int i=0;i<100;i++)
	{
		cout<<a->HostMem[i]<<endl;
		cout<<b->HostMem[i]<<endl;
	}
}

int main(int argc, char* argv[])
{
	//cout <<argc<<argv[0]<<endl;
	if(argc <= 1)
	{
	    cout<<"argv : -train : training; -predict : testing;"<<endl;
	    return 0;
	}
	if(strcmp(argv[1], "-train") == 0)
	{
	    //TestCuda();	
	    // DSSM Model Train.
		ModelTrain();
	}
	else if(strcmp(argv[1],"-predict") == 0)
	{
		// DSSM Model Predict.
		ModelPredict();
	}
	return 0;
}
