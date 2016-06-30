//util.cpp

#include <vector>
#include "util.h"

char** BasicUtil::TokenizeString(string sentence, int & token_num, int max_token_num, int max_token_len)
{
	char** tokens = (char **)malloc(sizeof(char *)* max_token_num);
	for (int i = 0; i<max_token_num; i++) tokens[i] = new char[max_token_len];

	string token;
	istringstream sen_tokens(sentence);
	token_num = 0;
	while (getline(sen_tokens, token, ' '))
	{
		strncpy(tokens[token_num], token.c_str(), max_token_len);
		if (++token_num >= max_token_num) break;
	}
	return tokens;
}

void BasicUtil::MatchSampling(int batchSize, int nTrail, int* srcIdx, int* tgtIdx, float* matchLabel)
{
	for (int k = 0; k < batchSize; k++)
	{
		srcIdx[k * (nTrail + 1)] = k;
		for (int i = 0; i < nTrail; i++) srcIdx[k * (nTrail + 1) + i + 1] = k;
		tgtIdx[k * (nTrail + 1)] = k;
		matchLabel[k * (nTrail + 1)] = 1;
	}

	for (int i = 0; i < nTrail; i++)
	{
		float range = 0.8 * batchSize - 0.1 * batchSize;
		int randpos = (int)( ((float)rand() / RAND_MAX) * range + 0.1 * batchSize );

		for (int k = 0; k < batchSize; k++)
		{
			int bs = (randpos + k) % batchSize;
			tgtIdx[k * (nTrail + 1) + i + 1] = bs;
			matchLabel[k * (nTrail + 1) + i + 1] = 0;
		}
	}
}

void BasicUtil::InverseMatchIdx(int* matchIdx, int matchNum, int* inverseSmpIdx, int* inverseElementIdx, int inverseSmpNum)
{
	vector<vector<int>> indexMatchList(inverseSmpNum, vector<int>());

	for (int i = 0; i < matchNum; i++) indexMatchList[matchIdx[i]].push_back(i);

	int element = 0;
	for (int i = 0; i < inverseSmpNum; i++)
	{
		inverseSmpIdx[i] = element + indexMatchList[i].size();

		for (vector<int>::iterator it = indexMatchList[i].begin(); it != indexMatchList[i].end(); ++it) 
			inverseElementIdx[element++] = *it;
	}
}



void BasicUtil::ReadInt(ifstream & mstream, int *integer)
{
	mstream.read((char *)integer,sizeof(int));
}


void BasicUtil::ReadFloat(ifstream & mstream, float *float_value)
{
	mstream.read((char *)float_value,sizeof(float));
}


void BasicUtil::WriteInt(ofstream &mstream, int * integer)
{
	mstream.write((char *)integer,sizeof(int));
}

void BasicUtil::WriteFloat(ofstream &mstream, float * float_value)
{
	mstream.write((char *)float_value,sizeof(float));
}
