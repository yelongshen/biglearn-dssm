//util.h

#ifndef UTIL_H
#define UTIL_H

#include <stdio.h>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <tuple>
#include <cstdlib>
#include <time.h>

using namespace std;

class BasicUtil
{
public:
	static char** TokenizeString(string sentence, int & token_num, int max_token_num, int max_token_len);
	static void MatchSampling(int batchSize, int nTrail, int* srcIdx, int* tgtIdx, float* matchLabel);
	static void InverseMatchIdx(int* matchIdx, int matchNum, int* inverseSmpIdx, int* inverseElementIdx, int inverseSmpNum);
};

#endif