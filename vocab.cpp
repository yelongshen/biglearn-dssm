#include "vocab.h"

#include <fstream>
#include <sstream>
#include <map>
#include <locale>
#include <stdio.h>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>
#include <tuple>
#include <math.h>
using namespace std;

int Vocab::FeatureExtract(const char * token, int * fea, int index ) const
{
    int tokenLen = (int)strlen(token);
    int rawFeaNum = tokenLen < MAX_TOKEN_LEN ? tokenLen : MAX_TOKEN_LEN;

    //int * validLetterID, a global array for this, with a length of MAX_TOKEN_LEN+2 !!! -- only for trigram convlutional DSSM!!
    int mValidLetterID[MAX_TOKEN_LEN+2];
    int validLetterIdx = 0;
    mValidLetterID[validLetterIdx++] = LetterIndex['#']; //id of letter boundary //currently we mixed # and BOUND. would like to seperate them in the future
    for(int i = 0; i < rawFeaNum; i++)
    {
        if(LetterIndex[token[i]] < LetterNum) mValidLetterID[validLetterIdx++] = LetterIndex[token[i]];
    }
    if(validLetterIdx == 1) mValidLetterID[validLetterIdx++] = LetterIndex['#']; //if the word is empty, filled with "#"
    mValidLetterID[validLetterIdx++] = LetterIndex['#']; //add the ending BOUND
    
    int feaNum = 0;
    for(int i = 1; i < validLetterIdx - 1; i++)
    {
        int feaIndex = mValidLetterID[i-1] * LetterNum * LetterNum + mValidLetterID[i] * LetterNum + mValidLetterID[i+1];

        if(IndexTable[feaIndex] < 65535)
        {
            fea[index] = IndexTable[feaIndex];
            index += 1;
            feaNum += 1;
        }
    }
    return feaNum;
}

int Vocab::FeatureExtract(const char * tokens[], const int tokenNum, int * seg, int * fea) const
{
    int segNum = tokenNum < MAX_TOKEN_NUM ? tokenNum : MAX_TOKEN_NUM;
    int feaIndex = 0;
    int segIndex = 0;

    for(int i=0;i<segNum;i++)
    {
        int feaNum = FeatureExtract(tokens[i], fea, feaIndex);
        if(feaNum > 0)
        {
            feaIndex += feaNum;
            seg[segIndex] = feaIndex;
            segIndex += 1;
        }
    }
    return segIndex;
}

int Vocab::LoadVocab(const char * vocabFile)
{
    ifstream vocabStream;
    vocabStream.open(vocabFile);
    if (vocabStream.is_open())
    {
        for (int i = 0; i<CHARACTER_NUM; i++) LetterIndex[i] = 255;

        int cnt = 0;
        for (int i = 'a'; i <= 'z'; ++i)  LetterIndex[i] = cnt++;
        for (int i = '0'; i <= '9'; ++i)  LetterIndex[i] = cnt++;
        LetterIndex['#'] = cnt++;
        
        int avail_cn_check = 37;
        int ngram_order_check = 3;

        LetterNum = avail_cn_check;

        int letterngram_num = LetterNum * LetterNum * LetterNum;
        //(int)pow((double)avail_cn_check, ngram_order_check);

        IndexTable.resize(letterngram_num, 65535);
        int i = 0;
        string line;
        while (getline(vocabStream, line))
        {
            int offset = 0;
            //// skip nonalpha-num chars, find the first # or 0-9a-z char
            while (line[offset] != '\0' && !(line[offset] == '#' || ('0' <= line[offset] && line[offset] <= '9') || ('a' <= line[offset] && line[offset] <= 'z')))
            {
                ++offset;
            }
            if (line[offset] == '\0')
            {
                return 0;
            }
            int feaNum = LetterIndex[line[offset]] * LetterNum * LetterNum + LetterIndex[line[offset+1]] * LetterNum + LetterIndex[line[offset+2]];
            IndexTable[feaNum] = i;
            ++i;
        }
        vocabStream.close();
        VocabSize = i;
        return 1;
    }
    else
    {
        return 0;
    }
}

