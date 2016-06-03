//vocab.h
#ifndef VOCAB_H
#define VOCAB_H
#include <iostream>
#include <vector>
#include <map>
#include <tuple>

#define CHARACTER_NUM 256
#define MAX_TOKEN_NUM 64 //max number of words per sentence. The remaining words will be ignored.
#define MAX_TOKEN_LEN 64 //max number of letters per word. The remaining letters will be ignored.

using namespace std;

class Vocab
{
public:
	int VocabSize;
	int LetterNum;

    /*
    Load letter-trigram vocab from a text file
    */
    int LoadVocab(const char * vocabFile);

    /* 
    extract feature from word.
    */
    int FeatureExtract(const char * token, int * fea, int index ) const;
    
    /*
    extract feature from sentence.
    */
    int FeatureExtract(const char * tokens[], const int tokenNum, int * seg, int * fea) const; 


    /*
    Convert text into a sequence of vectors of letter-trigram features.
    It will do normalization internally.
    
	Returns: a vector of sparse features of each slicing convolutional window,
		each is a tuple<int*, float*, int>, represents sparse feature Id arrays,
		sparse feature value arrays, and the length of sparse features Id/value arrays.
		sparse feature Id array and sparse feature value array have the same length.
		
    For examples please see int DSSMDecoder::FProp().
    */
	
private:
	vector<unsigned short> IndexTable;
    unsigned char LetterIndex[CHARACTER_NUM];
};

#endif