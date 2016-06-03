#ifndef RUNNERBEHAVIOR_H
#define RUNNERBEHAVIOR_H
#include <iostream>
#include <vector>
#include <map>
#include <stdio.h>
#include <cstring>
#include <cstdio>

#include "mathOperationManager.h"
#include "PieceMem.h"

using namespace std;

enum RUNMODE { RUNMODE_TRAIN, RUNMODE_PREDICT };

class RunnerBehavior
{
public:
	RUNMODE RunMode;
	DEVICE Device;
	IMathOperationManager * ComputeLib;
};
#endif