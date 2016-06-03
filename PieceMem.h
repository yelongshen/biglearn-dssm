//CudaPiece
#ifndef PIECEMEM_H
#define PIECEMEM_H
#include <iostream>
#include <vector>
#include <map>
#include <tuple>
#include <stdio.h>
#include <cstring>
#include <cstdio>
#include <cusparse.h>
#include <cuda_runtime.h> 
#include <cublas.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_surface_types.h>
#include <cublas_v2.h>

#pragma comment(lib, "cudart") 
#pragma comment(lib, "cuda") 
#pragma comment(lib, "cudnn")

using namespace std;

enum DEVICE { DEVICE_CPU, DEVICE_GPU };

template <class T> 
class PieceMem
{
public:
	int Size;
	DEVICE Device;
	T * HostMem;
	T * Mem;

	vector<T> * Tmp;

	PieceMem(int size, DEVICE device)
	{
		Size = size;
		Device = device;
		if(Size > 0)
		{
			switch(Device)
			{
				case DEVICE_CPU: Mem = (T*)malloc(sizeof(T)* Size); HostMem = Mem; break;
				case DEVICE_GPU: cudaMalloc((void **)&Mem, Size * sizeof(T)); HostMem = (T*)malloc(sizeof(T)* Size); break;
			}
		}
	}

	~PieceMem()
	{
		switch(Device)
		{
			case DEVICE_CPU : free(Mem); break;
			case DEVICE_GPU : cudaFree(Mem); break;
		}
	}

	PieceMem(PieceMem<T> * ref, int offset, int size)
	{
		Size = size;
		Device = ref->Device;
		Mem = ref->Mem + offset;
		HostMem = ref->HostMem + offset;
	}

	void Resize(int size)
	{
		int miniSize = Size > size ? size : Size;
		if(size <= 0) return;
		T* newMem = NULL;
		switch(Device)
		{
			case DEVICE_CPU : 
					newMem = (T*) malloc(sizeof(T) * size);
					memset(newMem, 0, sizeof(T) * size);
					memcpy(newMem, Mem, sizeof(T) * miniSize);
					free(Mem);
					HostMem = newMem;
					break;
			case DEVICE_GPU : 
					cudaMalloc((void **)&newMem, sizeof(T) * size);
					cudaMemset(newMem, 0, size * sizeof(T)); 
					cudaMemcpy(newMem, Mem, miniSize * sizeof(T), cudaMemcpyDeviceToDevice);
					cudaFree(Mem);

					free(HostMem);
					HostMem = (T*)malloc(sizeof(T)* size);
					break;
		}
		Mem = newMem;
		Size = size;
	}

	void SyncToHost(int offset, T * host, int size)
	{
		switch(Device)
		{
			case DEVICE_GPU: cudaMemcpy(host, Mem + offset, size * sizeof(T), cudaMemcpyDeviceToHost); break;
			case DEVICE_CPU: memcpy ( host, Mem + offset, size * sizeof(T)); break;
		}
	}

	void SyncToHost(int offset, int size)
	{
		switch (Device)
		{
			case DEVICE_GPU: cudaMemcpy(HostMem + offset, Mem + offset, size * sizeof(T), cudaMemcpyDeviceToHost); break;
		}
	}

	void SyncFromHost(int offset, T * host, int size)
	{
		switch(Device)
		{
			case DEVICE_GPU: cudaMemcpy(Mem + offset, host , size * sizeof(T), cudaMemcpyHostToDevice); break;
			case DEVICE_CPU: memcpy (Mem + offset, host, size * sizeof(T)); break;
		}
	}

	void SyncFromHost(int offset, int size)
	{
		switch (Device)
		{
			case DEVICE_GPU: cudaMemcpy(Mem + offset, HostMem + offset, size * sizeof(T), cudaMemcpyHostToDevice); break;
		}
	}

	void CopyFrom(const PieceMem & refPiece)
	{
		CopyFrom(0, refPiece, 0, Size);
	}

	void CopyFrom(int offset, const PieceMem & refPiece, int refOffset, int size)
    {
    	// GPU <-- GPU.
        if (Device == DEVICE_GPU && refPiece.Device == DEVICE_GPU)
        	cudaMemcpy(Mem + offset, refPiece.Mem + refOffset, size * sizeof(T), cudaMemcpyDeviceToDevice);

        // GPU <-- CPU.
        if(Device == DEVICE_GPU && refPiece.Device == DEVICE_CPU)
        	cudaMemcpy(Mem + offset, refPiece.Mem + refOffset , size * sizeof(T), cudaMemcpyHostToDevice);

        // CPU <-- GPU.
        if(Device == DEVICE_CPU && refPiece.Device == DEVICE_GPU)
        	cudaMemcpy(Mem + offset, refPiece.Mem + refOffset, size * sizeof(T), cudaMemcpyDeviceToHost);

        // CPU <-- CPU.
        if(Device == DEVICE_CPU && refPiece.Device == DEVICE_CPU)
        	memcpy ( Mem + offset, refPiece.Mem + refOffset, size * sizeof(T));
    }

	void QuickWatch()
	{
		if(Tmp != NULL) free(Tmp);
		Tmp = new vector<T>(Size);
		SyncToHost(0, Size);
		for(int i=0;i<Size;i++) (*Tmp)[i] = HostMem[i];
	}

	void Zero()
	{
		switch(Device)
		{
			case DEVICE_GPU: cudaMemset(Mem, 0, Size * sizeof(T)); break;
			case DEVICE_CPU: memset(Mem, 0, Size * sizeof(T)); break;
		}
	}
private:
	

};

#endif