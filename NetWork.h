#pragma once
#include "ActivateFunction.h"
#include "Matrix.h"
#include <fstream>
using namespace std;
struct data_NetWork {
	int L;
	int* size;
};
class NetWork
{
	int L;    // Layers - Слои
	int* size;
	ActivateFunction actFunc;
	Matrix* weights;
	double** bios;
	double** neurons_val, ** neurons_err;
	double* neurons_bios_val;
public:
	void Init(data_NetWork data);
	void PrintConfig();
	void SetInput(double* values);

	double ForwardFeed();
	int SearchMaxIndex(double* value);
	void PrintValues(int L);

	void BackPropogation(double expect);
	void WeightsUpdater(double lr);

	void SaveWeights();
	void ReadWeights();
};

