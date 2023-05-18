#pragma once
#include <iostream>
enum activateFunc { sigmoid = 1, ReLU, thx };    // Перечисление (sigmoid = 1, ReLu = 2, thx = 3)
class ActivateFunction
{
	activateFunc actFunc;
public:
	void set();
	void use(double* value, int n);
	void useDer(double* value, int n);
	double useDer(double value);    // Перегрузка функции useDer
};

