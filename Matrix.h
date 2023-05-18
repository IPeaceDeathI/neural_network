#pragma once
#include <iostream>
class Matrix
{
	double** matrix;
	int row, col;
public:
	void Init(int row, int col);    // Функция инициализации матрицы
	void Rand();    // Функция заполнения матрицы рандомными числами
	static void Multi(const Matrix& m, const double* b, int n, double* c);    // Перемножение обычной матрицы и вектора-столбца
	static void Multi_T(const Matrix& m, const double* b, int n, double* c);    // Перемножение транспонированной матрицы и вектора-столбца
	static void SumVector(double* a, const double* b, int n);    // Сложение элементов вектора
	double& operator ()(int i, int j);    // Перегрузка оператора "()"
	friend std::ostream& operator << (std::ostream& os, const Matrix& m);    // Перегрузка оператора потока вывода
	friend std::istream& operator >> (std::istream& is, Matrix& m);    // Перегрузка оператора потока ввода
};