#include "ActivateFunction.h"

void ActivateFunction::set()    // Выбор метода обучения
{
	std::cout << "Пожалуйста, выберите метод обучения\n(рекомендуется метод ReLU, т.к. его реализация проще => быстрее и, как показывает практика, он более эфективен)\n1 - sigmoid \n2 - ReLU \n3 - th(x) \n";
	int tmp;
	std::cin >> tmp;
	switch (tmp)
	{
	case sigmoid:
		actFunc = sigmoid;
		break;
	case ReLU:
		actFunc = ReLU;
		break;
	case thx:
		actFunc = thx;
		break;
	default:
		throw std::runtime_error("Ошибка чтения");
		break;
	}
}

void ActivateFunction::use(double* value, int n)    // Использование методов обучения, с помощью их функций
{
	switch (actFunc)
	{
	case activateFunc::sigmoid:
		for (int i = 0; i < n; i++)
			value[i] = 1 / (1 + exp(-value[i]));
		break;
	case activateFunc::ReLU:
		for (int i = 0; i < n; i++)
		{
			if (value[i] < 0)
				value[i] *= 0.01;
			else if (value[i] > 1)
				value[i] = 1. + 0.01 * (value[i] - 1.);
			//else value = value;
		}
		break;

	case activateFunc::thx:
		for (int i = 0; i < n; i++) {
			if (value[i] < 0)
				value[i] = 0.01 * (exp(value[i]) - exp(-value[i])) / (exp(value[i]) + exp(-value[i]));
			else
				value[i] = (exp(value[i]) - exp(-value[i])) / (exp(value[i]) + exp(-value[i]));
		}
		break;
	default:
		throw std::runtime_error("Ошибка в функции use \n");
		break;
	}
}

void ActivateFunction::useDer(double* value, int n)    // Формулы с производными
{
	switch (actFunc)
	{
	case activateFunc::sigmoid:
		for (int i = 0; i < n; i++)
			value[i] = value[i] * (1 - value[i]);
		break;
	case activateFunc::ReLU:
		for (int i = 0; i < n; i++)
		{
			if (value[i] < 0 || value[i] > 1)
				value[i] = 0.01;
			else
				value[i] = 1;
		}
		break;
	case activateFunc::thx:
		for (int i = 0; i < n; i++)
		{
			if (value[i] < 0)
				value[i] = 0.01 * (1 - value[i] * value[i]);
			else
				value[i] = 1 - value[i] * value[i];
		}
		break;
	default:
		throw std::runtime_error("Ошибка в функции useDer 1/2 \n");
		break;
	}
}

double ActivateFunction::useDer(double value)    // Перегруженный useDer с значением на входе/выходе - double
{
	switch (actFunc)
	{
	case activateFunc::sigmoid:
		value = 1 / (1 + exp(-value));
		break;
	case activateFunc::ReLU:
		if (value < 0 || value > 1)
			value = 0.01;
		break;
	case activateFunc::thx:
		if (value < 0)
			value = 0.01 * (exp(value) - exp(-value)) / (exp(value) + exp(-value));
		else
			value = (exp(value) - exp(-value)) / (exp(value) + exp(-value));
		break;

	default:
		throw std::runtime_error("Ошибка в функции useDer 2/2 \n");
		break;
	}
	return value;
}