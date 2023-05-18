#include "NetWork.h"
#include <chrono>

struct data_info    // Структура для цифр (цифра и пиксели к ней)
{
    double* pixels;
    int digit;
};
data_NetWork ReadDataNetWork(string path)    // Чтение конфигурации нейросети
{
    data_NetWork data{};
    ifstream fin;
    fin.open(path);
    if (!fin.is_open())
    {
        cout << "Ошибка чтения файла " << path << endl;
        system("pause");
    }
    else
    {
        cout << path << " Загрузка...\n";
    }
    string tmp;
    int L;
    while (!fin.eof())
    {
        fin >> tmp;
        if (tmp == "NetWork")
        {
            fin >> L;
            data.L = L;
            data.size = new int[L];
            for (int i = 0; i < L; i++)
            {
                fin >> data.size[i];
            }
        }
    }
    fin.close();
    return data;
}
data_info* ReadData(string path, const data_NetWork& data_NW, int& examples)    // Считывание данных по цифрам из файла
{
    data_info* data;
    ifstream fin;
    fin.open(path);
    if (!fin.is_open())
    {
        cout << "Ошибка чтения файла " << path << endl;
        system("pause");
    }
    else
        cout << path << " Загрузка... \n";
    string tmp;
    fin >> tmp;
    if (tmp == "Examples")
    {
        fin >> examples;
        cout << "Примеры: " << examples << endl;
        data = new data_info[examples];
        for (int i = 0; i < examples; ++i)
            data[i].pixels = new double[data_NW.size[0]];

        for (int i = 0; i < examples; ++i)
        {
            fin >> data[i].digit;
            for (int j = 0; j < data_NW.size[0]; ++j)
            {
                fin >> data[i].pixels[j];
            }
        }
        fin.close();
        if (examples == 60048)
        {
            cout << "60000+ примеров для обучения MNIST загружены... \n";
        }
        return data;
    }
    else
    {
        cout << "Ошибка загрузки: " << path << endl;
        fin.close();
        return nullptr;
    }
}
int main()
{
    setlocale(0, "");
    NetWork NW{};
    data_NetWork NW_config;
    data_info* data;
    double ra = 0, right, predict, maxra = 0;    // ra - right answer, right - правильная цифра, predict - предсказание нейросети, maxra - макс. кол-во правильных ответов за 1 эпоху
    int epoch = 1;    // эпохи
    bool study, repeat = true;
    chrono::duration<double> time;

    NW_config = ReadDataNetWork("Configuration.txt");
    NW.Init(NW_config);
    NW.PrintConfig();

    while (repeat)
    {
        cout << "Обучаться? (1/0)" << endl;
        cin >> study;
        if (study)
        {
            int examples;
            data = ReadData("Examples_for_learning_MNIST.txt", NW_config, examples);
            auto begin = chrono::steady_clock::now();
            while (ra / examples * 100 < 100)
            {
                ra = 0;
                auto t1 = chrono::steady_clock::now();
                for (int i = 0; i < examples; ++i)
                {
                    NW.SetInput(data[i].pixels);
                    right = data[i].digit;
                    predict = NW.ForwardFeed();
                    if (predict != right)
                    {
                        NW.BackPropogation(right);
                        NW.WeightsUpdater(0.15 * exp(-epoch / 10.));    // Экспоненциальное затухание 
                    }
                    else
                        ra++;
                }
                auto t2 = chrono::steady_clock::now();
                time = t2 - t1;
                if (ra > maxra) maxra = ra;
                cout << "Правильные ответы: " << ra / examples * 100 << "%\t" << "Макс. кол-во правильных ответов за 1 эпоху: " << maxra / examples * 100 << "%\t" << "Эпоха: " << epoch << "\tВРЕМЯ: " << time.count() << "сек" << endl;
                epoch++;
                if (epoch == 11)
                    break;
            }
            auto end = chrono::steady_clock::now();
            time = end - begin;
            cout << "ВРЕМЯ: " << time.count() / 60. << " мин" << endl;
            NW.SaveWeights();
        }
        else {
            NW.ReadWeights();
        }
        cout << "Провести тестирование? (1/0)\n";
        bool test_flag;
        cin >> test_flag;
        if (test_flag)
        {
            int ex_tests;
            data_info* data_test;
            data_test = ReadData("Test_10k.txt", NW_config, ex_tests);
            ra = 0;
            for (int i = 0; i < ex_tests; ++i)
            {
                NW.SetInput(data_test[i].pixels);
                predict = NW.ForwardFeed();
                right = data_test[i].digit;
                if (right == predict)
                    ra++;
            }
            cout << "Правильные ответы: " << ra / ex_tests * 100 << "%" << endl;
        }
        cout << "Повторить? (1/0)\n";
        cin >> repeat;
    }

    int cont;
    cout << "Перейти к рисованию цифр? (1/0)" << endl;
    cin >> cont;
    if (cont)
    {
        int ex;
        data_info* data_test;
        data_test = ReadData("1.txt", NW_config, ex);
        ra = 0;
        for (int i = 0; i < ex; ++i)
        {
            NW.SetInput(data_test[i].pixels);
            predict = NW.ForwardFeed();
            right = data_test[i].digit;
            if (right == predict)
                ra++;
        }
        cout << "Правильные ответы: " << ra / ex * 100 << "%" << endl;

    }
    system("pause");
    return 0;
}
