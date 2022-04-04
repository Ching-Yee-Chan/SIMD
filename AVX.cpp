#include <xmmintrin.h> //SSE
#include <emmintrin.h> //SSE2
#include <pmmintrin.h> //SSE3
#include <tmmintrin.h> //SSSE3
#include <smmintrin.h> //SSE4.1
#include <nmmintrin.h> //SSSE4.2
#include <immintrin.h> //AVX、AVX2
#include<assert.h>
#include<windows.h>
#include <stdio.h>
#include<iostream>
#include <time.h>
#include<cmath>
#define INTERVAL 1000
typedef long long ll;
const int dim = 128;
const int trainNum = 1024;
const int testNum = 128;
float train[trainNum][dim];
float test[testNum][dim];
float dist[testNum][trainNum];
void plain()
{
	for (int i = 0;i < testNum;i++)
	{
		for (int j = 0;j < trainNum;j++)
		{
			float sum = 0.0;
			for (int k = 0;k < dim;k++)
			{
				float temp = test[i][k] - train[j][k];
				temp *= temp;
				sum += temp;
			}
			dist[i][j] = sqrtf(sum);
		}
	}
}

void sqrt_unwrapped()
{
	for (int i = 0;i < testNum;i++)
	{
		for (int j = 0;j < trainNum;j++)
		{
			assert(dim % 8 == 0);//首先假定维度为8的倍数
			__m256 sum = _mm256_setzero_ps();
			for (int k = 0;k < dim;k += 8)
			{
				__m256 temp_test = _mm256_load_ps(&test[i][k]);
				__m256 temp_train = _mm256_load_ps(&train[j][k]);
				temp_test = _mm256_sub_ps(temp_test, temp_train);
				temp_test = _mm256_mul_ps(temp_test, temp_test);
				sum = _mm256_add_ps(sum, temp_test);
			}
			__m256 hi = _mm256_permute2f128_ps(sum, sum, 1);
			sum = _mm256_add_ps(sum, hi);
			sum = _mm256_hadd_ps(sum, sum);
			sum = _mm256_hadd_ps(sum, sum);
			float tempArray[8];
			_mm256_store_ps(tempArray, sum);
			dist[i][j] = tempArray[0];
		}
		for (int j = 0;j < trainNum;j += 8)
		{
			__m256 temp_dist = _mm256_load_ps(&dist[i][j]);
			temp_dist = _mm256_sqrt_ps(temp_dist);
			_mm256_store_ps(&dist[i][j], temp_dist);
		}
	}
}

void timing(void (*func)())
{
	ll head, tail, freq;
	double time = 0;
	int counter = 0;
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	while (INTERVAL > time)
	{
		func();
		QueryPerformanceCounter((LARGE_INTEGER*)&tail);
		counter++;
		time = (tail - head) * 1000.0 / freq;
	}
	std::cout << time / counter << '\n';
}

void init()
{
	for (int i = 0;i < testNum;i++)
		for (int k = 0;k < dim;k++)
			test[i][k] = rand() / double(RAND_MAX) * 1000;//0-100间随机浮点数
	for (int i = 0;i < trainNum;i++)
		for (int k = 0;k < dim;k++)
			train[i][k] = rand() / double(RAND_MAX) * 1000;//0-100间随机浮点数
}

int main()
{
	float distComp[testNum][trainNum];
	init();
    printf("%s%p\n", "train首地址", train);
	printf("%s%p\n", "test首地址", test);
	printf("%s%p\n", "dist首地址", dist);
	printf("%s%p\n", "distComp首地址", distComp);
	printf("%s", "朴素算法耗时：");
	timing(plain);
	float error = 0;
	for (int i = 0;i < testNum;i++)
		for (int j = 0;j < trainNum;j++)
			distComp[i][j] = dist[i][j];
	printf("%s", "AVX算法耗时：");
	timing(sqrt_unwrapped);
	for (int i = 0;i < testNum;i++)
		for (int j = 0;j < trainNum;j++)
			error += (distComp[i][j] - dist[i][j]) * (distComp[i][j] - dist[i][j]);
	printf("误差%f", error);
	system("pause");
	return 0;
}
