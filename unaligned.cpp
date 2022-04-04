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
typedef long long ll;
const int dim = 128;
const int trainNum = 1028;
const int testNum = 128;
float train[trainNum+1][dim+1];
float test[testNum+1][dim+1];
float dist[testNum+1][trainNum+1];
void plain()
{
	for (int i = 1;i <= testNum;i++)
	{
		for (int j = 1;j <= trainNum;j++)
		{
			float sum = 0.0;
			for (int k = 1;k <= dim;k++)
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
	for (int i = 1;i <= testNum;i++)
	{
		for (int j = 1;j <= trainNum;j++)
		{
			assert(dim % 4 == 0);//首先假定维度为4的倍数
			__m128 sum = _mm_setzero_ps();
			for (int k = 1;k <= dim;k += 4)
			{
				__m128 temp_test = _mm_loadu_ps(&test[i][k]);
				__m128 temp_train = _mm_loadu_ps(&train[j][k]);
				temp_test = _mm_sub_ps(temp_test, temp_train);
				temp_test = _mm_mul_ps(temp_test, temp_test);
				sum = _mm_add_ps(sum, temp_test);
			}
			sum = _mm_hadd_ps(sum, sum);
			sum = _mm_hadd_ps(sum, sum);
			_mm_store_ss(dist[i] + j, sum);
		}
		for (int j = 1;j <= trainNum;j += 4)
		{
			__m128 temp_dist = _mm_loadu_ps(&dist[i][j]);
			temp_dist = _mm_sqrt_ps(temp_dist);
			_mm_storeu_ps(&dist[i][j], temp_dist);
		}
	}
}

void aligned()
{
	for (int i = 1;i <= testNum;i++)
	{
		for (int j = 1;j <= trainNum;j++)
		{
			assert(dim % 4 == 0);//首先假定维度为4的倍数
			__m128 sum = _mm_setzero_ps();
			float serial_sum = 0;
			for(int k = 1;k <= 3;k++)//处理1-3（头部）
            {
                float temp = test[i][k] - train[j][k];
				temp *= temp;
				serial_sum += temp;
            }
            //处理dim（尾部）
            float temp = test[i][dim] - train[j][dim];
            temp *= temp;
            serial_sum += temp;
			for (int k = 4;k < dim;k += 4)//4~dim-1是对齐的
			{
				__m128 temp_test = _mm_load_ps(&test[i][k]);
				__m128 temp_train = _mm_load_ps(&train[j][k]);
				temp_test = _mm_sub_ps(temp_test, temp_train);
				temp_test = _mm_mul_ps(temp_test, temp_test);
				sum = _mm_add_ps(sum, temp_test);
			}
			sum = _mm_hadd_ps(sum, sum);
			sum = _mm_hadd_ps(sum, sum);
			_mm_store_ss(dist[i] + j, sum);
			dist[i][j] += serial_sum;//串行与并行结果合并
		}
        for(int j = 1;j<=3;j++)//处理1-3（头部）
            dist[i][j] = sqrtf(dist[i][j]);
        dist[i][trainNum] = sqrtf(dist[i][trainNum]);//处理trainNum（尾部）
		for (int j = 4;j < trainNum;j += 4)
		{
			__m128 temp_dist = _mm_load_ps(&dist[i][j]);
			temp_dist = _mm_sqrt_ps(temp_dist);
			_mm_store_ps(&dist[i][j], temp_dist);
		}
	}
}
void timing(void (*func)())
{
	ll head, tail, freq;
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	func();
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	std::cout << (tail - head) * 1000.0 / freq << '\n';
}

void init()
{
	for (int i = 0;i <= testNum;i++)
		for (int k = 0;k <= dim;k++)
			test[i][k] = rand() / double(RAND_MAX) * 1000;//0-100间随机浮点数
	for (int i = 0;i <= trainNum;i++)
		for (int k = 0;k <= dim;k++)
			train[i][k] = rand() / double(RAND_MAX) * 1000;//0-100间随机浮点数
}

int main()
{
	float distComp[testNum+1][trainNum+1];
	init();
	printf("%s%p\n", "train首地址", train);
	printf("%s%p\n", "test首地址", test);
	printf("%s%p\n", "dist首地址", dist);
	printf("%s%p\n", "distComp首地址", distComp);
	printf("%s", "朴素算法耗时：");
	timing(plain);
	float error = 0;
	for (int i = 1;i <= testNum;i++)
		for (int j = 1;j <= trainNum;j++)
			distComp[i][j] = dist[i][j];
	printf("%s", "不对齐SIMD算法耗时：");
	timing(sqrt_unwrapped);
	for (int i = 1;i <= testNum;i++)
		for (int j = 1;j <= trainNum;j++)
			error += (distComp[i][j] - dist[i][j]) * (distComp[i][j] - dist[i][j]);
	printf("误差%f\n", error);
	error = 0;
	printf("%s", "对齐SIMD算法耗时：");
	timing(sqrt_unwrapped);
	for (int i = 1;i <= testNum;i++)
		for (int j = 1;j <= trainNum;j++)
			error += (distComp[i][j] - dist[i][j]) * (distComp[i][j] - dist[i][j]);
	printf("误差%f\n", error);
	system("pause");
	return 0;
}
