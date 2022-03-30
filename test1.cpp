#include <arm_neon.h>
#include<assert.h>
#include <stdio.h>
#include <time.h>
#include<cmath>
const int dim = 32;
const int trainNum = 1028;
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

void one_cycle_unwrapped()
{
	for (int i = 0;i < testNum;i++)
	{
		for (int j = 0;j < trainNum;j++)
		{
			assert(dim % 4 == 0);//首先假定维度为4的倍数
			float32x4_t sum = vmovq_n_f32(0);
			for (int k = 0;k < dim;k += 4)
			{
				float32x4_t temp_test = vld1q_f32(&test[i][k]);
				float32x4_t temp_train = vld1q_f32(&train[j][k]);
				temp_test = vsubq_f32(temp_test, temp_train);
				//temp_test = vmulq_f32(temp_test, temp_test);
				//sum = vaddq_f32(sum, temp_test);
				sum = vmlaq_f32(sum, temp_test, temp_test);
			}
			float32x2_t sumlow = vget_low_f32(sum);
			float32x2_t sumhigh = vget_high_f32(sum);
			sumlow = vpadd_f32(sumlow, sumhigh);
			float32_t sumlh = vpadds_f32(sumlow);
			dist[i][j] = sqrtf((float)sumlh);
		}
	}
}


void timing(void (*func)())
{
	struct timespec sts, ets;
	timespec_get(&sts, TIME_UTC);
	func();
	timespec_get(&ets, TIME_UTC);
	time_t dsec = ets.tv_sec-sts.tv_sec;
	long dnsec = ets.tv_nsec-sts.tv_nsec;
	double time_in_ms = dsec * 1000.0 + dnsec / 1000000.0;
	printf("%fms\n", time_in_ms);
}

void init()
{
	for (int i = 0;i < testNum;i++)
		for (int k = 0;k < dim;k++)
			test[i][k] = rand() / double(RAND_MAX) * 1000;//0-100间随机浮点数
	for(int i = 0;i<trainNum;i++)
		for(int k = 0;k<dim;k++)
			train[i][k] = rand() / double(RAND_MAX) * 1000;//0-100间随机浮点数
}

int main()
{
	float distComp[testNum][trainNum];
	init();
	printf("%s", "朴素算法耗时：");
	timing(plain);
	float error = 0;
	for (int i = 0;i < testNum;i++)
		for (int j = 0;j < trainNum;j++)
			distComp[i][j] = dist[i][j];
	printf("%s", "SIMD算法耗时：");
	timing(one_cycle_unwrapped);
	for (int i = 0;i < testNum;i++)
		for (int j = 0;j < trainNum;j++)
			error += (distComp[i][j] - dist[i][j]) * (distComp[i][j] - dist[i][j]);
	printf("误差%f", error);
}