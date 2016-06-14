#include "gpu.cuh"

#define SIGNAL_SIZE		8192	  //H
#define CHANNEL			32		  //Point
#define SCAN_LINE		81	      //W
#define HALFN			4097	  //8192/2 +1 for hilbert transform
#define STARTLOG		1e-9	  //Matlab logcompressdb

/*
__device__ void channelcalc(double *sum, int p, int nl, const double *tdr, const double *raw_data)
{
int Ntdr = tdr[threadIdx.x + (p * CHANNEL) + (nl * SIGNAL_SIZE * CHANNEL)];
if (Ntdr < SIGNAL_SIZE) //protect out of SIGNAL_SIZE bound
{
*sum += raw_data[Ntdr + (threadIdx.x * SIGNAL_SIZE) + (nl * CHANNEL * SIGNAL_SIZE)];//H+C+W
}
}
*/

__global__ void beamforming1scanline(int nl, float2 *vout, const double *tdr, const double *raw_data)
{
	const int nThd = blockDim.x * gridDim.x;
	const int tID = blockIdx.x * blockDim.x + threadIdx.x;
	double sum;
	int Ntdr = 0;
	for (int p = tID; p < SIGNAL_SIZE; p += nThd)//1 scanline 8192 point
	{
		sum = 0;
		//printf("W = %d H = %d\n", nl,threadID);
		for (int i = 0; i < CHANNEL; i++)//1 point = 32 channel
		{
			//channelcalc << <1, 32 >> >(sum, p, nl, tdr, raw_data); // my computer not suppport (compute capability > 3.5)
			Ntdr = tdr[i + (p * CHANNEL) + (nl * SIGNAL_SIZE * CHANNEL)];
			//printf("%d ", Ntdr);
			if (Ntdr < SIGNAL_SIZE) //protect out of SIGNAL_SIZE bound
			{
				sum += raw_data[Ntdr + (i * SIGNAL_SIZE) + (nl * CHANNEL * SIGNAL_SIZE)];//H+C+W
			}
		}
		vout[p + (nl * SIGNAL_SIZE)].x = sum;
		vout[p + (nl * SIGNAL_SIZE)].y = 0;
		//printf("sum = %lf\n",sum]);
	}
}

__global__ void improve(float2 *vout, const double *tdr, const double *raw_data)
{
	const int nThdx = blockDim.x * gridDim.x;
	const int nThdy = blockDim.y * gridDim.y;
	const int tIDx = blockIdx.x * blockDim.x + threadIdx.x;
	const int tIDy = blockIdx.y * blockDim.y + threadIdx.y;
	double sum;
	int Ntdr = 0;
	for (int nl = tIDy; nl < SCAN_LINE; nl += nThdy) //81 scanline
	{
		for (int p = tIDx; p < SIGNAL_SIZE; p += nThdx)//1 scanline 8192 point
		{
			sum = 0;
			//printf("W = %d H = %d\n", nl,threadID);
			for (int i = 0; i < CHANNEL; i++)//1 point = 32 channel
			{
				//channelcalc << <1, 32 >> >(sum, p, nl, tdr, raw_data); // my computer not suppport (compute capability > 3.5)
				Ntdr = tdr[i + (p * CHANNEL) + (nl * SIGNAL_SIZE * CHANNEL)];
				//printf("%d ", Ntdr);
				if (Ntdr < SIGNAL_SIZE) //protect out of SIGNAL_SIZE bound
				{
					sum += raw_data[Ntdr + (i * SIGNAL_SIZE) + (nl * CHANNEL * SIGNAL_SIZE)];//H+C+W
				}
			}
			vout[p + (nl * SIGNAL_SIZE)].x = sum;
			vout[p + (nl * SIGNAL_SIZE)].y = 0;
			//printf("sum = %lf\n",sum]);
		}
	}
}

__global__ void hilbert_step2(float2 *signal)
{
	const int nThdx = blockDim.x * gridDim.x;
	const int nThdy = blockDim.y * gridDim.y;
	const int tIDx = blockIdx.x * blockDim.x + threadIdx.x;
	const int tIDy = blockIdx.y * blockDim.y + threadIdx.y;
	for (int nl = tIDy; nl < SCAN_LINE; nl += nThdy) //81 scanline
	{
		for (int p = tIDx; p < SIGNAL_SIZE; p += nThdx)//1 scanline 8192 point
		{
			if (p == 0);
			else if (p < HALFN)
			{
				signal[p + (nl * SIGNAL_SIZE)].x *= 2; signal[p + (nl * SIGNAL_SIZE)].y *= 2;
			}
			else
			{
				signal[p + (nl * SIGNAL_SIZE)].x = 0.0; signal[p + (nl * SIGNAL_SIZE)].y = 0.0;
			}
		}
	}
}

__global__ void abscomplex(double *env, float2 *signal)
{
	const int nThdx = blockDim.x * gridDim.x;
	const int nThdy = blockDim.y * gridDim.y;
	const int tIDx = blockIdx.x * blockDim.x + threadIdx.x;
	const int tIDy = blockIdx.y * blockDim.y + threadIdx.y;
	for (int nl = tIDy; nl < SCAN_LINE; nl += nThdy) //81 scanline
	{
		for (int p = tIDx; p < SIGNAL_SIZE; p += nThdx)//1 scanline 8192 point
		{
			env[p + (nl * SIGNAL_SIZE)] = pow(signal[p + (nl * SIGNAL_SIZE)].x, 2) + pow(signal[p + (nl * SIGNAL_SIZE)].y, 2);
		}
	}
}


__global__ void Gpu_median_filter(double *Input_Image, double *Output_Image, int img_h, int img_w) 
{
	float ingpuArray[9];
	int count = 0;
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	if ((x >= (img_h - 1)) || (y >= img_w - 1) || (x == 0) || (y == 0)) //กรอบ 0
		return;
	for (int r = x - 1; r <= x + 1; r++)
	{
		for (int c = y - 1; c <= y + 1; c++)
		{
			ingpuArray[count++] = Input_Image[c*img_h + r];
		}
	}
	for (int i = 0; i<5; ++i)
	{
		int min = i;
		for (int l = i + 1; l<9; ++l)
			if (ingpuArray[l] < ingpuArray[min])
				min = l;
		//swap(a,b)
		float temp = ingpuArray[i];
		ingpuArray[i] = ingpuArray[min];
		ingpuArray[min] = temp;
	}
	Output_Image[(y*img_h) + x] = ingpuArray[4]; // 4 mid
}

__global__ void find_maximum(double *array, double *max, int *mutex, unsigned int n)
{
	unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int stride = gridDim.x*blockDim.x;
	unsigned int offset = 0;
	__shared__ double cache[256];
	double temp = -1.0;
	while (index + offset < n)
	{
		temp = fmaxf(temp, array[index + offset]);
		offset += stride;
	}
	cache[threadIdx.x] = temp;
	__syncthreads();
	unsigned int i = blockDim.x / 2; // reduction
	while (i != 0)
	{
		if (threadIdx.x < i)
			cache[threadIdx.x] = fmaxf(cache[threadIdx.x], cache[threadIdx.x + i]);
		__syncthreads();
		i /= 2;
	}
	if (threadIdx.x == 0)
	{
		while (atomicCAS(mutex, 0, 1) != 0);  //lock
		*max = fmaxf(*max, cache[0]);
		atomicExch(mutex, 0);  //unlock
	}
}

__global__ void logCompressDB(double *env, double *d_max)
{
	const int nThdx = blockDim.x * gridDim.x;
	const int nThdy = blockDim.y * gridDim.y;
	const int tIDx = blockIdx.x * blockDim.x + threadIdx.x;
	const int tIDy = blockIdx.y * blockDim.y + threadIdx.y;
	for (int nl = tIDy; nl < SCAN_LINE; nl += nThdy)//81 scanline
	{
		for (int p = tIDx; p < SIGNAL_SIZE; p += nThdx)//1 scanline 8192 point
		{
			env[p + (nl * SIGNAL_SIZE)] = 10.0 * __logf(__fdividef(env[p + (nl * SIGNAL_SIZE)], (float)*d_max) + STARTLOG);
		}
	}

}
