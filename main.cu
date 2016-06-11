#include <fstream>
#include <iostream>
#include <ctime>

#include "filecon.h"
#include "delaycalc.h"

//Cuda header
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"
#include <cufft.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#define FREQ_FPGA_CLOCK 100000000 //100Mhz
#define FREQ_SAMPLING	40000000  //40Mhz
#define PITCH			0.0005	  //0.5mn
#define SOUND_SPEED		1540	  //1540m/s
#define SIGNAL_SIZE		8192	  //H
#define CHANNEL			32		  //Point
#define SCAN_LINE		81	      //W
#define NBEFOREPULSE	538
#define NRX				32
#define NTX				32
#define HALFN			4097

using namespace std;

int Div0Up(int a, int b)//fix int/int=0
{
	return ((a % b) != 0) ? (a / b + 1) : (a / b);
}
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
			env[p + (nl * SIGNAL_SIZE)] = sqrt(pow(signal[p + (nl * SIGNAL_SIZE)].x, 2) + pow(signal[p + (nl * SIGNAL_SIZE)].y, 2));
		}
	}
}

void delaysum_beamforming(double *output, const double *tdr, const double *raw_signal)
{
	double sum = 0;
	int Ntdr = 0;
	for (int nl = 0; nl < SCAN_LINE; nl++) //81 scanline
	{
		for (int p = 0; p < SIGNAL_SIZE; p++)//1 scanline 8192 point
		{
			sum = 0;
			for (int i = 0; i < CHANNEL; i++)//1 point = 32 channel
			{
				Ntdr = int(tdr[i + (p * CHANNEL) + (nl * SIGNAL_SIZE * CHANNEL)]);
				if (Ntdr < SIGNAL_SIZE) //protect out of SIGNAL_SIZE bound
				{
					sum += raw_signal[Ntdr + (i * SIGNAL_SIZE) + (nl * CHANNEL * SIGNAL_SIZE)];//H+C+W
				}
			}
			output[p + (nl * SIGNAL_SIZE)] = sum;
		}
	}
}

int main()
{
	//Cpu mem init
	int    dataLength = SIGNAL_SIZE * SCAN_LINE; // 663552
	int	   Fullsize = dataLength * CHANNEL *sizeof(double);
	int    Imgsize = dataLength * sizeof(double);
	int	   *tdfindex = new int[CHANNEL * SIGNAL_SIZE];
	double *t0 = new double[SCAN_LINE];
	double *max_ps_delay = new double[SCAN_LINE];
	double *tdmin = new double[SIGNAL_SIZE];
	double *elementRxs = new double[CHANNEL * SCAN_LINE];
	double *tdf = new double[2 * CHANNEL * SIGNAL_SIZE];
	double *tdds = new double[2 * CHANNEL * SIGNAL_SIZE];
	double *vout = new double[SIGNAL_SIZE * SCAN_LINE];
	double *raw_data = new double[SIGNAL_SIZE * CHANNEL * SCAN_LINE];
	double *tdr = new double[SIGNAL_SIZE * CHANNEL * SCAN_LINE];
	float2 *vout_com = new float2[SIGNAL_SIZE * SCAN_LINE];

	//Cuda mem init
	float2 *d_vout;
	double *d_tdr;
	double *d_raw_signal;
	double *d_env;

	loadRawData("D:\\ultrasound\\loadData.dat", raw_data); // channel*scanline size
	loadData("D:\\ultrasound\\loadPsDelay.dat", SCAN_LINE, max_ps_delay);
	loadElementRxs("D:\\ultrasound\\loadElementRxs.dat", elementRxs); // channel*scanline size
	for (int i = 0; i < SCAN_LINE; i++)
		t0[i] = NBEFOREPULSE + (max_ps_delay[i] / FREQ_FPGA_CLOCK * FREQ_SAMPLING);
	calc_TimeDelay(tdf, tdmin, NTX * 2, PITCH, SOUND_SPEED, FREQ_SAMPLING); // TDF
	calc_tdds(tdds, NRX * 2, tdf, tdmin, FREQ_SAMPLING); //TDDS	
	calc_tdfindex(tdfindex, NRX, elementRxs);// Index TDF
	calc_tdr(tdr, NRX, tdds, tdfindex, t0);//TDR

	//clock_t startTime1 = clock();
	//delaysum_beamforming(vout, tdr, raw_data);
	//cout << "delaysum_beamforming times = "<<double(clock() - startTime1) / (double)CLOCKS_PER_SEC*1000 << " ms." << endl;

	cufftHandle plan;
	cufftPlan1d(&plan, SIGNAL_SIZE, CUFFT_C2C, 81);

	cudaMalloc((void **)&d_vout, SIGNAL_SIZE * SCAN_LINE * sizeof(float2));
	cudaMalloc((void **)&d_tdr, Fullsize);
	cudaMalloc((void **)&d_raw_signal, Fullsize);
	cudaMalloc((void **)&d_env, Imgsize);

	cudaMemcpy(d_tdr, tdr, Fullsize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_raw_signal, raw_data, Fullsize, cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float mi = 0, sm = 0 ,mh=0;

	/*for (int nl = 0; nl < SCAN_LINE; nl++){
		cudaEventRecord(start);
		beamforming1scanline << < 32, 256 >> >(nl, d_vout, d_tdr, d_raw_signal);
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&mi, start, stop);
		sm += mi;
	}
	cout << "Gpubasic Beamfrom times = " << sm << "ms\n";
	*/
	
	cudaEventRecord(start);
	improve<< <dim3(256,1,1),dim3(32,32,1) >> >(d_vout, d_tdr, d_raw_signal);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&mi, start, stop);
	cout <<"GpuImprove beamfrom times = "<<mi<< "ms\n";
	
	cudaEventRecord(start);
	cufftExecC2C(plan, (cufftComplex *)d_vout, (cufftComplex *)d_vout, CUFFT_FORWARD);
	hilbert_step2 << <dim3(256, 1, 1), dim3(32, 32, 1) >> >(d_vout);
	cufftExecC2C(plan, (cufftComplex *)d_vout, (cufftComplex *)d_vout, CUFFT_INVERSE);
	abscomplex << <dim3(256, 1, 1), dim3(32, 32, 1) >> >(d_env, d_vout);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&mh, start, stop);
	cout << "Gpu Hilbert times = " << mh << "ms\n";

	cudaMemcpy(vout, d_env, Imgsize, cudaMemcpyDeviceToHost);
	writeFile("D:\\ultrasound\\save.dat", dataLength, vout); //output Vout

	delete tdfindex;
	delete raw_data;
	delete max_ps_delay;
	delete elementRxs;
	delete t0;
	delete tdf;
	delete tdds;
	delete tdmin;
	delete tdr;
	delete vout;
}
