#include <fstream>
#include <iostream>
#include <ctime>

//Cuda header
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"
#include <cufft.h>
#include <helper_functions.h>
#include <helper_cuda.h>

//OpenCV
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/photo.hpp"
#include "opencv2/cudawarping.hpp" //Cuda opencv
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudafilters.hpp"
#include "opencv2/photo/cuda.hpp"
#include "opencv2/cudaarithm.hpp"

//Project
#include "filecon.h"
#include "delaycalc.h"
#include "gpu.cuh"

#define FREQ_FPGA_CLOCK 100000000 //100Mhz
#define FREQ_SAMPLING	40000000  //40Mhz
#define PITCH			0.0005	  //0.5mn
#define SOUND_SPEED		1545	  //1540m/s
#define SIGNAL_SIZE		8192	  //H
#define CHANNEL			32		  //Point
#define SCAN_LINE		81	      //W
#define NBEFOREPULSE	538
#define NRX				32
#define NTX				32
#define HALFN			4097	  //8192/2 +1 for hilbert transform
#define STARTLOG		1e-9	  //Matlab logcompressdb for minimum
#define NUMCOFFILTER	38		  //Cof Filter

using namespace std;
using namespace cv;

__global__ void img_calc(unsigned char *out,double *in)
{
	const int nThdx = blockDim.x * gridDim.x;
	const int nThdy = blockDim.y * gridDim.y;
	const int tIDx = blockIdx.x * blockDim.x + threadIdx.x;
	const int tIDy = blockIdx.y * blockDim.y + threadIdx.y;
	for (int nl = tIDy; nl < SCAN_LINE; nl += nThdy) //81 scanline
	{
		for (int p = tIDx; p < SIGNAL_SIZE; p += nThdx)//1 scanline 8192 point
		{
			if (in[p + (nl*SIGNAL_SIZE)] + 180 < 0) out[p + (nl*SIGNAL_SIZE)] = 0;
			else if (in[p + (nl*SIGNAL_SIZE)] + 180 > 255) out[p + (nl*SIGNAL_SIZE)] = 255;
			else out[p + (nl*SIGNAL_SIZE)] = char(ceilf(in[p + (nl*SIGNAL_SIZE)] + 180));
		}
	}
}

int Div0Up(int a, int b)//fix int/int=0
{
	return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

void delaysum_beamforming(double *output, const double *tdr, const __int16 *raw_signal)
{
	__int16 sum = 0 ,  Ntdr = 0;
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
	double *tdr = new double[SIGNAL_SIZE * CHANNEL * SCAN_LINE];
	double *filter = new double[SIGNAL_SIZE];
	float2 *filter_com = new float2[SIGNAL_SIZE];
	__int16 *raw_data = new __int16[dataLength * CHANNEL];

	//Cuda mem init
	unsigned char *d_img;
	__int16 *d_raw_signal;
	float2 *d_Signalfilter;
	float2 *d_filter_com;
	float2 *d_vout;
	double *d_tdr;
	double *d_max;
	double *d_env;
	int *d_mutex;
	
	//LOAD DATA
	loadRawData("D:\\ultrasound\\RawData\\RawData20150613\\RMSGray1NoTGC\\20160613_122303_RawDataD_0.data", raw_data); // channel*scanline size
	loadData("D:\\ultrasound\\ProgramData\\loadPsDelay.dat", SCAN_LINE, max_ps_delay);
	loadData("D:\\ultrasound\\Filter\\BandFilter38Cof.dat", NUMCOFFILTER, filter);
	loadElementRxs("D:\\ultrasound\\ProgramData\\loadElementRxs.dat", elementRxs); // channel*scanline size

	//deswitch noise
	for (int i = 0; i < SCAN_LINE; i++)
		t0[i] = NBEFOREPULSE + (max_ps_delay[i] / FREQ_FPGA_CLOCK * FREQ_SAMPLING);

	//Filter zero add
	for (int i = 0; i < NUMCOFFILTER; ++i)
	{
		filter_com[i].x = (float)filter[i];
		filter_com[i].y = 0;
	}
	for (int i = NUMCOFFILTER; i < SIGNAL_SIZE; ++i)
	{
		filter_com[i].x = 0;
		filter_com[i].y = 0;
	}

	//calc Delay time
	calc_TimeDelay(tdf, tdmin, NTX * 2, PITCH, SOUND_SPEED, FREQ_SAMPLING); // TDF
	calc_tdds(tdds, NRX * 2, tdf, tdmin, FREQ_SAMPLING); //TDDS	
	calc_tdfindex(tdfindex, NRX, elementRxs);// Index TDF
	calc_tdr(tdr, NRX, tdds, tdfindex, t0);//TDR

	//Clear RAM
	delete max_ps_delay;
	delete filter;
	delete tdmin;
	delete tdf;
	delete tdds;
	delete elementRxs;
	delete t0;
	delete tdfindex;
	
	//Stopwatch cpu
	/*clock_t startTime1 = clock();
	delaysum_beamforming(vout, tdr, raw_data);
	cout << "CPU delaysum_beamforming times = "<<double(clock() - startTime1) / (double)CLOCKS_PER_SEC*1000 << " ms." << endl;*/

	//CUDA INIT
	cufftHandle plan;
	cufftPlan1d(&plan, SIGNAL_SIZE, CUFFT_C2C, 81); //For Singal
	cufftHandle plan1;
	cufftPlan1d(&plan1, SIGNAL_SIZE, CUFFT_C2C, 1); //For Filter
	cudaMalloc((void **)&d_img, dataLength);
	cudaMalloc((void **)&d_filter_com, SIGNAL_SIZE * sizeof(float2));
	cudaMalloc((void **)&d_vout, SIGNAL_SIZE * SCAN_LINE * sizeof(float2));
	cudaMalloc((void **)&d_Signalfilter, SIGNAL_SIZE * SCAN_LINE * sizeof(float2));
	cudaMalloc((void **)&d_tdr, Fullsize);
	cudaMalloc((void **)&d_raw_signal, dataLength * CHANNEL * sizeof(__int16));
	cudaMalloc((void **)&d_env, Imgsize);
	cudaMalloc((void**)&d_max, sizeof(double));
	cudaMalloc((void**)&d_mutex, sizeof(int));
	cudaMemset(d_max, 0, sizeof(float));
	cudaMemset(d_mutex, 0, sizeof(float));
	cudaMemcpy(d_tdr, tdr, Fullsize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_raw_signal, raw_data, dataLength * CHANNEL * sizeof(__int16), cudaMemcpyHostToDevice);
	cudaMemcpy(d_filter_com, filter_com, SIGNAL_SIZE * sizeof(float2), cudaMemcpyHostToDevice);

	//Clear RAM
	delete tdr;
	delete raw_data;
	delete filter_com;

	//Stopwatch gpu
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float mb=0,mh=0,mf=0,ml=0;

	//beamfroming
	cudaEventRecord(start);
	improve<< <dim3(256,1,1),dim3(32,32,1) >> >(d_vout, d_tdr, d_raw_signal);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&mb, start, stop);
	cout <<"Gpu Improve beamfrom times = "<<mb<< "ms\n";
	
	cudaFree(d_tdr);
	cudaFree(d_raw_signal);

	//Filter,Hilbert,abs
	cudaEventRecord(start);
	cufftExecC2C(plan, (cufftComplex *)d_vout, (cufftComplex *)d_vout, CUFFT_FORWARD);
	cufftExecC2C(plan1, (cufftComplex *)d_filter_com, (cufftComplex *)d_filter_com, CUFFT_FORWARD);
	FilterCalcImprove1 << <dim3(256, 1, 1), dim3(32, 32, 1) >> >(d_Signalfilter, d_vout, d_filter_com);
	hilbert_step2 << <dim3(256, 1, 1), dim3(32, 32, 1) >> >(d_Signalfilter);
	cufftExecC2C(plan, (cufftComplex *)d_Signalfilter, (cufftComplex *)d_Signalfilter, CUFFT_INVERSE);
	abscomplex << <dim3(256, 1, 1), dim3(32, 32, 1) >> >(d_env, d_Signalfilter);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&mh, start, stop);
	cout << "Gpu Filter,Hilbert,abs times = " << mh << "ms\n";
	
	cufftDestroy(plan);
	cufftDestroy(plan1);
	cudaFree(d_vout);
	cudaFree(d_Signalfilter);
	
	//LogCompression
	cudaEventRecord(start);
	find_maximum << < 32, 256 >> >(d_env, d_max, d_mutex, SIGNAL_SIZE*SCAN_LINE); //<-danger
	logCompressDB << <dim3(256, 1, 1), dim3(32, 32, 1) >> >(d_env, d_max);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ml, start, stop);
	cout << "Gpu LogCompression times = " << ml << "ms\n";

	//Median Filter
	/*
	cudaEventRecord(start);
	Gpu_median_filter << <dim3(780, 1, 1), dim3(8, 128, 1) >> >(d_env, d_env, SIGNAL_SIZE, SCAN_LINE); // (x/FREQ_SAMPLING*SOUND_SPEED/2*100) = cm , if 12 cm x=6234 ,6234/8 = 780 Fullpic948
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&mf, start, stop);
	cout << "Gpu Median Filter times = " << mf << "ms\n";
	*/

	cudaFree(d_mutex);
	cudaFree(d_max);

	//Output File
	cudaMemcpy(vout, d_env, Imgsize, cudaMemcpyDeviceToHost);
	writeFile("D:\\ultrasound\\save.dat", dataLength, vout); //output DATAF 
	
	//OpenCV IMG - non localmean
	
	img_calc << <dim3(256, 1, 1), dim3(32, 32, 1) >> >(d_img, d_env);
	Mat ixA;
	cuda::GpuMat d_ixA = cuda::GpuMat(81, 8192, CV_8U, d_img);
	d_ixA = d_ixA(Rect(0, 0, 6234, 81));
	clock_t startTime1 = clock();
	cuda::fastNlMeansDenoising(d_ixA, d_ixA, 30, 15, 7);
	cout << "Non local Mean Filter = " << double(clock() - startTime1) / (double)CLOCKS_PER_SEC * 1000 << " ms." << endl;
	
	//vertical , lateral
	Mat X;
	d_ixA.download(X);
	unsigned char* dataMat = X.data;
	ofstream output1("D://Y.dat", ios::binary | ios::out);
	for (int i = 40 * 6234; i < 40 * 6234 + 6234; i++)
	{
		output1.write((char *)&dataMat[i], sizeof(char));
	}
	output1.close();
	transpose(X, X);
	dataMat = X.data;
	ofstream output("D://X.dat", ios::binary | ios::out);
	for (int i = 3166 * 81; i < 3166 * 81 + 81; i++)
	{
		output.write((char *)&dataMat[i], sizeof(char));
	}
	output.close();

	cuda::resize(d_ixA, d_ixA, Size(768, 243), CV_INTER_NN);
	cuda::transpose(d_ixA, d_ixA);
	d_ixA.download(ixA);
	//imshow("Image1", ixA);

	//All img
	Scalar mean;
	Scalar stddev;
	cv::meanStdDev(ixA, mean, stddev);
	double mean_pxl = mean.val[0];
	double stddev_pxl = stddev.val[0];
	cout << "mean = "<<mean_pxl << " stddev = " << stddev_pxl << endl;

	//Point img
	Mat ixB = ixA(Rect(100, 390, 45, 45));
	//imshow("Image2", ixB);
	Scalar mean2;
	Scalar stddev2;
	cv::meanStdDev(ixB, mean2, stddev2);
	double mean_pxl2 = mean2.val[0];
	double stddev_pxl2 = stddev2.val[0];
	cout << "mean2 = " << mean_pxl2 << " stddev2 = " << stddev_pxl2 << endl;

	//Ground img
	Mat ixC = ixA(Rect(100, 200, 45, 45));
	//imshow("Image3", ixC);
	Scalar mean3;
	Scalar stddev3;
	cv::meanStdDev(ixC, mean3, stddev3);
	double mean_pxl3 = mean3.val[0];
	double stddev_pxl3 = stddev3.val[0];
	cout << "mean3 = " << mean_pxl3 << " stddev3 = " << stddev_pxl3 << endl;

	cout << "CNR = "<<20*log10(abs(mean_pxl2 - mean_pxl3) / sqrt(pow(stddev_pxl2, 2) + pow(stddev_pxl3, 2)))<<" dB"<<endl;
	
	//Img Save
	imwrite("D:\\Gray_Image.bmp", ixA);
	imwrite("D:\\Point.bmp", ixB);
	imwrite("D:\\Ground.bmp", ixC);
	waitKey(0);
	
	//Clear RAM
	delete vout;

	//Clear MEM GPU
	cudaFree(d_env);
	cudaFree(d_img);
}
