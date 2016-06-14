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
#include "opencv2/ml/ml.hpp"

//Project
#include "filecon.h"
#include "delaycalc.h"
#include "gpu.cuh"

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
#define HALFN			4097	  //8192/2 +1 for hilbert transform
#define STARTLOG		1e-9	  //Matlab logcompressdb for minimum
#define NUMCOFFILTER	38		  //Cof Filter

using namespace std;
using namespace cv;

int Div0Up(int a, int b)//fix int/int=0
{
	return ((a % b) != 0) ? (a / b + 1) : (a / b);
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
	double *filter = new double[SIGNAL_SIZE];
	float2 *vout_com = new float2[SIGNAL_SIZE * SCAN_LINE];
	float2 *filter_com = new float2[SIGNAL_SIZE];

	//Cuda mem init
	float2 *d_Signalfilter;
	float2 *d_filter_com;
	float2 *d_vout;
	double *d_tdr;
	double *d_raw_signal;
	double *d_max;
	double *d_env;
	int *d_mutex;

	loadRawData("D:\\20160613_122303_RawDataD_0.data", raw_data); // channel*scanline size
	loadData("D:\\ultrasound\\loadPsDelay.dat", SCAN_LINE, max_ps_delay);
	loadData("D:\\ultrasound\\Filter\\BandFilter38Cof.dat", NUMCOFFILTER, filter);
	loadElementRxs("D:\\ultrasound\\loadElementRxs.dat", elementRxs); // channel*scanline size

	for (int i = 0; i < SCAN_LINE; i++)
		t0[i] = NBEFOREPULSE + (max_ps_delay[i] / FREQ_FPGA_CLOCK * FREQ_SAMPLING);

	//Filter zero add
	for (int i = 0; i < NUMCOFFILTER; ++i)
	{
		filter_com[i].x = filter[i]; filter_com[i].y = 0;
	}
	for (int i = NUMCOFFILTER; i < SIGNAL_SIZE; ++i)
	{
		filter_com[i].x = 0; filter_com[i].y = 0;
	}

	//calc Delay time
	calc_TimeDelay(tdf, tdmin, NTX * 2, PITCH, SOUND_SPEED, FREQ_SAMPLING); // TDF
	calc_tdds(tdds, NRX * 2, tdf, tdmin, FREQ_SAMPLING); //TDDS	
	calc_tdfindex(tdfindex, NRX, elementRxs);// Index TDF
	calc_tdr(tdr, NRX, tdds, tdfindex, t0);//TDR

	clock_t startTime1 = clock();
	delaysum_beamforming(vout, tdr, raw_data);
	cout << "CPU delaysum_beamforming times = "<<double(clock() - startTime1) / (double)CLOCKS_PER_SEC*1000 << " ms." << endl;

	//CUDA INIT
	cufftHandle plan;
	cufftPlan1d(&plan, SIGNAL_SIZE, CUFFT_C2C, 81);
	cufftHandle plan1;
	cufftPlan1d(&plan1, SIGNAL_SIZE, CUFFT_C2C,1);
	cudaMalloc((void **)&d_filter_com, SIGNAL_SIZE * sizeof(float2));
	cudaMalloc((void **)&d_vout, SIGNAL_SIZE * SCAN_LINE * sizeof(float2));
	cudaMalloc((void **)&d_Signalfilter, SIGNAL_SIZE * SCAN_LINE * sizeof(float2));
	cudaMalloc((void **)&d_tdr, Fullsize);
	cudaMalloc((void **)&d_raw_signal, Fullsize);
	cudaMalloc((void **)&d_env, Imgsize);
	cudaMalloc((void**)&d_max, sizeof(double));
	cudaMalloc((void**)&d_mutex, sizeof(int));
	cudaMemset(d_max, 0, sizeof(float));
	cudaMemset(d_mutex, 0, sizeof(float));
	cudaMemcpy(d_filter_com, filter_com, SIGNAL_SIZE * sizeof(float2), cudaMemcpyHostToDevice);
	cudaMemcpy(d_tdr, tdr, Fullsize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_raw_signal, raw_data, Fullsize, cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float mi = 0, sm = 0 ,mh=0 ,mf=0 ,ml=0;

	/*
	for (int nl = 0; nl < SCAN_LINE; nl++){
		cudaEventRecord(start);
		beamforming1scanline << < 32, 256 >> >(nl, d_vout, d_tdr, d_raw_signal);
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&mi, start, stop);
		sm += mi;
	}
	cout << "Gpu basic Beamfrom times = " << sm << "ms\n";
	*/

	cudaEventRecord(start);
	improve<< <dim3(256,1,1),dim3(32,32,1) >> >(d_vout, d_tdr, d_raw_signal);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&mi, start, stop);
	cout <<"Gpu Improve beamfrom times = "<<mi<< "ms\n";
	
	//DATAF Output
	/*
	cudaMemcpy(vout_com, d_vout, SIGNAL_SIZE * SCAN_LINE * sizeof(float2), cudaMemcpyDeviceToHost);
	for (int i = 0; i < 8192 * 81; i++)
		vout[i] = vout_com[i].x;
	writeFile("D:\\ultrasound\\save.dat", dataLength, vout); //output Vout
	*/

	//OLD FILTER SLOW
	/*
	cufftExecC2C(plan, (cufftComplex *)d_filter_com, (cufftComplex *)d_filter_com, CUFFT_FORWARD);
	for (int nl = 0; nl < 81; nl++ )
	{
		cufftExecC2C(plan1, (cufftComplex *)(d_vout + nl*SIGNAL_SIZE), (cufftComplex *)(d_vout + nl*SIGNAL_SIZE), CUFFT_FORWARD);
		FilterCalc << <dim3(256, 1, 1), dim3(1024, 1, 1) >> >(d_Signalfilter , d_vout, d_filter_com, nl);
		hilbert_1line_step2 << <dim3(256, 1, 1), dim3(1024, 1, 1) >> >(d_Signalfilter + nl*SIGNAL_SIZE);
		cufftExecC2C(plan1, (cufftComplex *)(d_Signalfilter + nl*SIGNAL_SIZE), (cufftComplex *)(d_Signalfilter + nl*SIGNAL_SIZE), CUFFT_INVERSE);
	}
	*/

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
	
	//Median Filter
	cudaEventRecord(start);
	Gpu_median_filter << <dim3(780, 1, 1), dim3(8, 128, 1) >> >(d_env, d_env, SIGNAL_SIZE, SCAN_LINE); // (x/FREQ_SAMPLING*SOUND_SPEED/2*100) = cm , if 12 cm x=6234 ,6234/8 = 780 Fullpic948
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&mf, start, stop);
	cout << "Gpu Median Filter times = " << mf << "ms\n";
	
	//LogCompression
	cudaEventRecord(start);
	find_maximum << < 32, 256 >> >(d_env, d_max, d_mutex, SIGNAL_SIZE*SCAN_LINE); //<-danger
	logCompressDB << <dim3(256, 1, 1), dim3(32, 32, 1) >> >(d_env, d_max);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ml, start, stop);
	cout << "Gpu LogCompression times = " << ml << "ms\n";

	//Output
	cudaMemcpy(vout, d_env, Imgsize, cudaMemcpyDeviceToHost);
	writeFile("D:\\ultrasound\\save.dat", dataLength, vout); //output Vout
	
	for (int i = 0; i < 8192 * 81; i++)
		vout[i] = (vout[i] + 180) / 255;
	Mat A = Mat(81, 8192, CV_64FC1, vout);
	A = A(Rect(0, 0, 6234, 81)); //Crop 6234 = 12  Cm
	resize(A, A, Size(768, 243), CV_INTER_CUBIC);
	transpose(A, A);
	imshow("Image", A);
	waitKey(0);

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
	delete filter;
	delete vout_com;
	delete filter_com;

	cufftDestroy(plan);
	cufftDestroy(plan1);
	cudaFree(d_vout);
	cudaFree(d_tdr);
	cudaFree(d_raw_signal);
	cudaFree(d_mutex);
	cudaFree(d_max);
	cudaFree(d_env);
	cudaFree(d_Signalfilter);
}
