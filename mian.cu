#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <fstream>
#include <iostream>
#include <ctime>
#include "header/filecon.h"

//Cuda header
#include <cuda_runtime.h>
#include <cufft.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#define FREQ_FPGA_CLOCK 100000000 //100Mhz
#define FREQ_SAMPLING	40000000  //40Mhx
#define PITCH			0.0005	  //0.5mn
#define SOUND_SPEED		1540	  //1540m/s
#define SIGNAL_SIZE		8192	  
#define CHANNEL			32
#define SCAN_LINE		81
#define NBEFOREPULSE	538
#define NRX				32
#define NTX				32

using namespace std;

void calTimeDelay(double *tdf, double *tdmin, const int Ntx, const double pitch, const double soundspeed, const double FreqFPGASim)
{
	for (int p = 0; p < SIGNAL_SIZE; p++)
	{
		tdmin[p] = 10;
		for (int i = 0; i < Ntx; i++)//64
		{
			/*
			td = sqrt(([1:Nrx]-(Nrx/2+0.5)).^2*(pitch/c)^2 + (p/(2*fs)).^2); % time delay [s]
			tdmin = min(td);
			*/
			tdf[i + (p*Ntx)] = sqrt((pow(((i+1) - (Ntx / 2 + 0.5)), 2) * pow((pitch / soundspeed), 2)) + pow((p + 1) / (2 * FreqFPGASim), 2)); // i start 1 not 0
			if (tdf[i + (p*Ntx) ] < tdmin[p])
				tdmin[p] = tdf[i + (p*Ntx)];
		}
	}
}

void tddsProcess(double *tdds, const int Nrx, const double *tdf, const double *tdmin, const double FreqFPGASim)
{
	/*
	tdd = tdf - tdmin; % time delay different from the center element (s)
	tdds = tdd.*FREQ_SAMPLING; % (sample)
	*/
	for (int p = 0; p < SIGNAL_SIZE; p++)
		for (int i = 0; i < Nrx; i++)//64
			tdds[i + (p*Nrx)] = (tdf[i + (p*Nrx)] - tdmin[p])*FreqFPGASim;
}

void induProcess(int *indu, const int Nrx, const double *elementRxs)
{
	for (int nl = 0; nl < SCAN_LINE; nl++)
		for (int i = 0; i < Nrx; i++) //32.
			indu[i + (nl * Nrx)] = (int)elementRxs[i + (nl * Nrx)] - nl - 1 + Nrx + 1; // index for tdfs
}

void tdrProcess(double *tdr, const int Nrx, const double *tdds, const int *indu, const double *t0)
{
	for (int nl = 0; nl < SCAN_LINE; nl++)
		for (int p = 0; p < SIGNAL_SIZE; p++)
			for (int i = 0; i < Nrx; i++) //32.
				tdr[i + (p * Nrx) + (nl * SIGNAL_SIZE * Nrx)] = round(t0[nl] + tdds[indu[i + (nl * Nrx)] - 1 + (p * 64)] + p);
}

void delaysum_beamforming(double *output, const double *tdr, const double *raw_signal)
{
	double sum = 0;
	int ttdr = 0;
	for (int nl = 0; nl < SCAN_LINE; nl++) //81 scanline
	{
		for (int p = 0; p < SIGNAL_SIZE; p++)//1 scanline 8192 point
		{
			sum = 0;
			for (int i = 0; i < CHANNEL; i++)//1 point = 32 channel
			{
				ttdr = tdr[i + (p * CHANNEL) + (nl * SIGNAL_SIZE * CHANNEL)];
				if (ttdr < SIGNAL_SIZE)
				{
					sum += raw_signal[ttdr + (i * SIGNAL_SIZE) + (nl * CHANNEL * SIGNAL_SIZE)];
				}
			}
			output[p + (nl * SIGNAL_SIZE)] = sum;
		}
	}
}

int main()
{
	int    dataLength = SIGNAL_SIZE * SCAN_LINE;
	int	   *tdfindex = new int[CHANNEL * SIGNAL_SIZE];
	double *raw_datal = new double[SIGNAL_SIZE * CHANNEL * SCAN_LINE];
	double *max_ps_delay = new double[SCAN_LINE];
	double *elementRxs = new double[CHANNEL * SCAN_LINE];
	double *t0 = new double[SCAN_LINE];
	double *tdf = new double[2 * CHANNEL * SIGNAL_SIZE];
	double *tdds = new double[2 * CHANNEL * SIGNAL_SIZE];
	double *tdmin = new double[SIGNAL_SIZE];
	double *tdr = new double[SIGNAL_SIZE * CHANNEL * SCAN_LINE];
	double *vout = new double[SIGNAL_SIZE * SCAN_LINE];
	
	loadRawData("D:\\data.dat", raw_datal); // channel*scanline size
	loadData("D:\\loadPsDelay.dat", SCAN_LINE, max_ps_delay);
	loadElementRxs("D:\\loadElementRxs.dat", elementRxs); // channel*scanline size

	for (int i = 0; i < SCAN_LINE; i++) { *(t0 + i) = NBEFOREPULSE + *(max_ps_delay + i) / FREQ_FPGA_CLOCK * FREQ_SAMPLING; }
	calTimeDelay(tdf, tdmin, NTX * 2, PITCH, SOUND_SPEED, FREQ_SAMPLING); // TDF
	tddsProcess(tdds, NRX * 2, tdf, tdmin, FREQ_SAMPLING); //TDDS
	induProcess(tdfindex, NRX, elementRxs);// Index TDF
	tdrProcess(tdr, NRX, tdds, tdfindex, t0);//TDR
	delaysum_beamforming(vout, tdr, raw_datal);

	writeFile("D:\\save.dat", dataLength, vout); //output Vout

	delete tdfindex;
	delete raw_datal;
	delete max_ps_delay;
	delete elementRxs;
	delete t0;
	delete tdf;
	delete tdds;
	delete tdmin;
	delete tdr;
	delete vout;
}
