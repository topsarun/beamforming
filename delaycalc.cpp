#include <iostream>
#include "delaycalc.h"

using namespace std;
#define SIGNAL_SIZE		8192	  
#define CHANNEL			32
#define SCAN_LINE		81
#define FREQ_SAMPLING	40000000  //40Mhz
#define M_PI 3.14159265358979323846

void calc_TimeDelay(double *tdf, double *tdmin, const int Ntx, const double pitch, const double soundspeed, const double FreqFPGASim)
{
	for (int p = 0; p < SIGNAL_SIZE; p++)
	{
		tdmin[p] = 10;
		for (int i = 0; i < Ntx; i++)//64
		{
			/*
			td (time delay to pick a sample in the data)
			td = sqrt(([1:Nrx]-(Nrx/2+0.5)).^2*(pitch/c)^2 + (p/(2*fs)).^2); % time delay [s]
			tdmin = min(td);
			*/
			tdf[i + (p*Ntx)] = sqrt((pow(((i + 1) - (Ntx / 2 + 0.5)), 2) * pow((pitch / soundspeed), 2)) + pow((p + 1) / (2 * FreqFPGASim), 2)); // i start 1 not 0
			if (tdf[i + (p*Ntx)] < tdmin[p])
				tdmin[p] = tdf[i + (p*Ntx)];
		}
	}
}

void calc_tdds(double *tdds, const int Nrx, const double *tdf, const double *tdmin, const double FreqFPGASam)
{
	/*
	tdd = tdf - tdmin; % time delay different from the center element (s)
	tdds = tdd.*FREQ_SAMPLING; % (sample)
	*/
	for (int p = 0; p < SIGNAL_SIZE; p++)
		for (int i = 0; i < Nrx; i++)//64
			tdds[i + (p*Nrx)] = (tdf[i + (p*Nrx)] - tdmin[p])*FreqFPGASam;
}

void calc_tdfindex(int *tdfindex, const int Nrx, const double *elementRxs)
{
	for (int nl = 0; nl < SCAN_LINE; nl++)
		for (int i = 0; i < Nrx; i++)//32.
			tdfindex[i + (nl * Nrx)] = (int)elementRxs[i + (nl * Nrx)] - nl - 1 + Nrx + 1; // index for tdfs
}

void calc_tdr(double *tdr, const int Nrx, const double *tdds, const int *indu, const double *t0)
{
	//round(td)
	for (int k = 0; k < SCAN_LINE; k++)
		for (int j = 0; j < SIGNAL_SIZE; j++)
			for (int i = 0; i < Nrx; i++)//32.
				tdr[i + (j * Nrx) + (k * SIGNAL_SIZE * Nrx)] = round(t0[k] + tdds[indu[i + (k * Nrx)] - 1 + (j * 64)] + j);
}