#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <fstream>
#include <iostream>
#include <ctime>

//Cuda header
#include <cuda_runtime.h>
#include <cufft.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#define FREQ_FPGA_CLOCK 100000000
#define FREQ_SAMPLING	40000000
#define NBEFOREPULSE	538
#define PITCH			0.0005
#define SOUND_SPEED		1540
#define CHANNEL			32
#define SIGNAL_SIZE		8192
#define SCAN_LINE		81
#define NRX				32

using namespace std;

void loadRawData(const char* filename, double* readArray) {
	ifstream file(filename, ios::binary | ios::in);
	if (!file.is_open())
	{
		cout << "Cannot open file." << endl;
		return;
	}
	double aDouble = 0;
	for (int k = 0; k< SCAN_LINE && !file.eof(); k++)
	{
		for (int j = 0; j < CHANNEL; j++)
		{
			for (int i = 0; i < SIGNAL_SIZE; i++)
			{
				file.read((char*)(&aDouble), sizeof(double));
				readArray[i + (j*SIGNAL_SIZE) + (k*SIGNAL_SIZE*CHANNEL)] = aDouble;
			}
		}
	}
	file.close();
}

void loadData(const char* filename, int size, double* readArray) {
	ifstream file(filename, ios::binary | ios::in);
	if (!file.is_open())
	{
		cout << "Cannot open file." << endl;
		return;
	}
	for (int i = 0; i<size && !file.eof(); i++)
	{
		double aDouble = 0;
		file.read((char*)(&aDouble), sizeof(double));
		readArray[i] = aDouble;
	}
	file.close();
}

void loadElementRxs(const char* filename, double* readArray) {
	ifstream file(filename, ios::binary | ios::in);
	if (!file.is_open())
	{
		cout << "Cannot open file." << endl;
		return;
	}
	double aDouble = 0;
	for (int j = 0; j<SCAN_LINE && !file.eof(); j++)
	{
		for (int i = 0; i < CHANNEL; i++)
		{
			file.read((char*)(&aDouble), sizeof(double));
			readArray[i + (j*CHANNEL)] = aDouble;
		}
	}
	file.close();
}

void writeFileRawData(const char *filename, double* readArray)
{
	ofstream output(filename, std::ios::binary | std::ios::out);
	for (int k = 0; k < SCAN_LINE; k++)
	{
		for (int j = 0; j < CHANNEL; j++)
		{
			for (int i = 0; i < SIGNAL_SIZE; i++)
			{
				output.write((char *)&readArray[i + (j*SIGNAL_SIZE) + (k*SIGNAL_SIZE*CHANNEL)], sizeof(double));
			}
		}
	}
	output.close();
}

void writeFile(const char *filename, const int size, double* readArray)
{
	ofstream output(filename, std::ios::binary | std::ios::out);
	for (int i = 0; i < size; i++)
	{
		output.write((char *)&readArray[i], sizeof(double));
	}
	output.close();
}

void writeFileElementRxs(const char *filename, double* readArray)
{
	ofstream output(filename, std::ios::binary | std::ios::out);
	for (int j = 0; j < SCAN_LINE; j++)
	{
		for (int i = 0; i < CHANNEL; i++)
		{
			output.write((char *)&readArray[i + (j*CHANNEL)], sizeof(double));
		}
	}
	output.close();
}

void calTimeDelay(double *tdf, double *tdmin, const int N, const double pitch, const double c, const double fs)
{
	for (int p = 0; p < SIGNAL_SIZE; p++)
	{
		tdmin[p] = 10;
		for (int i = 0; i < N; i++)//64
		{
			/*
			td = sqrt(([1:Nrx]-(Nrx/2+0.5)).^2*(pitch/c)^2 + (p/(2*fs)).^2); % time delay [s]
			tdmin = min(td);
			*/
			tdf[i + (p*N)] = sqrt((pow(((i + 1) - (N / 2 + 0.5)), 2) * pow((pitch / c), 2)) + pow((p + 1) / (2 * fs), 2));
			if (tdf[i + (p*N)] < tdmin[p])
			{
				tdmin[p] = tdf[i + (p*N)];
			}

		}
	}
}
void tddsProcess(double *tdds, const int N, const double *tdf, const double *tdmin, const double fs)
{
	for (int p = 0; p < SIGNAL_SIZE; p++)
	{
		for (int i = 0; i < N; i++)//64
		{
			/*
			tdd = tdf - tdmin; % time delay different from the center element (s)
			tdds = tdd.*FREQ_SAMPLING; % (sample)
			*/
			tdds[i + (p*N)] = (tdf[i + (p*N)] - tdmin[p])*fs;
		}
	}
}
void induProcess(int *indu, const int N, const double *elementRxs)
{
	for (int nl = 0; nl < SCAN_LINE; nl++)
	{
		for (int i = 0; i < N; i++) // 32
		{
			indu[i + (nl * N)] = (int)elementRxs[i + (nl * N)] - nl - 1 + N + 1; // index for tdfs
		}
	}
}

void tdrProcess(double *tdr, const int N, const double *tdds, const int *indu, const double *t0)
{
	for (int nl = 0; nl < SCAN_LINE; nl++)
	{
		for (int p = 0; p < SIGNAL_SIZE; p++)
		{
			for (int i = 0; i < N; i++)// 32
			{
				tdr[i + (p * N) + (nl * SIGNAL_SIZE * N)] = round(t0[nl] + tdds[indu[i + (nl * N)] - 1 + (p * 64)] + p);
			}
		}
	}
}

void beamformingProcessingCPU(double *vout, const double *tdr, const double *raw_signal)
{
	double sum;
	int ttdr;
	for (int nl = 0; nl < SCAN_LINE; nl++)
	{
		for (int p = 0; p < SIGNAL_SIZE; p++)
		{
			sum = 0;
			for (int i = 0; i < CHANNEL; i++)
			{
				ttdr = tdr[i + (p * CHANNEL) + (nl * SIGNAL_SIZE * CHANNEL)];
				if (ttdr < SIGNAL_SIZE)
				{
					sum += raw_signal[ttdr + (i * SIGNAL_SIZE) + (nl * CHANNEL * SIGNAL_SIZE)];
				}
			}
			vout[p + (nl * SIGNAL_SIZE)] = sum;
		}
	}
}

int main()
{
	int	   *indu = new int[CHANNEL * SIGNAL_SIZE];
	double *raw_signal = new double[SIGNAL_SIZE * CHANNEL * SCAN_LINE];
	double *max_ps_delay = new double[SCAN_LINE];
	double *elementRxs = new double[CHANNEL * SCAN_LINE];
	double *t0 = new double[SCAN_LINE];
	double *tdf = new double[2 * CHANNEL * SIGNAL_SIZE];
	double *tdds = new double[2 * CHANNEL * SIGNAL_SIZE];
	double *tdmin = new double[SIGNAL_SIZE];
	double *tdr = new double[SIGNAL_SIZE * CHANNEL * SCAN_LINE];
	double *vout = new double[SIGNAL_SIZE * SCAN_LINE];

	loadRawData("D:\\data.dat", raw_signal); // channel*scanline size
	loadData("D:\\loadPsDelay.dat", SCAN_LINE, max_ps_delay);
	loadElementRxs("D:\\loadElementRxs.dat", elementRxs); // channel*scanline size

	for (int i = 0; i < SCAN_LINE; i++) { *(t0 + i) = NBEFOREPULSE + *(max_ps_delay + i) / FREQ_FPGA_CLOCK * FREQ_SAMPLING; }
	calTimeDelay(tdf, tdmin, NRX * 2, PITCH, SOUND_SPEED, FREQ_SAMPLING); // TDF
	tddsProcess(tdds, NRX * 2, tdf, tdmin, FREQ_SAMPLING); //TDDS
	induProcess(indu, NRX, elementRxs);// Index TDF
	tdrProcess(tdr, NRX, tdds, indu, t0);//TDR
	beamformingProcessingCPU(vout, tdr, raw_signal);

	writeFile("D:\\save.dat", 8192 * 81, vout); //output Vout

	delete indu;
	delete raw_signal;
	delete max_ps_delay;
	delete elementRxs;
	delete t0;
	delete tdf;
	delete tdds;
	delete tdmin;
	delete tdr;
	delete vout;
}
