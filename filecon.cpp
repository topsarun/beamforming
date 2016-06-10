#include <iostream>
#include <fstream>
#include "filecon.h"
using namespace std;

#define SIGNAL_SIZE		8192	  
#define CHANNEL			32
#define SCAN_LINE		81

void loadRawData(const char* filename, double* readArray) {
	ifstream file(filename, ios::binary | ios::in);
	if (!file.is_open())
	{
		printf("Error can't open file Rawdata.\n");
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

void loadElementRxs(const char* filename, double* readArray) {
	ifstream file(filename, ios::binary | ios::in);
	if (!file.is_open())
	{
		printf("Error can't open file ElementRxs.\n");
		return;
	}
	double aDouble = 0;
	for (int j = 0; j < SCAN_LINE && !file.eof(); j++)
		for (int i = 0; i < CHANNEL; i++)
		{
			file.read((char*)(&aDouble), sizeof(double));
			readArray[i + (j*CHANNEL)] = aDouble;
		}
	file.close();
}

void loadData(const char* filename, int size, double* readArray) {
	ifstream file(filename, ios::binary | ios::in);
	if (!file.is_open())
	{
		printf("Error can't open file.\n");
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

void writeFileRawData(const char *filename, double* readArray)
{
	ofstream output(filename, std::ios::binary | std::ios::out);
	for (int k = 0; k < SCAN_LINE; k++)
		for (int j = 0; j < CHANNEL; j++)
			for (int i = 0; i < SIGNAL_SIZE; i++)
				output.write((char *)&readArray[i + (j*SIGNAL_SIZE) + (k*SIGNAL_SIZE*CHANNEL)], sizeof(double));
	output.close();
}

void writeFileElementRxs(const char *filename, double* readArray)
{
	ofstream output(filename, std::ios::binary | std::ios::out);
	for (int j = 0; j < SCAN_LINE; j++)
		for (int i = 0; i < CHANNEL; i++)
			output.write((char *)&readArray[i + (j*CHANNEL)], sizeof(double));
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