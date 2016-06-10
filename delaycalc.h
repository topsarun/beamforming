#ifndef DELAYCALC
#define DELAYCALC

void calTimeDelay(double *tdf, double *tdmin, const int Ntx, const double pitch, const double soundspeed, const double FreqFPGASim);
void tddsProcess(double *tdds, const int Nrx, const double *tdf, const double *tdmin, const double FreqFPGASam);
void tdfindexProcess(int *indu, const int Nrx, const double *elementRxs);
void tdrProcess(double *tdr, const int Nrx, const double *tdds, const int *indu, const double *t0);

#endif