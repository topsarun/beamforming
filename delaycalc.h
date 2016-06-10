#ifndef DELAYCALC
#define DELAYCALC

void calc_TimeDelay(double *tdf, double *tdmin, const int Ntx, const double pitch, const double soundspeed, const double FreqFPGASim);
void calc_tdds(double *tdds, const int Nrx, const double *tdf, const double *tdmin, const double FreqFPGASam);
void calc_tdfindex(int *indu, const int Nrx, const double *elementRxs);
void calc_tdr(double *tdr, const int Nrx, const double *tdds, const int *indu, const double *t0);

#endif