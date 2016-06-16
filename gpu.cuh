#ifndef GPU
#define GPU

__global__ void beamforming1scanline(int nl, float2 *vout, const double *tdr, const __int16 *raw_data);
__global__ void improve(float2 *vout, const double *tdr, const __int16 *raw_data);
__global__ void hilbert_1line_step2(float2 *signal);
__global__ void hilbert_step2(float2 *signal);
__global__ void abscomplex(double *env, float2 *signal);
__global__ void Gpu_median_filter(double *Input_Image, double *Output_Image, int img_h, int img_w);
__global__ void find_maximum(double *array, double *max, int *mutex, unsigned int n);
__global__ void logCompressDB(double *env, double *d_max);
__global__ void FilterCalc(float2 *signal, float2 *in, float2 *filter, int nl);
__global__ void FilterCalcImprove1(float2 *signal, float2 *in, float2 *filter);

#endif