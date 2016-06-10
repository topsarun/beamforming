#ifndef FILECON
#define FILECON

void loadRawData(const char* filename, double* readArray);
void loadElementRxs(const char* filename, double* readArray);
void loadData(const char* filename, int size, double* readArray);
void writeFileRawData(const char *filename, double* readArray);
void writeFileElementRxs(const char *filename, double* readArray);
void writeFile(const char *filename, const int size, double* readArray);

#endif 