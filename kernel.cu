#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <time.h>
#include <string>
using namespace std;

/*
*  Global Configuration
*/
#define PARALLEL2          // {SERIAL, PARALLEL1, PARALLEL2}
#define TIME            // measures time

#define NUMBER_OF_TRACES 10000
#define POINTS_PER_TRACE 10000
#define NUMBER_OF_TEXTS 10000
#define BYTES_PER_TEXT 16     // fix for AES
#define BYTES_PER_KEY 16       // possible values for AES are 16, 24 and 32 (128, 192 or 256 bits)
#define NUMBER_OF_KEY_CANDIDATES 256

#define TRACE_STARTPOINT 550
#define TRACE_ENDPOINT 600

string TRACE_FILE = "Traces00000.dat";
string PLAINTEXT_FILE = "plaintexts.dat";
string CIPHERTEXT_FILE = "ciphertexts.dat";

/*

CORRECT CIPHER KEY: 2b  7e  15  16  28  ae  d2  a6  ab  f7  15  88  09  cf  4f  3c
dec: 043 126 021 022 040 174 210 166 171 247 021 136 009 207 079 060		 
*/

cudaError_t getHwWithCuda(int *hw, int *plaintext);
cudaError_t computeCoeffWithCuda(double *cc, int *traces, int *hw);
cudaError_t computeCoeffWithCuda_Unrolled(double *cc, int *traces, int *hw);

// #################### CURRENTLY UNUSED FUNCTIONS #####################

void printDevProp(cudaDeviceProp devProp)
{
    printf("Major revision number:         %d\n",  devProp.major);
    printf("Minor revision number:         %d\n",  devProp.minor);
    printf("Name:                          %s\n",  devProp.name);
    printf("Total global memory:           %u\n",  devProp.totalGlobalMem);
    printf("Total shared memory per block: %u\n",  devProp.sharedMemPerBlock);
    printf("Total registers per block:     %d\n",  devProp.regsPerBlock);
    printf("Warp size:                     %d\n",  devProp.warpSize);
    printf("Maximum memory pitch:          %u\n",  devProp.memPitch);
    printf("Maximum threads per block:     %d\n",  devProp.maxThreadsPerBlock);
    for (int i = 0; i < 3; ++i)
        printf("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
    for (int i = 0; i < 3; ++i)
        printf("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
    printf("Clock rate:                    %d\n",  devProp.clockRate);
    printf("Total constant memory:         %u\n",  devProp.totalConstMem);
    printf("Texture alignment:             %u\n",  devProp.textureAlignment);
    printf("Concurrent copy and execution: %s\n",  (devProp.deviceOverlap ? "Yes" : "No"));
    printf("Number of multiprocessors:     %d\n",  devProp.multiProcessorCount);
    printf("Kernel execution timeout:      %s\n",  (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
    return;
}

void printDevInfos()
{
    // Number of CUDA devices
    int devCount;
    cudaGetDeviceCount(&devCount);
    printf("CUDA Device Query...\n");
    printf("There are %d CUDA devices.\n", devCount);

    // Iterate through devices
    for (int i = 0; i < devCount; ++i)
    {
        // Get device properties
        printf("\nCUDA Device #%d\n", i);
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, i);
        printDevProp(devProp);
    }

    printf("\nPress any key to exit...");
    char c;
    scanf("%c", &c);
}


// #################### ACTUAL PROGRAM #####################

/*
*	Function to read in values of traces
*/
void read_traces(int *traces, string filename) {
    streampos size;
    char * memblock;

    ifstream file (filename, ios::in|ios::binary|ios::ate);
    if (file.is_open())
    {
        size = file.tellg();
        memblock = new char [size];
        file.seekg (0, ios::beg);
        file.read (memblock, size);
        file.close();

        std::cout << "Content of file " << filename << " is in memory." << endl;   
    }
    else
    {
        std::cout << "Unable to open file" << filename << endl;
        return;
    }

    for (int i = 0; i < NUMBER_OF_TRACES*POINTS_PER_TRACE; i++)
    {
        traces[i] = static_cast<int>(memblock[i]);   // easier this way, so we don't need the array traces_at_tracepoint
    }

    delete[] memblock;
}


/*
*	Function to read in plaintexts or ciphertexts
*/
void read_texts(unsigned _int8 **texts, string filename) {
    streampos size;
    char * memblock;

    ifstream file (filename, ios::in|ios::binary|ios::ate);
    if (file.is_open())
    {
        size = file.tellg();
        memblock = new char [size];
        file.seekg (0, ios::beg);
        file.read (memblock, size);
        file.close();

        std::cout << "Content of file " << filename << " is in memory." << endl;    
    }
    else
    {
        std::cout << "Unable to open file " << filename << endl;
        return;
    }

    for (int j = 0; j < BYTES_PER_TEXT; j++)
    {
        for (int i = 0; i < NUMBER_OF_TEXTS; i++)
        {
            texts[i][j] = memblock[j*NUMBER_OF_TEXTS + i];
        }
    }
    delete[] memblock;
}


/*
*	Function that computes the hamming weight
*/
unsigned int get_Hw(unsigned int b)
{
    unsigned int hw = 0;
    while (b) 
    {
        hw += (b & 1);
        b >>= 1;
    }
    return hw;
}


/*
*	Function that computes the T-Table output for a plaintext-byte and a key candidate
*/
unsigned int get_TTable_Out(unsigned int plaintext_byte, unsigned int key_candidate)
{
    // AES T-Table LUT
    unsigned int ttable0 [] = 
    { 0xC66363A5, 0xF87C7C84, 0xEE777799, 0xF67B7B8D, 0xFFF2F20D, 0xD66B6BBD, 0xDE6F6FB1, 0x91C5C554, 
    0x60303050, 0x02010103, 0xCE6767A9, 0x562B2B7D, 0xE7FEFE19, 0xB5D7D762, 0x4DABABE6, 0xEC76769A, 
    0x8FCACA45, 0x1F82829D, 0x89C9C940, 0xFA7D7D87, 0xEFFAFA15, 0xB25959EB, 0x8E4747C9, 0xFBF0F00B, 
    0x41ADADEC, 0xB3D4D467, 0x5FA2A2FD, 0x45AFAFEA, 0x239C9CBF, 0x53A4A4F7, 0xE4727296, 0x9BC0C05B, 
    0x75B7B7C2, 0xE1FDFD1C, 0x3D9393AE, 0x4C26266A, 0x6C36365A, 0x7E3F3F41, 0xF5F7F702, 0x83CCCC4F, 
    0x6834345C, 0x51A5A5F4, 0xD1E5E534, 0xF9F1F108, 0xE2717193, 0xABD8D873, 0x62313153, 0x2A15153F, 
    0x0804040C, 0x95C7C752, 0x46232365, 0x9DC3C35E, 0x30181828, 0x379696A1, 0x0A05050F, 0x2F9A9AB5, 
    0x0E070709, 0x24121236, 0x1B80809B, 0xDFE2E23D, 0xCDEBEB26, 0x4E272769, 0x7FB2B2CD, 0xEA75759F, 
    0x1209091B, 0x1D83839E, 0x582C2C74, 0x341A1A2E, 0x361B1B2D, 0xDC6E6EB2, 0xB45A5AEE, 0x5BA0A0FB, 
    0xA45252F6, 0x763B3B4D, 0xB7D6D661, 0x7DB3B3CE, 0x5229297B, 0xDDE3E33E, 0x5E2F2F71, 0x13848497, 
    0xA65353F5, 0xB9D1D168, 0x00000000, 0xC1EDED2C, 0x40202060, 0xE3FCFC1F, 0x79B1B1C8, 0xB65B5BED, 
    0xD46A6ABE, 0x8DCBCB46, 0x67BEBED9, 0x7239394B, 0x944A4ADE, 0x984C4CD4, 0xB05858E8, 0x85CFCF4A, 
    0xBBD0D06B, 0xC5EFEF2A, 0x4FAAAAE5, 0xEDFBFB16, 0x864343C5, 0x9A4D4DD7, 0x66333355, 0x11858594, 
    0x8A4545CF, 0xE9F9F910, 0x04020206, 0xFE7F7F81, 0xA05050F0, 0x783C3C44, 0x259F9FBA, 0x4BA8A8E3, 
    0xA25151F3, 0x5DA3A3FE, 0x804040C0, 0x058F8F8A, 0x3F9292AD, 0x219D9DBC, 0x70383848, 0xF1F5F504, 
    0x63BCBCDF, 0x77B6B6C1, 0xAFDADA75, 0x42212163, 0x20101030, 0xE5FFFF1A, 0xFDF3F30E, 0xBFD2D26D, 
    0x81CDCD4C, 0x180C0C14, 0x26131335, 0xC3ECEC2F, 0xBE5F5FE1, 0x359797A2, 0x884444CC, 0x2E171739, 
    0x93C4C457, 0x55A7A7F2, 0xFC7E7E82, 0x7A3D3D47, 0xC86464AC, 0xBA5D5DE7, 0x3219192B, 0xE6737395, 
    0xC06060A0, 0x19818198, 0x9E4F4FD1, 0xA3DCDC7F, 0x44222266, 0x542A2A7E, 0x3B9090AB, 0x0B888883, 
    0x8C4646CA, 0xC7EEEE29, 0x6BB8B8D3, 0x2814143C, 0xA7DEDE79, 0xBC5E5EE2, 0x160B0B1D, 0xADDBDB76, 
    0xDBE0E03B, 0x64323256, 0x743A3A4E, 0x140A0A1E, 0x924949DB, 0x0C06060A, 0x4824246C, 0xB85C5CE4, 
    0x9FC2C25D, 0xBDD3D36E, 0x43ACACEF, 0xC46262A6, 0x399191A8, 0x319595A4, 0xD3E4E437, 0xF279798B, 
    0xD5E7E732, 0x8BC8C843, 0x6E373759, 0xDA6D6DB7, 0x018D8D8C, 0xB1D5D564, 0x9C4E4ED2, 0x49A9A9E0, 
    0xD86C6CB4, 0xAC5656FA, 0xF3F4F407, 0xCFEAEA25, 0xCA6565AF, 0xF47A7A8E, 0x47AEAEE9, 0x10080818, 
    0x6FBABAD5, 0xF0787888, 0x4A25256F, 0x5C2E2E72, 0x381C1C24, 0x57A6A6F1, 0x73B4B4C7, 0x97C6C651, 
    0xCBE8E823, 0xA1DDDD7C, 0xE874749C, 0x3E1F1F21, 0x964B4BDD, 0x61BDBDDC, 0x0D8B8B86, 0x0F8A8A85, 
    0xE0707090, 0x7C3E3E42, 0x71B5B5C4, 0xCC6666AA, 0x904848D8, 0x06030305, 0xF7F6F601, 0x1C0E0E12, 
    0xC26161A3, 0x6A35355F, 0xAE5757F9, 0x69B9B9D0, 0x17868691, 0x99C1C158, 0x3A1D1D27, 0x279E9EB9, 
    0xD9E1E138, 0xEBF8F813, 0x2B9898B3, 0x22111133, 0xD26969BB, 0xA9D9D970, 0x078E8E89, 0x339494A7, 
    0x2D9B9BB6, 0x3C1E1E22, 0x15878792, 0xC9E9E920, 0x87CECE49, 0xAA5555FF, 0x50282878, 0xA5DFDF7A, 
    0x038C8C8F, 0x59A1A1F8, 0x09898980, 0x1A0D0D17, 0x65BFBFDA, 0xD7E6E631, 0x844242C6, 0xD06868B8, 
    0x824141C3, 0x299999B0, 0x5A2D2D77, 0x1E0F0F11, 0x7BB0B0CB, 0xA85454FC, 0x6DBBBBD6, 0x2C16163A };

    return ttable0[plaintext_byte ^ key_candidate];
}


/*
* Kernel that computes the Hamming Weight
*/
__global__ void getHwKernel(int *hw, int *plaintext)
{
    int idx = blockDim.x*blockIdx.x + threadIdx.x;

    int key_candidate = idx / NUMBER_OF_TRACES;

    // AES T-Table LUT
    unsigned int ttable0 [] = 
    { 0xC66363A5, 0xF87C7C84, 0xEE777799, 0xF67B7B8D, 0xFFF2F20D, 0xD66B6BBD, 0xDE6F6FB1, 0x91C5C554, 
    0x60303050, 0x02010103, 0xCE6767A9, 0x562B2B7D, 0xE7FEFE19, 0xB5D7D762, 0x4DABABE6, 0xEC76769A, 
    0x8FCACA45, 0x1F82829D, 0x89C9C940, 0xFA7D7D87, 0xEFFAFA15, 0xB25959EB, 0x8E4747C9, 0xFBF0F00B, 
    0x41ADADEC, 0xB3D4D467, 0x5FA2A2FD, 0x45AFAFEA, 0x239C9CBF, 0x53A4A4F7, 0xE4727296, 0x9BC0C05B, 
    0x75B7B7C2, 0xE1FDFD1C, 0x3D9393AE, 0x4C26266A, 0x6C36365A, 0x7E3F3F41, 0xF5F7F702, 0x83CCCC4F, 
    0x6834345C, 0x51A5A5F4, 0xD1E5E534, 0xF9F1F108, 0xE2717193, 0xABD8D873, 0x62313153, 0x2A15153F, 
    0x0804040C, 0x95C7C752, 0x46232365, 0x9DC3C35E, 0x30181828, 0x379696A1, 0x0A05050F, 0x2F9A9AB5, 
    0x0E070709, 0x24121236, 0x1B80809B, 0xDFE2E23D, 0xCDEBEB26, 0x4E272769, 0x7FB2B2CD, 0xEA75759F, 
    0x1209091B, 0x1D83839E, 0x582C2C74, 0x341A1A2E, 0x361B1B2D, 0xDC6E6EB2, 0xB45A5AEE, 0x5BA0A0FB, 
    0xA45252F6, 0x763B3B4D, 0xB7D6D661, 0x7DB3B3CE, 0x5229297B, 0xDDE3E33E, 0x5E2F2F71, 0x13848497, 
    0xA65353F5, 0xB9D1D168, 0x00000000, 0xC1EDED2C, 0x40202060, 0xE3FCFC1F, 0x79B1B1C8, 0xB65B5BED, 
    0xD46A6ABE, 0x8DCBCB46, 0x67BEBED9, 0x7239394B, 0x944A4ADE, 0x984C4CD4, 0xB05858E8, 0x85CFCF4A, 
    0xBBD0D06B, 0xC5EFEF2A, 0x4FAAAAE5, 0xEDFBFB16, 0x864343C5, 0x9A4D4DD7, 0x66333355, 0x11858594, 
    0x8A4545CF, 0xE9F9F910, 0x04020206, 0xFE7F7F81, 0xA05050F0, 0x783C3C44, 0x259F9FBA, 0x4BA8A8E3, 
    0xA25151F3, 0x5DA3A3FE, 0x804040C0, 0x058F8F8A, 0x3F9292AD, 0x219D9DBC, 0x70383848, 0xF1F5F504, 
    0x63BCBCDF, 0x77B6B6C1, 0xAFDADA75, 0x42212163, 0x20101030, 0xE5FFFF1A, 0xFDF3F30E, 0xBFD2D26D, 
    0x81CDCD4C, 0x180C0C14, 0x26131335, 0xC3ECEC2F, 0xBE5F5FE1, 0x359797A2, 0x884444CC, 0x2E171739, 
    0x93C4C457, 0x55A7A7F2, 0xFC7E7E82, 0x7A3D3D47, 0xC86464AC, 0xBA5D5DE7, 0x3219192B, 0xE6737395, 
    0xC06060A0, 0x19818198, 0x9E4F4FD1, 0xA3DCDC7F, 0x44222266, 0x542A2A7E, 0x3B9090AB, 0x0B888883, 
    0x8C4646CA, 0xC7EEEE29, 0x6BB8B8D3, 0x2814143C, 0xA7DEDE79, 0xBC5E5EE2, 0x160B0B1D, 0xADDBDB76, 
    0xDBE0E03B, 0x64323256, 0x743A3A4E, 0x140A0A1E, 0x924949DB, 0x0C06060A, 0x4824246C, 0xB85C5CE4, 
    0x9FC2C25D, 0xBDD3D36E, 0x43ACACEF, 0xC46262A6, 0x399191A8, 0x319595A4, 0xD3E4E437, 0xF279798B, 
    0xD5E7E732, 0x8BC8C843, 0x6E373759, 0xDA6D6DB7, 0x018D8D8C, 0xB1D5D564, 0x9C4E4ED2, 0x49A9A9E0, 
    0xD86C6CB4, 0xAC5656FA, 0xF3F4F407, 0xCFEAEA25, 0xCA6565AF, 0xF47A7A8E, 0x47AEAEE9, 0x10080818, 
    0x6FBABAD5, 0xF0787888, 0x4A25256F, 0x5C2E2E72, 0x381C1C24, 0x57A6A6F1, 0x73B4B4C7, 0x97C6C651, 
    0xCBE8E823, 0xA1DDDD7C, 0xE874749C, 0x3E1F1F21, 0x964B4BDD, 0x61BDBDDC, 0x0D8B8B86, 0x0F8A8A85, 
    0xE0707090, 0x7C3E3E42, 0x71B5B5C4, 0xCC6666AA, 0x904848D8, 0x06030305, 0xF7F6F601, 0x1C0E0E12, 
    0xC26161A3, 0x6A35355F, 0xAE5757F9, 0x69B9B9D0, 0x17868691, 0x99C1C158, 0x3A1D1D27, 0x279E9EB9, 
    0xD9E1E138, 0xEBF8F813, 0x2B9898B3, 0x22111133, 0xD26969BB, 0xA9D9D970, 0x078E8E89, 0x339494A7, 
    0x2D9B9BB6, 0x3C1E1E22, 0x15878792, 0xC9E9E920, 0x87CECE49, 0xAA5555FF, 0x50282878, 0xA5DFDF7A, 
    0x038C8C8F, 0x59A1A1F8, 0x09898980, 0x1A0D0D17, 0x65BFBFDA, 0xD7E6E631, 0x844242C6, 0xD06868B8, 
    0x824141C3, 0x299999B0, 0x5A2D2D77, 0x1E0F0F11, 0x7BB0B0CB, 0xA85454FC, 0x6DBBBBD6, 0x2C16163A };

    unsigned int b = ttable0[plaintext[idx % NUMBER_OF_TRACES]^key_candidate];

    hw[idx] = 0;
    while (b) 
    {
        hw[idx] += (b & 1);
        b >>= 1;
    }
}

/*
*	First kernel for computation of the Pearson Correlation Coefficient
*/
__global__ void CorrCoefKernel_Naiv(double *result, int *x, int *y, int first_col)
{
    int i = threadIdx.x + first_col;

    _Uint32t sum_x  = 0;
    _Uint32t sum_y  = 0;

    for(int j = 0; j < NUMBER_OF_TRACES; j++)
    {
        sum_x  += x[i+j*POINTS_PER_TRACE];
        sum_y  += y[j];
    }

    double x_average = sum_x/NUMBER_OF_TRACES;
    double y_average = sum_y/NUMBER_OF_TRACES;

    double dividend = 0;
    double divisor1 = 0;      
    double divisor2 = 0;

    for(int j = 0; j < NUMBER_OF_TRACES; j++)
    {
        dividend += (x[i+j*POINTS_PER_TRACE] - x_average)*(y[j] - y_average); 
        divisor1 += (x[i+j*POINTS_PER_TRACE] - x_average)*(x[i+j*POINTS_PER_TRACE] - x_average);  
        divisor2 += (y[j] - y_average)*(y[j] - y_average); 
    }

    double divisor = sqrt(divisor1)*sqrt(divisor2);

    if ((dividend == 0) || (divisor == 0))
    {
        result[i-first_col] = 0.0;
    }else{
        result[i-first_col] = dividend/divisor;
    }		
}


/*
*	Unrolled kernel for computation of the Pearson Correlation Coefficient
*/
__global__ void CorrCoefKernel_Unrolled(float *result, int *x, int *y)
{ 
    int trace_point = blockIdx.x;   //gridDim.x = (TRACE_ENDPOINT-TRACE_STARTPOINT)
    int key_candidate = threadIdx.x; //blockDim.x = NUMBER_OF_KEY_CANDIDATES

    _Uint32t sum_x  = 0;
    _Uint32t sum_y  = 0;

    int Yl[NUMBER_OF_TRACES];

    for(int j = 0; j < NUMBER_OF_TRACES; j++)
    {
        Yl[j] = y[key_candidate*NUMBER_OF_TRACES + j];
        sum_x  += x[trace_point*NUMBER_OF_TRACES+j];
        sum_y  += Yl[j];
    }

    float x_average = sum_x/NUMBER_OF_TRACES;
    float y_average = sum_y/NUMBER_OF_TRACES;

    float dividend = 0;
    float divisor1 = 0;      
    float divisor2 = 0;

    for(int j = 0; j < NUMBER_OF_TRACES; j++)
    {
        dividend += (x[trace_point*NUMBER_OF_TRACES+j] - x_average)*(Yl[j] - y_average); 
        divisor1 += (x[trace_point*NUMBER_OF_TRACES+j] - x_average)*(x[trace_point*NUMBER_OF_TRACES+j] - x_average);  
        divisor2 += (Yl[j] - y_average)*(Yl[j] - y_average); 
    }

    float divisor = sqrt(divisor1)*sqrt(divisor2);

    if ((dividend == 0) || (divisor == 0))
    {
        result[key_candidate *  (TRACE_ENDPOINT-TRACE_STARTPOINT)+ trace_point] = 0.0;
    }else{
        result[key_candidate *  (TRACE_ENDPOINT-TRACE_STARTPOINT)+ trace_point] = dividend/divisor;
    }		
}


/*
* Function to calculate the Pearson Correlation Coefficient
*/
double get_Corr_Coef(int *x, int *y, int n)
{
    _Uint32t sum_x  = 0;
    _Uint32t sum_y  = 0;

    for(int i = 0; i < n; i++)
    {
        sum_x  += x[i];
        sum_y  += y[i];
    }

    long double x_average = sum_x/n;
    long double y_average = sum_y/n;

    long double dividend = 0;
    long double divisor1 = 0;
    long double divisor2 = 0;

    for(int i = 0; i < n; i++)
    {
        dividend += (x[i] - x_average)*(y[i] - y_average); 
        divisor1 += (x[i] - x_average)*(x[i] - x_average); 
        divisor2 += (y[i] - y_average)*(y[i] - y_average); 
    }

    long double divisor = sqrt(divisor1)*sqrt(divisor2);

    if ((dividend == 0) || (divisor == 0))
    {
        return 0.0;
    }else{
        return dividend/divisor;
    }		
}


// Helper function for using CUDA to calculate Hamming Weight in parallel.
cudaError_t getHwWithCuda(int *hw, int *plaintext)
{
    int *dev_plaintext = 0;
    int *dev_hw = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_hw, NUMBER_OF_TRACES * NUMBER_OF_KEY_CANDIDATES * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_plaintext, NUMBER_OF_TRACES * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_plaintext, plaintext, NUMBER_OF_TRACES * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    getHwKernel<<<ceil(NUMBER_OF_TRACES*NUMBER_OF_KEY_CANDIDATES/1024), 1024>>>(dev_hw, dev_plaintext);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "getHwKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching getHwKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(hw, dev_hw, NUMBER_OF_TRACES * NUMBER_OF_KEY_CANDIDATES * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_hw);
    cudaFree(dev_plaintext);

    return cudaStatus;
}

// Helper function for computation of the correlation coefficient using CUDA.
//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
cudaError_t computeCoeffWithCuda(double *cc, int *traces, int *hw)
{
    int *dev_traces = 0;
    int *dev_hw = 0;
    double *dev_cc = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_cc, (TRACE_ENDPOINT - TRACE_STARTPOINT) * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_hw, NUMBER_OF_TRACES * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_traces, NUMBER_OF_TRACES * POINTS_PER_TRACE * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_hw, hw, NUMBER_OF_TRACES * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_traces, traces, NUMBER_OF_TRACES * POINTS_PER_TRACE * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each tracepoint.
    CorrCoefKernel_Naiv<<<1, TRACE_ENDPOINT-TRACE_STARTPOINT>>>(dev_cc, dev_traces, dev_hw, TRACE_STARTPOINT);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CorrCoefKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching CorrCoefKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(cc, dev_cc, (TRACE_ENDPOINT - TRACE_STARTPOINT) * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_cc);
    cudaFree(dev_hw);
    cudaFree(dev_traces);

    return cudaStatus;
}

// Helper function for computation of the correlation coefficient using CUDA.
//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
cudaError_t computeCoeffWithCuda_Unrolled(float *cc, int *traces_at_trace_point, int *hw)
{
    int *dev_hw = 0;
    float *dev_cc = 0;
    int *dev_traces_at_tp = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_cc, NUMBER_OF_KEY_CANDIDATES * (TRACE_ENDPOINT - TRACE_STARTPOINT) * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_hw, NUMBER_OF_KEY_CANDIDATES * NUMBER_OF_TRACES * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_traces_at_tp, NUMBER_OF_TRACES * (TRACE_ENDPOINT - TRACE_STARTPOINT) * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_hw, hw, NUMBER_OF_KEY_CANDIDATES * NUMBER_OF_TRACES * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_traces_at_tp, traces_at_trace_point, NUMBER_OF_TRACES * (TRACE_ENDPOINT - TRACE_STARTPOINT) * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each tracepoint and each key candidate: #Threads = (TRACE_ENDPOINT-TRACE_STARTPOINT) * NUMBER_OF_KEY_CANDIDATES
    CorrCoefKernel_Unrolled<<<(TRACE_ENDPOINT-TRACE_STARTPOINT), NUMBER_OF_KEY_CANDIDATES>>>(dev_cc, dev_traces_at_tp, dev_hw);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CorrCoefKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching CorrCoefKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(cc, dev_cc, NUMBER_OF_KEY_CANDIDATES * (TRACE_ENDPOINT - TRACE_STARTPOINT) * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_cc);
    cudaFree(dev_hw);
    cudaFree(dev_traces_at_tp);

    return cudaStatus;
}


int main()
{
#ifdef TIME
    // Start measuring time
    const clock_t begin_time = clock();
#ifdef SERIAL
    float t_hw_serial = 0.0;
    float t_cc_serial = 0.0;
#elif defined PARALLEL1
    float t_hw_serial = 0.0;
    float t_cc_parallel1 = 0.0;
#elif defined PARALLEL2
    float t_hw_parallel = 0.0;
    float t_cc_parallel2 = 0.0;
#endif
#endif

    int *traces;
    traces = new int [POINTS_PER_TRACE*NUMBER_OF_TRACES];

    //// Read traces and store in array
    read_traces(traces, TRACE_FILE);

#ifdef PARALLEL2
#ifdef TIME
    const clock_t begin_traces_copy = clock();
#endif
    int *traces_at_trace_point = new int [NUMBER_OF_TRACES*(TRACE_ENDPOINT-TRACE_STARTPOINT)];
    for (int trace_point = TRACE_STARTPOINT; trace_point < TRACE_ENDPOINT; trace_point++)
        for (int j=0; j<NUMBER_OF_TRACES; j++)
            traces_at_trace_point[(trace_point-TRACE_STARTPOINT)*NUMBER_OF_TRACES+j] = traces[trace_point+j*POINTS_PER_TRACE];
#ifdef TIME
    t_cc_parallel2 += float( clock() - begin_traces_copy ) / CLOCKS_PER_SEC;
#endif
#endif
#ifdef TIME
    // Stop measuring time
    std::cout << float( clock () - begin_time ) /  CLOCKS_PER_SEC << "sec" << endl;

    // Start measuring time
    const clock_t begin_time_plaintexts = clock();
#endif
    // Initialize plaintext array
    unsigned _int8 **plaintexts;
    plaintexts = new unsigned _int8 *[NUMBER_OF_TEXTS];
    for (int i = 0; i < NUMBER_OF_TEXTS; i++)
    {
        plaintexts[i] = new unsigned _int8[BYTES_PER_TEXT];
    }

    // Read plaintexts and store in array
    read_texts(plaintexts, PLAINTEXT_FILE);

#ifdef TIME
    // Stop measuring time
    std::cout << float( clock () - begin_time_plaintexts ) /  CLOCKS_PER_SEC << "sec" << endl;


    // Start measuring time
    const clock_t begin_time_calculation = clock();
#endif
#if !defined PARALLEL2
    int *hw_sl;
    hw_sl = new int [NUMBER_OF_TRACES];
#endif

    int *key;
    key = new int [BYTES_PER_KEY];

#ifdef PARALLEL1
    // Initialize corr array
    double **corr;
    corr = new double *[NUMBER_OF_KEY_CANDIDATES];
    for (int i = 0; i < NUMBER_OF_KEY_CANDIDATES; i++)
    {
        corr[i] = new double[TRACE_ENDPOINT - TRACE_STARTPOINT];
    }
#endif

#ifdef TIME
    // Start measuring time
        const clock_t begin_time_hw_sl = clock();
#endif
    // Loop through all key bytes
    for (int key_byte = 0; key_byte < BYTES_PER_KEY; key_byte++)
    {
        std::cout << "Compute key byte " << key_byte << " ..." << endl;

        float highest_cc = -1.0;
        float cc = -1.0;

        // ################# HW PARALLEL ######################
#ifdef PARALLEL2
#ifdef TIME
        // Start measuring time
        const clock_t begin_time_hw_pl = clock();
#endif
        int *hw_pl;
        hw_pl = new int [NUMBER_OF_TRACES * NUMBER_OF_KEY_CANDIDATES];

        int* plaintext_at_key_byte;
        plaintext_at_key_byte = new int[NUMBER_OF_TRACES];

        for(int i = 0; i < NUMBER_OF_TRACES; i++)
        {
            plaintext_at_key_byte[i] = plaintexts[i][key_byte];
        }

        // Calculate Hamming Weight in parallel.
        cudaError_t cudaStatusHw = getHwWithCuda(hw_pl, plaintext_at_key_byte);
        if (cudaStatusHw != cudaSuccess) {
            fprintf(stderr, "getHwWithCuda failed!");
            return 1;
        }

#ifdef TIME
        // Stop measuring time
        /*std::cout << " HW parallel:  " << float( clock () - begin_time_hw_pl ) /  CLOCKS_PER_SEC << "sec" << endl;*/
        t_hw_parallel += float( clock () - begin_time_hw_pl ) / CLOCKS_PER_SEC;
        const clock_t begin_time_coeff_parallel2 = clock();
#endif
        // Calculate Correlation Coefficient in parallel
        float *corr_unrolled = new float [NUMBER_OF_KEY_CANDIDATES *  (TRACE_ENDPOINT - TRACE_STARTPOINT)];
        cudaError_t cudaStatus = computeCoeffWithCuda_Unrolled(corr_unrolled, traces_at_trace_point, hw_pl);
        if (cudaStatusHw != cudaSuccess) {
            fprintf(stderr, "computeCoeffWithCuda_Unrolled failed!");
            return 1;
        }
        /*std::cout << " Coeff parallel unrolled:  " << float( clock () - begin_time_coeff_parallel2 ) /  CLOCKS_PER_SEC << "sec" << endl;*/
#ifdef TIME
        t_cc_parallel2 += float( clock () - begin_time_coeff_parallel2 ) / CLOCKS_PER_SEC;
#endif
        delete[] hw_pl;
#endif


        // ##################### HW SERIELL #####################

        // Loop through all key candidates
        for (int key_candidate = 0; key_candidate < NUMBER_OF_KEY_CANDIDATES; key_candidate++)
        {   
#if !defined PARALLEL2
#ifdef TIME
            // Start measuring time
            const clock_t begin_time_hw2 = clock();
#endif
            // Measure hamming weight for every trace
            for (int trace = 0; trace < NUMBER_OF_TRACES; trace++)
            {
                // Calculate the hamming weight
                hw_sl[trace] = get_Hw(get_TTable_Out(plaintexts[trace][key_byte], key_candidate));

                // Compares HW values of parallel and seriell computation
                /*if(hw_sl[trace] != hw_pl[key_candidate*NUMBER_OF_TRACES + trace])
                std::cout << "Error (HW)!! " << hw_sl[trace] << " vs " << hw_pl[key_candidate*NUMBER_OF_TRACES + trace] << endl;*/
            }
#ifdef TIME
            t_hw_serial += float( clock() - begin_time_hw2 ) / CLOCKS_PER_SEC;
#endif
#endif


            // Calculate Correlation Coefficient
#ifdef PARALLEL1
#ifdef TIME
            // Start measuring time
            const clock_t begin_time_cc_p1 = clock();
#endif
            cudaError_t cudaStatus = computeCoeffWithCuda(corr[key_candidate], traces, hw_sl);

            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "computeCoeffWithCuda failed!");
                return 1;
            }
#ifdef TIME
            t_cc_parallel1 += float( clock() - begin_time_cc_p1 ) / CLOCKS_PER_SEC;
#endif
#endif
            for (int trace_point = TRACE_STARTPOINT; trace_point < TRACE_ENDPOINT; trace_point++)
            {
#ifdef SERIAL
#ifdef TIME
            // Start measuring time
            const clock_t begin_time_cc_serial = clock();
#endif
                //// Create "Slice" of Traces at certain point
                int *traces_at_trace_point;
                traces_at_trace_point = new int [NUMBER_OF_TRACES];

                for (int t = 0; t < NUMBER_OF_TRACES; t++)
                {
                    traces_at_trace_point[t] = traces[trace_point+t*NUMBER_OF_TRACES];
                }

                // Correlation Coefficient 
                cc = get_Corr_Coef(traces_at_trace_point, hw_sl, NUMBER_OF_TRACES);
#ifdef TIME
                t_cc_serial += float( clock() - begin_time_cc_serial ) / CLOCKS_PER_SEC;
#endif
                delete[] traces_at_trace_point;
#endif
#ifdef PARALLEL1
                cc = corr[key_candidate][trace_point-TRACE_STARTPOINT];
#endif

#ifdef PARALLEL2
                cc = corr_unrolled[key_candidate * (TRACE_ENDPOINT - TRACE_STARTPOINT) + trace_point - TRACE_STARTPOINT];
#endif
                if(cc > highest_cc)
                {
                    highest_cc = cc;
                    key[key_byte] = key_candidate;
                }
            }
        }
        highest_cc = -1.0;     
    } 
    // Stop measuring time
    /*std::cout << "HW seriell:       " << float( clock () - begin_time_hw_sl ) /  CLOCKS_PER_SEC << "sec" << endl;*/

    std::cout << "CIPHER KEY =";
    for(int i = 0; i < BYTES_PER_KEY; i++)
    {
        std::cout << hex << " " << key[i];
    }
    std::cout << endl;

    // deleting everything

    for (int i = 0; i < NUMBER_OF_TEXTS; i++)
    {
        delete[] plaintexts[i];
    }
    delete[] traces;
    delete[] plaintexts;
    delete[] key;

#if !defined PARALLEL2
    delete[] hw_sl;
#else
    delete[] traces_at_trace_point;
#endif

#ifdef PARALLEL1
    for (int i = 0; i < 256; i++)
    {
        delete[] corr[i];
    }
    delete[] corr;
#endif

    /* cudaFree(dev_traces);*/
#if !defined SERIAL
    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
#endif

    // 	CORRECT CIPHER KEY: 2b  7e  15  16  28  ae  d2  a6  ab  f7  15  88  09  cf  4f  3c
    //                 dec: 043 126 021 022 040 174 210 166 171 247 021 136 009 207 079 060		 

    // Output of runtimes, when TIME is defined
#ifdef TIME
    std::cout << "Runtime: " << endl;
#ifdef SERIAL
    std::cout << "computation of HW: " << t_hw_serial << "sec" << endl << "computation of cc: " << t_cc_serial << "sec" << endl ;
#endif
#ifdef PARALLEL1
    std::cout << "computation of HW: " << t_hw_serial << "sec" << endl << "computation of cc: " << t_cc_parallel1 << "sec" << endl ;
#endif
#ifdef PARALLEL2
    std::cout << "computation of HW: " << t_hw_parallel << "sec" << endl << "computation of cc: " << t_cc_parallel2 << "sec" << endl ;
#endif
    std::cout << "in total: " << float( clock () - begin_time_calculation ) /  CLOCKS_PER_SEC << "sec" << endl;
#endif

    std::cout << "" << endl;
    return 0;
}
