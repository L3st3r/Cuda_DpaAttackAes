#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <time.h>
using namespace std;

/*
	PSEUDO CODE:

% Loop through all key bytes
for key = 0 to 15 do

	% Loop through all key candidates
	for key_candidate = 0 to 255 do

		% Measure hamming weight for every trace
		for trace = 0 to 9999 do

			% Calculate the hamming weight
			hw[trace] = HW(T-Table(plaintext[trace, key] XOR key_candidate));

		end for

		% Estimated points where first T-Table operation is executed
		% E.g., startpoint = 0; endpoint = 900
		for i = startpoint to endpoint do
		
			% Calculate the correlation coefficient
			cc_temp = 0;
			for trace = 0 to 9999 do
				cc_temp += CorrCoef(Trace[trace, i], hw[trace]);
			end for
			cc[key, i] = cc_temp;

		end for
	end for
end for
*/

/*
	NEEDED FUNCTIONS AND KERNELS:

for the computation:					// I just looked in your pseudo code so far
	(1) adding cc_temp -> kernel
	(2) for loop around (1) -> ?
	(3) HW calculation -> function (a kernel would be an overkill here, wouldn't it?)

	everything else should be done on the CPU (maybe even inside of the main()-function)

Input / Output:
	I1: Input from text files, reading in the power consumption values
	O1: Output of the results: correlation coefficient per key hypothesis (?)

*/

// #################### OLD SAMPLE STUFF #####################
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);



__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}


// #################### OWN PROGRAM #####################

/*
 *  Global Configuration
 */
int NUMBER_OF_TRACES = 10000;
int POINTS_PER_TRACE = 10000;

string TRACE_FILE = "Traces00000.dat";


/*
 *	Function to read in values of traces
 */
void read_traces(int **traces) {
  streampos size;
  char * memblock;

  ifstream file (TRACE_FILE, ios::in|ios::binary|ios::ate);
  if (file.is_open())
  {
    size = file.tellg();
    memblock = new char [size];
    file.seekg (0, ios::beg);
    file.read (memblock, size);
    file.close();

    cout << "the entire file content is in memory." <<endl;    
  }
  else cout << "Unable to open file" << endl;

  for (int i = 0; i < NUMBER_OF_TRACES; i++)
  {
	  for (int j = 0; j < POINTS_PER_TRACE; j++)
	  {
		traces[i][j] = static_cast<int>(memblock[i*NUMBER_OF_TRACES + j]);
		//cout << static_cast<int>(memblock[i]);
	  }
  }
  delete[] memblock;
}


int main()
{
	// #################### OWN PROGRAM #####################
	
	// Start measuring time
	const clock_t begin_time = clock();
	
	// Initialize trace array
	int **traces;
	traces = new int *[NUMBER_OF_TRACES];
	for (int i = 0; i < NUMBER_OF_TRACES; i++)
	{
		traces[i] = new int[POINTS_PER_TRACE];
	}

	// Read traces and store in array
	read_traces(traces);

	// Stop measuring time
	std::cout << float( clock () - begin_time ) /  CLOCKS_PER_SEC << "sec" << endl;




	// #################### OLD SAMPLE STUFF #####################

    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}


// #################### OLD SAMPLE STUFF #####################

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
