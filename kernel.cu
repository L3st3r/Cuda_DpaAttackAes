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
int NUMBER_OF_TEXTS = 10000;
int BYTES_PER_TEXT = 16;      // fix for AES
int BYTES_PER_KEY = 16;       // possible values for AES are 16, 24 and 32 (128, 192 or 256 bits)

int TRACE_STARTPOINT = 0;
int TRACE_ENDPOINT = 1000;

string TRACE_FILE = "Traces00000.dat";
string PLAINTEXT_FILE = "plaintexts.dat";
string CIPHERTEXT_FILE = "ciphertexts.dat";


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
  else
  {
    cout << "Unable to open file" << endl;
    return;
  }

  for (int i = 0; i < NUMBER_OF_TRACES; i++)
  {
	  for (int j = 0; j < POINTS_PER_TRACE; j++)
	  {
     //traces[i][j] = static_cast<int>(memblock[i*POINTS_PER_TRACE + j]);
     traces[j][i] = static_cast<int>(memblock[i*POINTS_PER_TRACE + j]);   // easier this way, so we don't need the array traces_at_tracepoint
     //cout << static_cast<int>(memblock[i]);
	  }
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

    cout << "the entire file content is in memory." <<endl;    
  }
  else
  {
    cout << "Unable to open file" << endl;
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

//__global__ void get_Corr_Coef_parallel(int *result, int **x, int *y, int *n)
//{
//    int i = threadIdx.x;
//    
//    _Uint32t sum_x  = 0;
//	   _Uint32t sum_y  = 0;
//
//	    for(int j = 0; j < n; j++)
//	    {
//       sum_x  += x[i][j];
//       sum_y  += y[j];
//	    }
//
//     long double x_average = sum_x/n;
//	    long double y_average = sum_y/n;
//
//	    long double dividend = 0;
//	    long double divisor1 = 0;
//	    long double divisor2 = 0;
//
//     for(int j = 0; j < n; j++)
//	    {
//		    dividend += (x[i][j] - x_average)*(y[i] - y_average); 
//		    divisor1 += (x[i][j] - x_average)*(x[i][j] - x_average); 
//		    divisor2 += (y[i] - y_average)*(y[i] - y_average); 
//	    }
//
//	    long double divisor = sqrt(divisor1)*sqrt(divisor2);
//
//	    if ((dividend == 0) || (divisor == 0))
//	    {
//		    result[i] = 0.0;
//	    }else{
//		    result[i] = dividend/divisor;
//	    }		
//}

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

int main()
{
	// #################### OWN PROGRAM #####################
	
	// Start measuring time
	const clock_t begin_time = clock();
	
	// Initialize trace array
	int **traces;
	/*traces = new int *[NUMBER_OF_TRACES];
	for (int i = 0; i < NUMBER_OF_TRACES; i++)
	{
		traces[i] = new int[POINTS_PER_TRACE];
	}*/
 traces = new int *[POINTS_PER_TRACE];
 for (int i = 0; i < POINTS_PER_TRACE; i++)
	{
		traces[i] = new int[NUMBER_OF_TRACES];
	}

	// Read traces and store in array
	read_traces(traces);

	// Stop measuring time
	std::cout << float( clock () - begin_time ) /  CLOCKS_PER_SEC << "sec" << endl;


 	// Start measuring time
	const clock_t begin_time_plaintexts = clock();
	
	// Initialize plaintext array
	unsigned _int8 **plaintexts;
	plaintexts = new unsigned _int8 *[NUMBER_OF_TEXTS];
	for (int i = 0; i < NUMBER_OF_TEXTS; i++)
	{
		plaintexts[i] = new unsigned _int8[BYTES_PER_TEXT];
	}

	// Read plaintexts and store in array
	read_texts(plaintexts, PLAINTEXT_FILE);

	// Stop measuring time
	std::cout << float( clock () - begin_time_plaintexts ) /  CLOCKS_PER_SEC << "sec" << endl;

	
 	// Start measuring time
	const clock_t begin_time_calculation = clock();

 int *hw;
 hw = new int [NUMBER_OF_TRACES];
 
 // Loop through all key bytes
 for (int key_byte = 0; key_byte < 1; key_byte++)//for (int key_byte = 0; key_byte < BYTES_PER_KEY; key_byte++)
 {
	  /*
	  // Initialize corr array (event. doch nicht nötig)
	  double **corr;
	  corr = new double *[256];
	  for (int i = 0; i < 256; i++)
	  {
		  corr[i] = new double[TRACE_ENDPOINT - TRACE_STARTPOINT];
	  }	
	  */

	  double highest_cc = 0.0;
   
   double cc = -1;
	  int key = -1;
   /*double highest_actual_cc = -1.0;
   double cc_sum = 0.0;
   double cc_avg = 0.0;*/
	 
	  // Loop through all key candidates
     for (int key_candidate = 0; key_candidate <= 255; key_candidate++)
     {
       
	     //cout << "Key Candidate = " << key_candidate << endl;
	   
	     // Measure hamming weight for every trace
         for (int trace = 0; trace < NUMBER_OF_TRACES; trace++)
         {
           // Calculate the hamming weight
	       // PM: eventuell parallelisieren??
           hw[trace] = get_Hw(get_TTable_Out(plaintexts[trace][key_byte], key_candidate));
	         }
	 
         // Calculate Correlation Coefficient 

	     for (int trace_point = TRACE_STARTPOINT; trace_point < TRACE_ENDPOINT; trace_point++)
	     {
		      // Create "Slice" of Traces at certain point
		      /*int *traces_at_trace_point;
		      traces_at_trace_point = new int [NUMBER_OF_TRACES];
			
		      for (int t = 0; t < NUMBER_OF_TRACES; t++)
		      {
			       traces_at_trace_point[t] = traces[t][trace_point];
		      }*/
		      // Correlation Coefficient 
		      //cc = get_Corr_Coef(traces_at_trace_point, hw, NUMBER_OF_TRACES);
        cc = get_Corr_Coef(traces[trace_point], hw, NUMBER_OF_TRACES);

        //delete[] traces_at_trace_point;

        if(cc > highest_cc)
		    {
			    highest_cc = cc;
			    key = key_candidate;

			    cout << "Highest CC = " << highest_cc << ", Key Candidate = " << key << endl;

		    }
      //if (cc > highest_actual_cc)
      //{
      //  highest_actual_cc = cc;
			   // //key = key_candidate;

      //  

      //}
       // cc_sum += cc;


		      //corr[key_candidate][trace_point - TRACE_STARTPOINT] = get_Corr_Coef(traces_at_trace_point, hw, NUMBER_OF_TRACES);

		      // Leider bricht der Alg. bei mir immer nach 43 Kandidaten ab... :-/

	     }
     // cout << "actual CC = " << highest_actual_cc << ", actual Key Candidate = " << key_candidate << endl;
    
        //cc_avg = cc_sum / NUMBER_OF_TRACES;
        ////cout << "key candidate= " << key_candidate << " average = " << cc_avg << " max cc = " << highest_actual_cc <<  endl;
        //cc_avg = 0.0;
        //cc_sum = 0.0;
        //  highest_actual_cc = -1;
      		    // Find the highest Correlation Coefficient 
		    
     }
     cout << "key byte " << key_byte << " complete" << endl;
 }


 // deleting everything
 	for (int i = 0; i < NUMBER_OF_TRACES; i++)
	{
		delete[] traces[i];
	}
 for (int i = 0; i < NUMBER_OF_TEXTS; i++)
	{
		delete[] plaintexts[i];
	}
 delete[] traces;
 delete[] plaintexts;
 delete[] hw;

 // TESTING CORR. COEF.
 /*		int *x;
		x = new int [10];
		int *y;
		y = new int [10];
		

		x[0] = 11111111;
		x[1] = 1;
		x[2] = 1;
		x[3] = 1;
		x[4] = 1;
		x[5] = 1;
		x[6] = 1;
		x[7] = 1;
		x[8] = 1;
		x[9] = 1;

		y[0] = 2;
		y[1] = 2;
		y[2] = 2;
		y[3] = 2;
		y[4] = 200;
		y[5] = 1;
		y[6] = 1;
		y[7] = 1;
		y[8] = 1;
		y[9] = 100;

		double cc = get_Corr_Coef(x, y, 10);
		std::cout << " CC = " << cc << endl;
*/

 //                                                             //JUST FOR TESTING
 ////TEST for T-Table:
 //std::cout << "T-Table: plaintext, key_candidate - T-Table(plaintext ^ key_candidate)" << endl;
 //std::cout << std::hex << get_TTable_Out(15, 240) << endl;
 //std::cout << std::hex << get_TTable_Out(127, 255) << endl;



 //                                                            //JUST FOR TESTING
 ////this can be used to test the hamming weight function
 //std::cout << "hw: i - hw(i)" << endl;
 //for (int i = 0; i < 256; i++)
 //{
 //  std::cout << i << " - " << (int) get_Hw(i) << endl;
 //}

	// Stop measuring time
	std::cout << float( clock () - begin_time_calculation ) /  CLOCKS_PER_SEC << "sec" << endl;


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
