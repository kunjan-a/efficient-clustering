#ifndef SETTINGS_H_INCLUDED
#define SETTINGS_H_INCLUDED
#include <string>
using namespace std;


////////////////////////////////////////////
//Common Settings
#define INPUTFILE_DEF "input_200lakh"
#define ENV_INPUTFILE "INPUT_FILE"
#define ENV_OUTPUTFILE "OUTPUT_FILE"
#define OUTPUTFILE_DEF "input_200lakh_result"

#define ENV_NUMPOINT "INPUT_NUMPOINT"
#define ENV_NUMITER "INPUT_NUMITER"
#define ENV_NUMCLUSTER "INPUT_NUMCLUSTER"
#define DEF_NUMATTRIB 3
#define HARDCODED_POINTS 1
#define WRITE_POINTS 1                    // if 1 then each point with the index and the co-ordinates it was assigned to is printed
#define CHECK_THRESHOLD 0
#define PROFILE_TIME 1                    //0: don't profile internal times   1:profile internal function times too
#define DONT_CHNG_CENTROIDS 0          //0: change centroids in each iteration as per the new mean values obtained. 1: Leave the centroids as it is for each iteration.

#define USE_SQRT 0
#if USE_SQRT == 1
  #define SQRT_MACRO(x)  sqrt((double)x)
#else
  #define SQRT_MACRO(x)  x
#endif



// CUDA Settings
#define APPROACH 155
#define LOG_BLOCKDIM 8         // Log of number of threads per block
#define BLOCKDIM (1<<LOG_BLOCKDIM)
#define MIN_NUM_BLOCKS_PER_MULTIPROC 1    // This value cannot be greater than 8 as so far for all architectuers CUDA only allows maximum 8 resident blocks per multiprocessor.
#define NUM_MULTIPROC 30                  // This value should be read from the properties of the card.
#define POINTS_PER_THREAD 2
#define LOG_HALF_WARP 4                 // Since warp size is 32 threads i.e. threads are executed in batches called warps of 32
#define HALF_WARP (1<<LOG_HALF_WARP)
#define FULL_WARP (HALF_WARP<<1)
#define TRY_CACHE_BW_DEV_AND_SHARED 0  // As per paper told by Ajith there is a cache b/w device and shared mem. even in 1.3 card. So try and make use of it by placing syncthread b4 and after coalesced accesses from dev.
#define TRY_ATOMIC_CNTR_IN_LOCAL_REDCN 0  // Use the fact that in coalesced access of shared memory during local reduction more than one thread per warp might have served the ir thread and so we may not need to do all iterations.
#define ENABLE_LOAD_CHANGES 0             // New changes to make sure that all the processors are occupied even in the cases where POINTS_PER_THREAD is more than 1

#define RESTRICT __restrict__
#define INLINE inline
#define VOLATILE_STORE volatile           // While doing store_nearest_in_ calls there should not be a need of having arg as volatile as it only calls find_closest_centroid which does have args as volatile
//////////////////////////////////////////////////

void readInput(int &n, int &m, int &k, float &threshold, float *h_datapt[DEF_NUMATTRIB]);
void writeOutput(int n, int k, float *ref_datapt[DEF_NUMATTRIB], int *h_clusterno, float *ref_centroid[DEF_NUMATTRIB]);
void alloc2d(float *arr2d[DEF_NUMATTRIB], const int size);
void free2d(float *arr2d[DEF_NUMATTRIB]);

inline void fprintAll(FILE * fout,float *const arr, const int size, const int incr);
void printAll(float *const arr, const int size, const int incr);
string GetEnv( const char * var );
int string2int(string value);

#endif



