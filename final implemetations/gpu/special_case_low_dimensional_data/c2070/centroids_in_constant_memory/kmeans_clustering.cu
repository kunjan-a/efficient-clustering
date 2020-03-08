
#include <stdio.h>
#include "cuda.h"
#include "cublas.h"
#include <unistd.h>
#include <cutil.h>

#include "kmeans.h"
#include "kernel_assignLabels.cu"
//
//	generic functions first
//

    //! Check for CUDA error
#ifdef _DEBUG
#  define MY_CUT_CHECK_ERROR(errorMessage) {                                  \
    cudaError_t err = cudaGetLastError();                                     \
    if( cudaSuccess != err) {                                                 \
        fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",     \
                errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) ); \
    }                                                                         \
    err = CUT_DEVICE_SYNCHRONIZE();                                           \
    if( cudaSuccess != err) {                                                 \
        fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",     \
                errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) ); \
        exit(EXIT_FAILURE);                                                   \
    }                                                                         \
    }
#else
#  define MY_CUT_CHECK_ERROR(errorMessage) {                                  \
    cudaError_t err = cudaGetLastError();                                     \
    if( cudaSuccess != err) {                                                 \
        fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",     \
                errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) ); \
        exit(EXIT_FAILURE);                                                   \
    }                                                                         \
    }
#endif

#define INVOKE_ASSIGN_LABELS(i, assign_time)                                            \
  callAssignLabels(n, k, dData_assign, dIndex, pitchData_assign, i, blockDim_assign, threadDim_assign);  \
  BEGIN_TIMING( assign_time );                                                          \
  callAssignLabels(n, k, dData_assign, dIndex, pitchData_assign, i, blockDim_assign, threadDim_assign);  \
  END_TIMING( assign_time );


#define BEGIN_TIMING(time)	                                                  \
{	                                                                            \
	{	                                                                          \
    MY_CUT_CHECK_ERROR("Kernel execution failed");                            \
		start_event();	                                                          \
		for( unsigned int iteration = 0; iteration < numIterations; iteration++ ){

#define END_TIMING(time) }                          \
		time = stop_event();	                          \
    MY_CUT_CHECK_ERROR("Kernel execution failed");  \
	}	                                                \
}


//declare time event
cudaEvent_t start_timer;
float event_time;
void start_event()
{
  event_time=0;

  static bool initialised = false;
  if(!initialised)
  {
    cudaEventCreate(&start_timer);
    initialised=true;
  }

  cudaEventRecord( start_timer, 0 );

}

cudaEvent_t stop_timer;
float stop_event()
{

  static bool initialised = false;
  if(!initialised)
  {
    cudaEventCreate(&stop_timer);
    initialised=true;
  }

  cudaEventRecord( stop_timer, 0 );
  cudaEventSynchronize(stop_timer);
  cudaEventElapsedTime( &event_time, start_timer, stop_timer );
  return event_time;
}

void destroy_event()
{
    cudaEventDestroy(start_timer);
    cudaEventDestroy(stop_timer);
}

void error( char *message )
{
	fprintf( stderr, "ERROR: %s\n", message );
	exit (1);
}

#define assert( condition, ... ) { if( !( condition ) ) error( __VA_ARGS__ ); }

inline void Q( cudaError_t status ) { assert( status == cudaSuccess, "CUDA fails" ); }



void copyTranspose(float* src, float* dest, int srcCols, int srcRows){
  for(int i =0 ; i<srcRows; i++){
    float *currSrcRow = src + srcCols*i;
    for(int j=0; j<srcCols; j++){
      dest[j*srcRows + i] = currSrcRow[j];
    }
  }
}


float kmeans_clustering(float **attributes,       /* in: [numObjects][numAttributes]    */
                        int     numAttributes,
                        int     numObjects,
                        int     numClusters,
                        float **cluster_centres,  /* in: [numClusters][numAttributes]   */
                        int    *cluster_assign,   /* out: [numObjects]                  */
                        int     numIterations,
                        int     threadDim_assign)
{
  int n                     = numObjects;
  int k                     = numClusters;
  int d                     = numAttributes;
  float *Data_rowMajor      = attributes[0];
  float *Centroid_rowMajor  = cluster_centres[0];
  int *Index                = cluster_assign;
  float *Data_colMajor      = NULL;
  float *Centroid_colMajor  = NULL;


  int blockDim_assign = n/threadDim_assign;
  if(blockDim_assign*threadDim_assign < n) blockDim_assign++;
  printf("Total number of blocks = %d\n",blockDim_assign);


  // Copy centroids
  int sizeCentroid = k*d;
#ifdef ACCESS_CENT_COALESCED
  Centroid_colMajor = (float*)malloc( sizeCentroid*sizeof( float ) );
  assert( Centroid_colMajor != NULL , "memory allocation error for column-major copy of centroids" );
  copyTranspose(Centroid_rowMajor,Centroid_colMajor,d,k);
  cudaMemcpyToSymbol(dCentroid, Centroid_colMajor, sizeCentroid*sizeof(float));
#else
  cudaMemcpyToSymbol(dCentroid, Centroid_rowMajor, sizeCentroid*sizeof(float));
#endif
  //  printf("\ndCentroid:%u, sizeCentroid:%d\n",dCentroid,sizeCentroid);


  // Allocate and copy data - points
  int sizeData = n*d;
  float *dData_assign;
  size_t pitchData_assign,pitchHostData,width,height;
#ifdef ACCESS_DATA_COALESCED
  Data_colMajor = (float*)malloc( sizeData*sizeof( float ) );
  assert( Data_colMajor != NULL , "memory allocation error for column-major copy of data points" );
  copyTranspose(Data_rowMajor,Data_colMajor,d,n);

  width = n*sizeof(float);
  height = d;
  pitchHostData = n*sizeof(float);
  Q( cudaMallocPitch( (void **) &dData_assign, &pitchData_assign, width, height ) );
  Q( cudaMemcpy2D( (void *)dData_assign, pitchData_assign, (const void *)Data_colMajor, pitchHostData, width, height,
                  cudaMemcpyHostToDevice) );
#else
  width = d*sizeof(float);
  height = n;
  pitchHostData = d*sizeof(float);
  Q( cudaMallocPitch( (void **) &dData_assign, &pitchData_assign, width, height ) );
  Q( cudaMemcpy2D( (void *)dData_assign, pitchData_assign, (const void *)Data_rowMajor, pitchHostData, width, height,
                  cudaMemcpyHostToDevice) );
#endif
  //  printf("\npitch:%u, width:%d\n",pitchData_assign,width);
  //  printf("\ndData_assign:%u, sizeData:%d bytes\n",dData_assign,height*pitchData_assign);


  // Allocate membership array in device memory
  int *dIndex;
  Q(cudaMalloc( (void**) &dIndex, n*sizeof(int) ));
  //  printf("\ndIndex:%u, n:%d\n",dIndex,sizeIndex);

  cudaThreadSynchronize();
  MY_CUT_CHECK_ERROR("Something before kernel execution failed");


  // Invoke label assignment
  float our_time;
  INVOKE_ASSIGN_LABELS(d,our_time)


  // Copy deviceResult to membership array
  Q(cudaMemcpy( Index, dIndex, sizeof(int)*n, cudaMemcpyDeviceToHost));


  Q(cudaFree(dData_assign));
  Q(cudaFree(dIndex));


  if(Data_colMajor != NULL)
  {
      free(Data_colMajor);
      Data_colMajor = NULL;
  }
  if(Centroid_colMajor != NULL)
  {
      free(Centroid_colMajor);
      Centroid_colMajor = NULL;
  }


  //	report the results
  printf( "Total time taken for label assignment is: %f milliseconds\n",our_time);


	//	shutdown
  destroy_event();
	return our_time;
}

