
#include <stdio.h>
#include "cuda.h"
#include "cublas.h"
#include <unistd.h>
#include <cutil.h>

#include "kmeans.h"
#include "kernel_assignLabels.cu"
#include "kernel_reduce.cu"
#define CHECK_WITH_CPU
#define SHOW_DATAPOINTS
#define SHOW_CENTROIDS
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
  BEGIN_TIMING( assign_time );                                                          \
  callAssignLabels(n, k, d, dData_assign, dIndex, pitchData_assign, i, blockDim_assign, threadDim_assign);  \
  END_TIMING( assign_time );

#define INVOKE_REDUCE(i,reduce_time)                                          \
  BEGIN_TIMING( reduce_time );                                                \
  callReduceCluster(n, k, d, dData_reduce, pitchData_reduce, dIndex, dCentroidSum, lenPerBlock_reduce,\
                    dCentroidCount, i, blockDim_reduce, threadDim_reduce);    \
  END_TIMING( reduce_time );

#define INVOKE_EXTRNL_REDUCE(i, extrn_reduce_time)                            \
  BEGIN_TIMING( extrn_reduce_time );                                          \
  callSetCluster(n, k, d, dCentroidSum, lenPerBlock_reduce, blocksPerCluster_reduce, dCentroidCount, dCentroid_reduce, i); \
  END_TIMING( extrn_reduce_time );


#define BEGIN_TIMING(time)	                                                  \
{	                                                                            \
	{	                                                                          \
    MY_CUT_CHECK_ERROR("Kernel execution failed");                            \
		start_event();

#define END_TIMING(time)                            \
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
                        int     threadDim_assign,
                        int     distLen,
                        int     threadDim_reduce,
                        int     minWarpPerSm_reduce)
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
  printf("Total number of blocks for assigning labels: %d\n",blockDim_assign);

  // Allocate space for column-major storage of centroids
  int sizeCentroid = k*d;
  Centroid_colMajor = (float*)malloc( sizeCentroid*sizeof( float ) );
  assert( Centroid_colMajor != NULL , "memory allocation error for column-major copy of centroids" );


  // Allocate and copy data - points for label assignment
  int sizeData = n*d;
  float *dData_assign;
  size_t pitchData_assign,pitchHostData,width,height;
  Data_colMajor = (float*)malloc( sizeData*sizeof( float ) );
  assert( Data_colMajor != NULL , "memory allocation error for column-major copy of data points" );
  copyTranspose(Data_rowMajor,Data_colMajor,d,n);

  width = n*sizeof(float);
  height = d;
  pitchHostData = n*sizeof(float);
  Q( cudaMallocPitch( (void **) &dData_assign, &pitchData_assign, width, height ) );
  Q( cudaMemcpy2D( (void *)dData_assign, pitchData_assign, (const void *)Data_colMajor, pitchHostData, width, height,
                  cudaMemcpyHostToDevice) );
  //  printf("\npitch:%u, width:%d\n",pitchData_assign,width);
  //  printf("\ndData_assign:%u, sizeData:%d bytes\n",dData_assign,height*pitchData_assign);


  // Allocate membership array in device memory
  int *dIndex;
  Q(cudaMalloc( (void**) &dIndex, n*sizeof(int) ));
  //  printf("\ndIndex:%u, n:%d\n",dIndex,sizeIndex);


  int blocksPerSM_reduce   = 1;
  int warpsPerBlock_reduce = threadDim_reduce/32;
  if(warpsPerBlock_reduce < minWarpPerSm_reduce)  {
    blocksPerSM_reduce = (int)(minWarpPerSm_reduce/warpsPerBlock_reduce);
    if(blocksPerSM_reduce*warpsPerBlock_reduce < minWarpPerSm_reduce)  blocksPerSM_reduce++;
  }
  int blockDim_reduce = blocksPerSM_reduce*SM_NUM;
  blockDim_reduce = (blockDim_reduce%k)==0?blockDim_reduce:blockDim_reduce+k-(blockDim_reduce%k);
  printf("Total number of blocks for intra-block reduction: %d\n",blockDim_reduce);


  // Allocate space for row-major storage of centroids in device memory
  float *dCentroid_reduce;
  Q(cudaMalloc( (void**) &dCentroid_reduce, sizeCentroid*sizeof(float) ));
    //printf("\ndCentroid_reduce:%u, sizeCentroid:%d\n",dCentroid_reduce,sizeCentroid);


  // Allocate and copy data - points for reduction and computing new centroids
  float *dData_reduce;
  size_t pitchData_reduce;
  width = d*sizeof(float);
  height = n;
  pitchHostData = d*sizeof(float);
  Q( cudaMallocPitch( (void **) &dData_reduce, &pitchData_reduce, width, height ) );
  Q( cudaMemcpy2D( (void *)dData_reduce, pitchData_reduce, (const void *)Data_rowMajor, pitchHostData, width, height,
                  cudaMemcpyHostToDevice) );
  //  printf("\npitch:%u, width:%d\n",pitchData_reduce,width);
  //  printf("\ndData_reduce:%u, sizeData:%d bytes\n",dData_reduce,height*pitchData_reduce);


  int blocksPerCluster_reduce = blockDim_reduce/k;

  // Allocate centroid sum array that will store per block reduction results.
  float *dCentroidSum;
  int lenPerBlock_reduce  = d%64 == 0 ? d : d + 64-(d%64);
  int sizeCentroidSum     = blockDim_reduce*lenPerBlock_reduce;
  Q(cudaMalloc( (void**) &dCentroidSum, sizeCentroidSum*sizeof(float)));
    //printf("\ndCentroidSum:%u, sizeCentroidSum:%d, lenPerBlock:%d\n",dCentroidSum,sizeCentroidSum,lenPerBlock_reduce);


  // Allocate centroid count array that will store per block reduction results.
  int *dCentroidCount;
  int sizeCentroidCount = blockDim_reduce;
  Q(cudaMalloc( (void**) &dCentroidCount, sizeCentroidCount*sizeof(int)));
    //printf("\ndCentroidCount:%u, sizeCentroidCount:%d\n",dCentroidCount,sizeCentroidCount);


  cudaThreadSynchronize();
  MY_CUT_CHECK_ERROR("Something before kernel execution failed");


  float label_assignment_time,intra_block_reduction_time,inter_block_reduction_time,our_time;
  label_assignment_time = intra_block_reduction_time = inter_block_reduction_time = 0.0f;
  int dimPerThread_reduce = ((d%32) == 0)? d/32 : (d/32)+1;
  //printf("dimPerthread:%d\n",dimPerThread);
  for( unsigned int iteration = 0; iteration < numIterations; iteration++ ){

    // Copy centroids to column-major storage.
    copyTranspose(Centroid_rowMajor,Centroid_colMajor,d,k);
    cudaMemcpyToSymbol(dCentroid, Centroid_colMajor, sizeCentroid*sizeof(float));

    // Invoke label assignment
    INVOKE_ASSIGN_LABELS(distLen,our_time)
    label_assignment_time += our_time;

    // initialize centroid sum and count arrays to zero
    call_initialize(sizeCentroidSum, dCentroidSum, 0.0f);
    cudaThreadSynchronize();
    MY_CUT_CHECK_ERROR("Kernel execution failed while initialising sum array to store per block reduction result");
    call_initialize(sizeCentroidCount, dCentroidCount, 0);
    cudaThreadSynchronize();
    MY_CUT_CHECK_ERROR("Kernel execution failed while initialising count array to store per block reduction result");

    // Perform intra-block reduction
    INVOKE_REDUCE(dimPerThread_reduce,our_time)
    intra_block_reduction_time += our_time;

    // Perform inter-block reduction
    INVOKE_EXTRNL_REDUCE(dimPerThread_reduce,our_time)
    inter_block_reduction_time += our_time;

    Q( cudaMemcpy( Centroid_rowMajor, dCentroid_reduce, sizeCentroid*sizeof(float), cudaMemcpyDeviceToHost ) );

  }


  // Copy centroids to column-major storage.
  copyTranspose(Centroid_rowMajor,Centroid_colMajor,d,k);
  cudaMemcpyToSymbol(dCentroid, Centroid_colMajor, sizeCentroid*sizeof(float));

  // Invoke label assignment
  INVOKE_ASSIGN_LABELS(distLen,our_time)

  // Copy deviceResult to membership array
  Q(cudaMemcpy( Index, dIndex, sizeof(int)*n, cudaMemcpyDeviceToHost));


  Q(cudaFree(dData_assign));
  Q(cudaFree(dIndex));
  Q(cudaFree(dData_reduce));
  Q(cudaFree(dCentroid_reduce));
  Q(cudaFree(dCentroidSum));
  Q(cudaFree(dCentroidCount));


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
  printf( "Total time taken for label assignment is: %f milliseconds\n",label_assignment_time);
  printf( "Total time taken for intra-block reduction is: %f milliseconds\n",intra_block_reduction_time);
  printf( "Total time taken for inter-block reduction is: %f milliseconds\n",inter_block_reduction_time);


	//	shutdown
  destroy_event();
  our_time = label_assignment_time+intra_block_reduction_time+inter_block_reduction_time;
	return our_time;
}

