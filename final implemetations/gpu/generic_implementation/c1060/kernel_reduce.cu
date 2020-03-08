/*************************************************************************/
/**   File:         kmeans_clustering.c                                 **/
/**   Description:  Implementation of compute centroids for k-means     **/
/**                 clustering algorithm                                **/
/*************************************************************************/

#include "kmeans.h"

#ifndef FLT_MAX
#define FLT_MAX 3.40282347e+38
#endif

////////////////////////////////////////////////////////////////////////////////
//!Initialize all elements of an array with default values
////////////////////////////////////////////////////////////////////////////////
__global__ void initialize(int length, int *const __restrict__ array, int value)
{
    for(int index = (blockIdx.x * blockDim.x)+threadIdx.x;index < length; index+=(gridDim.x*blockDim.x))
        array[index]=value;
}

////////////////////////////////////////////////////////////////////////////////
//!Initialize all elements of an array with default values
////////////////////////////////////////////////////////////////////////////////
__global__ void initialize(int length, float *const __restrict__ array, float value)
{
    for(int index = (blockIdx.x * blockDim.x)+threadIdx.x;index < length; index+=(gridDim.x*blockDim.x))
        array[index]=value;
}

////////////////////////////////////////////////////////////////////////////////
//!Calls initialize kernel with appropriate number of blocks
////////////////////////////////////////////////////////////////////////////////
__host__ void call_initialize(int length, int *const  array, int value)
{
    int num_blocks = length/512;

    if(num_blocks>MAX_BLOCKS)
      num_blocks=MAX_BLOCKS;
    if(num_blocks == 0) num_blocks++;

    initialize <<< dim3(num_blocks), dim3(512), 0 >>> (length,array, value);

}

////////////////////////////////////////////////////////////////////////////////
//!Calls initialize kernel with appropriate number of blocks
////////////////////////////////////////////////////////////////////////////////
__host__ void call_initialize(int length, float *const  array, float value)
{
    int num_blocks = length/512;

    if(num_blocks>MAX_BLOCKS)
      num_blocks=MAX_BLOCKS;
    if(num_blocks == 0) num_blocks++;

    initialize <<< dim3(num_blocks), dim3(512), 0 >>> (length,array, value);

}

#define INIT_VARS(len) {        \
  centCount=0;                  \
  switch(len) {                 \
    case 10: centSum[9] = 0.0f; \
    case 9: centSum[8] = 0.0f;  \
    case 8: centSum[7] = 0.0f;  \
    case 7: centSum[6] = 0.0f;  \
    case 6: centSum[5] = 0.0f;  \
    case 5: centSum[4] = 0.0f;  \
    case 4: centSum[3] = 0.0f;  \
    case 3: centSum[2] = 0.0f;  \
    case 2: centSum[1] = 0.0f;  \
    case 1: centSum[0] = 0.0f;  \
  }                             \
}

#define ADD_DATA_POINT(len) {                   \
  Data += indxInWarp;                           \
  centCount++;                                  \
  if(indxInWarp+32*0< d)                        \
    centSum[0] += Data[0];                      \
  if(len > 1) {                                 \
    if(indxInWarp+32*1 < d) {                   \
      Data = Data + 32;                         \
      centSum[1] += Data[0];                    \
    }                                           \
  }                                             \
  if(len > 2) {                                 \
    if(indxInWarp+32*2 < d) {                   \
      Data = Data + 32;                         \
      centSum[2] += Data[0];                    \
    }                                           \
  }                                             \
  if(len > 3) {                                 \
    if(indxInWarp+32*3 < d) {                   \
      Data = Data + 32;                         \
      centSum[3] += Data[0];                    \
    }                                           \
  }                                             \
  if(len > 4) {                                 \
    if(indxInWarp+32*4 < d) {                   \
      Data = Data + 32;                         \
      centSum[4] += Data[0];                    \
    }                                           \
  }                                             \
  if(len > 5) {                                 \
    if(indxInWarp+32*5 < d) {                   \
      Data = Data + 32;                         \
      centSum[5] += Data[0];                    \
    }                                           \
  }                                             \
  if(len > 6) {                                 \
    if(indxInWarp+32*6 < d) {                   \
      Data = Data + 32;                         \
      centSum[6] += Data[0];                    \
    }                                           \
  }                                             \
  if(len > 7) {                                 \
    if(indxInWarp+32*7 < d) {                   \
      Data = Data + 32;                         \
      centSum[7] += Data[0];                    \
    }                                           \
  }                                             \
  if(len > 8) {                                 \
    if(indxInWarp+32*8 < d) {                   \
      Data = Data + 32;                         \
      centSum[8] += Data[0];                    \
    }                                           \
  }                                             \
  if(len > 9) {                                 \
    if(indxInWarp+32*9 < d) {                   \
      Data = Data + 32;                         \
      centSum[9] += Data[0];                    \
    }                                           \
  }                                             \
}

#define STORE_CENTROID_SUM(len) {                                     \
  float *CentroidSum = dCentroidSum + d_centroidSumIndx + indxInWarp; \
  CentroidSum[0] = centSum[0];                                        \
  if(len > 1) {                                                       \
    CentroidSum += 32;                                                \
    CentroidSum[0] = centSum[1];                                      \
  }                                                                   \
  if(len > 2) {                                                       \
    CentroidSum += 32;                                                \
    CentroidSum[0] = centSum[2];                                      \
  }                                                                   \
  if(len > 3) {                                                       \
    CentroidSum += 32;                                                \
    CentroidSum[0] = centSum[3];                                      \
  }                                                                   \
  if(len > 4) {                                                       \
    CentroidSum += 32;                                                \
    CentroidSum[0] = centSum[4];                                      \
  }                                                                   \
  if(len > 5) {                                                       \
    CentroidSum += 32;                                                \
    CentroidSum[0] = centSum[5];                                      \
  }                                                                   \
  if(len > 6) {                                                       \
    CentroidSum += 32;                                                \
    CentroidSum[0] = centSum[6];                                      \
  }                                                                   \
  if(len > 7) {                                                       \
    CentroidSum += 32;                                                \
    CentroidSum[0] = centSum[7];                                      \
  }                                                                   \
  if(len > 8) {                                                       \
    CentroidSum += 32;                                                \
    CentroidSum[0] = centSum[8];                                      \
  }                                                                   \
  if(len > 9) {                                                       \
    CentroidSum += 32;                                                \
    CentroidSum[0] = centSum[9];                                      \
  }                                                                   \
}


#define REDUCE_SUM_IN_SHARED(len) {                               \
  s_centroidSums[s_centroidSumIndx] = centSum[0] = centSum[0] + s_centroidSums[reduceWith]; \
  if(len > 1)                                                                                 \
    s_centroidSums[s_centroidSumIndx+blockDim.x*1] = centSum[1] = centSum[1] + s_centroidSums[reduceWith+blockDim.x*1]; \
  if(len > 2)                                                                                 \
    s_centroidSums[s_centroidSumIndx+blockDim.x*2] = centSum[2] = centSum[2] + s_centroidSums[reduceWith+blockDim.x*2]; \
  if(len > 3)                                                                                 \
    s_centroidSums[s_centroidSumIndx+blockDim.x*3] = centSum[3] = centSum[3] + s_centroidSums[reduceWith+blockDim.x*3]; \
  if(len > 4)                                                                                 \
    s_centroidSums[s_centroidSumIndx+blockDim.x*4] = centSum[4] = centSum[4] + s_centroidSums[reduceWith+blockDim.x*4]; \
  if(len > 5)                                                                                 \
    s_centroidSums[s_centroidSumIndx+blockDim.x*5] = centSum[5] = centSum[5] + s_centroidSums[reduceWith+blockDim.x*5]; \
  if(len > 6)                                                                                 \
    s_centroidSums[s_centroidSumIndx+blockDim.x*6] = centSum[6] = centSum[6] + s_centroidSums[reduceWith+blockDim.x*6]; \
  if(len > 7)                                                                                 \
    s_centroidSums[s_centroidSumIndx+blockDim.x*7] = centSum[7] = centSum[7] + s_centroidSums[reduceWith+blockDim.x*7]; \
  if(len > 8)                                                                                 \
    s_centroidSums[s_centroidSumIndx+blockDim.x*8] = centSum[8] = centSum[8] + s_centroidSums[reduceWith+blockDim.x*8]; \
  if(len > 9)                                                                                 \
    s_centroidSums[s_centroidSumIndx+blockDim.x*9] = centSum[9] = centSum[9] + s_centroidSums[reduceWith+blockDim.x*9]; \
}

#define   STORE_SUM_IN_SHARED(len)  {                             \
  s_centroidSums[s_centroidSumIndx] = centSum[0];                 \
  if(len > 1)                                                     \
    s_centroidSums[s_centroidSumIndx+blockDim.x*1] = centSum[1];  \
  if(len > 2)                                                     \
    s_centroidSums[s_centroidSumIndx+blockDim.x*2] = centSum[2];  \
  if(len > 3)                                                     \
    s_centroidSums[s_centroidSumIndx+blockDim.x*3] = centSum[3];  \
  if(len > 4)                                                     \
    s_centroidSums[s_centroidSumIndx+blockDim.x*4] = centSum[4];  \
  if(len > 5)                                                     \
    s_centroidSums[s_centroidSumIndx+blockDim.x*5] = centSum[5];  \
  if(len > 6)                                                     \
    s_centroidSums[s_centroidSumIndx+blockDim.x*6] = centSum[6];  \
  if(len > 7)                                                     \
    s_centroidSums[s_centroidSumIndx+blockDim.x*7] = centSum[7];  \
  if(len > 8)                                                     \
    s_centroidSums[s_centroidSumIndx+blockDim.x*8] = centSum[8];  \
  if(len > 9)                                                     \
    s_centroidSums[s_centroidSumIndx+blockDim.x*9] = centSum[9];  \
}

__device__ void inline setSharedVariables(int *s_centroidSumIndx, int *s_centroidCountIndx){
  int tidx = threadIdx.x;

  s_centroidCountIndx[0] = (tidx>>5);
  s_centroidSumIndx[0]   = tidx;
}


__device__ void inline setReductionVariables(int n, int k, int d, int lenPerBlock,
            int *centroidSumIndx, int *centroidCountIndx,int *reduceFor, int *dataIndx, int *pointsToProcessByCurrWarp,
            int *dataIndxIncr) {
  int tidx    = threadIdx.x;
  int bidx    = blockIdx.x;
  int threads = blockDim.x;
  int blocks  = gridDim.x;
  int warps   = threads>>5;


  int blocksPerCluster  = blocks/k;
  int pointsPerBlock    = n/blocksPerCluster;
  // ensure all n points are covered
  if(pointsPerBlock*blocksPerCluster < n)
    pointsPerBlock++;
  // ensure points per block is atleast equal to threadsPerBlock , otherwise initial block's threads might end up reducing points not meant for them.
  if(pointsPerBlock < threads)
    pointsPerBlock = threads;
  // ensure points per block is a multiple of WARP SIZE, otherwise initial boack's initial warps might end up reducing points not meant for them.
  if(pointsPerBlock & 31)
    pointsPerBlock = ((pointsPerBlock>>5)+1)<<5;

  int pointsPerWarpPerBlock     = pointsPerBlock/warps;
  int pointsLeftPerBlock        = pointsPerBlock - pointsPerWarpPerBlock*warps;

  pointsToProcessByCurrWarp[0]  = (pointsLeftPerBlock > ((tidx>>5)<<5))?pointsPerWarpPerBlock+32:pointsPerWarpPerBlock;
  reduceFor[0]                  = bidx % k;
  centroidCountIndx[0]          = bidx;
  dataIndx[0]                   = (int)(bidx/k)*pointsPerBlock +((tidx>>5)<<5);
  dataIndxIncr[0]               = threads;
  centroidSumIndx[0]            = (reduceFor[0]*blocksPerCluster + (int)(bidx/k))*lenPerBlock;
}

// Assumption number of blocks is a multiple of k
template<int dimPerThread>
__global__ void reduceCluster(int n, int k, int d, float *dData, size_t pitchData, int *Index,
            float *dCentroidSum, int lenPerBlock, int *CentroidCount)  {

  extern __shared__ char s_memory[];
  int *s_closest  = (int*)(s_memory);
  int s_closestIndx = ((threadIdx.x>>5)<<5);


  float centSum[dimPerThread];
  int centCount;
  INIT_VARS(dimPerThread)

  int d_centroidSumIndx, d_centroidCountIndx, reduceFor, dataIndx, pointsToProcessByCurrWarp, dataIndxIncr;
  setReductionVariables(n, k, d, lenPerBlock, &d_centroidSumIndx, &d_centroidCountIndx, &reduceFor,
                        &dataIndx, &pointsToProcessByCurrWarp, &dataIndxIncr);


  int indxInWarp = (threadIdx.x & 31);
  if(pointsToProcessByCurrWarp > 0) {
    // Add co-ordinates of centroids for the assigned cluster
    for(int pointsIter = 0; pointsIter < pointsToProcessByCurrWarp; pointsIter+=32)  {
      if((dataIndx+indxInWarp) < n){
        s_closest[s_closestIndx+indxInWarp] = Index[dataIndx+indxInWarp];
      }else {
        s_closest[s_closestIndx+indxInWarp] = -1;
      }
      #pragma unroll
      for(int i = 0; i<32;i++)  {
        float *Data = (float*)((char*)dData + (dataIndx+i)*pitchData);
        if(s_closest[s_closestIndx+i] == reduceFor) {
          ADD_DATA_POINT(dimPerThread)
        }
      }
      dataIndx += dataIndxIncr;
    }
  }

  // Perform reduction between the warps.
  char *s_centroidSumsAndCounts = s_memory;
  int *s_centroidCounts         = (int*)(s_centroidSumsAndCounts);
  float *s_centroidSums         = (float*)(s_centroidCounts + (blockDim.x>>5));

  int s_centroidSumIndx, s_centroidCountIndx;
  setSharedVariables(&s_centroidSumIndx, &s_centroidCountIndx);

  __syncthreads();
  if(indxInWarp == 0){
    s_centroidCounts[s_centroidCountIndx] = centCount;
  }
  STORE_SUM_IN_SHARED(dimPerThread)

  for(int stride = 512;stride>=32;stride/=2){
    __syncthreads();
    if(blockDim.x >= (stride<<1) && threadIdx.x < stride){
      int reduceWith = threadIdx.x + stride;
      if(indxInWarp == 0){
        s_centroidCounts[s_centroidCountIndx] = centCount = centCount + s_centroidCounts[reduceWith>>5];
      }
      REDUCE_SUM_IN_SHARED(dimPerThread)
    }
  }

  // Store the results after internal reduction
  if(threadIdx.x == 0)  {
    CentroidCount[d_centroidCountIndx] = centCount;
  }
  if(threadIdx.x < 32) {
    STORE_CENTROID_SUM(dimPerThread);
  }
}

#define INVOKE_REDUCE_DEV(i)   \
  reduceCluster<i><<<dim3(blockDim),dim3(threadDim),sharedForCount+sharedForSum>>>(n, k, d, dData, pitchData, Index, dCentroidSum, lenPerBlock, CentroidCount); \
  break;

__host__ void callReduceCluster(int n, int k, int d, float *dData, size_t pitchData, int *Index,
            float *dCentroidSum, int lenPerBlock, int *CentroidCount, int dimPerThread, int blockDim, int threadDim) {
  int sharedForCount  =   (threadDim>>5);
  sharedForCount      *=  sizeof(int);
  int sharedForSum    =   (d%32) == 0? (d>>5)*threadDim : ((d>>5)+1)*threadDim;
  sharedForSum        *=  sizeof(float);
   // printf("sharedForCount:%d bytes, sharedForSum:%d bytes\n",sharedForCount, sharedForSum);
  switch(dimPerThread)  {
    case 1:INVOKE_REDUCE_DEV(1)
    case 2:INVOKE_REDUCE_DEV(2)
    case 3:INVOKE_REDUCE_DEV(3)
    case 4:INVOKE_REDUCE_DEV(4)
    case 5:INVOKE_REDUCE_DEV(5)
    case 6:INVOKE_REDUCE_DEV(6)
    case 7:INVOKE_REDUCE_DEV(7)
    case 8:INVOKE_REDUCE_DEV(8)
    case 9:INVOKE_REDUCE_DEV(9)
    case 10:INVOKE_REDUCE_DEV(10)
    default: printf("Only 320 attributes can be reduced in a single reduce call.\n");
             exit(EXIT_FAILURE);
  }
}

#define ADD_BLOCK_SUM(len)  {                   \
  int index = block_iter*lenPerBlock + indxInWarp;  \
  if(indxInWarp+32*0< d)                        \
    centSum[0] += CentroidSum[index];           \
  if(len > 1) {                                 \
    if(indxInWarp+32*1 < d) {                   \
      centSum[1] += CentroidSum[index+32*1];    \
    }                                           \
  }                                             \
  if(len > 2) {                                 \
    if(indxInWarp+32*2 < d) {                   \
      centSum[2] += CentroidSum[index+32*2];    \
    }                                           \
  }                                             \
  if(len > 3) {                                 \
    if(indxInWarp+32*3 < d) {                   \
      centSum[3] += CentroidSum[index+32*3];    \
    }                                           \
  }                                             \
  if(len > 4) {                                 \
    if(indxInWarp+32*4 < d) {                   \
      centSum[4] += CentroidSum[index+32*4];    \
    }                                           \
  }                                             \
  if(len > 5) {                                 \
    if(indxInWarp+32*5 < d) {                   \
      centSum[5] += CentroidSum[index+32*5];    \
    }                                           \
  }                                             \
  if(len > 6) {                                 \
    if(indxInWarp+32*6 < d) {                   \
      centSum[6] += CentroidSum[index+32*6];    \
    }                                           \
  }                                             \
  if(len > 7) {                                 \
    if(indxInWarp+32*7 < d) {                   \
      centSum[7] += CentroidSum[index+32*7];    \
    }                                           \
  }                                             \
  if(len > 8) {                                 \
    if(indxInWarp+32*8 < d) {                   \
      centSum[8] += CentroidSum[index+32*8];    \
    }                                           \
  }                                             \
  if(len > 9) {                                 \
    if(indxInWarp+32*9 < d) {                   \
      centSum[9] += CentroidSum[index+32*9];    \
    }                                           \
  }                                             \
}

#define STORE_CENTROID(len) {                 \
  Centroid = Centroid + indxInWarp;           \
  if(indxInWarp+32*0< d)                      \
    Centroid[0] = centSum[0]/centCount;       \
  if(len > 1) {                               \
    Centroid += 32;                           \
    if(indxInWarp+32*1< d)                    \
      Centroid[0] = centSum[1]/centCount;     \
  }                                           \
  if(len > 2) {                               \
    Centroid += 32;                           \
    if(indxInWarp+32*2< d)                    \
      Centroid[0] = centSum[2]/centCount;     \
  }                                           \
  if(len > 3) {                               \
    Centroid += 32;                           \
    if(indxInWarp+32*3< d)                    \
      Centroid[0] = centSum[3]/centCount;     \
  }                                           \
  if(len > 4) {                               \
    Centroid += 32;                           \
    if(indxInWarp+32*4< d)                    \
      Centroid[0] = centSum[4]/centCount;     \
  }                                           \
  if(len > 5) {                               \
    Centroid += 32;                           \
    if(indxInWarp+32*5< d)                    \
      Centroid[0] = centSum[5]/centCount;     \
  }                                           \
  if(len > 6) {                               \
    Centroid += 32;                           \
    if(indxInWarp+32*6< d)                    \
      Centroid[0] = centSum[6]/centCount;     \
  }                                           \
  if(len > 7) {                               \
    Centroid += 32;                           \
    if(indxInWarp+32*7< d)                    \
      Centroid[0] = centSum[7]/centCount;     \
  }                                           \
  if(len > 8) {                               \
    Centroid += 32;                           \
    if(indxInWarp+32*8< d)                    \
      Centroid[0] = centSum[8]/centCount;     \
  }                                           \
  if(len > 9) {                               \
    Centroid += 32;                           \
    if(indxInWarp+32*9< d)                    \
      Centroid[0] = centSum[9]/centCount;     \
  }                                           \
}

__device__ void inline setInterBlockReductionVariables(int n, int k, int d, int lenPerBlock,
            int blocksPerCluster, int *d_centroidSumLoc, int *reduceFor)  {
  int tidx    = threadIdx.x;
  int bidx    = blockIdx.x;
  int threads = blockDim.x;
  int blocks  = gridDim.x;

  reduceFor[0]        = bidx;
  d_centroidSumLoc[0] = reduceFor[0]*blocksPerCluster*lenPerBlock;
}

// Assumption number of blocks is same as value k.
// Only first warp is expected to work
template<int dimPerThread>
__global__ void setCluster(int n, int k, int d, float *dCentroidSum, int lenPerBlock,
            int blocksPerCluster, int *dCentroidCount, float *dCentroid)  {

  float centSum[dimPerThread];
  int centCount;
  INIT_VARS(dimPerThread)

  int d_centroidSumLoc, reduceFor;
  setInterBlockReductionVariables(n, k, d, lenPerBlock, blocksPerCluster, &d_centroidSumLoc, &reduceFor);


  int indxInWarp = (threadIdx.x & 31);
  float *CentroidSum = dCentroidSum + d_centroidSumLoc;
  int *CentroidCount = dCentroidCount;

  if(threadIdx.x < 32)  {
    for(int block_iter = 0; block_iter < blocksPerCluster; block_iter++) {
      centCount += CentroidCount[block_iter*k + reduceFor];

      ADD_BLOCK_SUM(dimPerThread)
    }

    float *Centroid = dCentroid + (d*reduceFor);
    if(centCount > 0)  {
      STORE_CENTROID(dimPerThread)
    }
  }
}

#define INVOKE_EXTRNL_REDUCE_DEV(i)  \
  setCluster<i><<<dim3(k),dim3(32)>>>(n, k, d, dCentroidSum, lenPerBlock, blocksPerCluster, dCentroidCount, dCentroid); \
  break;

__host__ void callSetCluster(int n, int k, int d, float *dCentroidSum, int lenPerBlock,
            int blocksPerCluster, int *dCentroidCount, float *dCentroid, int dimPerThread) {
  switch(dimPerThread)  {
    case 1:INVOKE_EXTRNL_REDUCE_DEV(1)
    case 2:INVOKE_EXTRNL_REDUCE_DEV(2)
    case 3:INVOKE_EXTRNL_REDUCE_DEV(3)
    case 4:INVOKE_EXTRNL_REDUCE_DEV(4)
    case 5:INVOKE_EXTRNL_REDUCE_DEV(5)
    case 6:INVOKE_EXTRNL_REDUCE_DEV(6)
    case 7:INVOKE_EXTRNL_REDUCE_DEV(7)
    case 8:INVOKE_EXTRNL_REDUCE_DEV(8)
    case 9:INVOKE_EXTRNL_REDUCE_DEV(9)
    case 10:INVOKE_EXTRNL_REDUCE_DEV(10)
    default: printf("Only 320 attributes can be reduced in a single reduce call.\n");
             exit(EXIT_FAILURE);
  }
}

