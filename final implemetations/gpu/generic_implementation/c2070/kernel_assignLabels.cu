/*************************************************************************/
/**   File:         kmeans_clustering.c                                 **/
/**   Description:  Implementation of find cluster for k-means          **/
/**                 clustering algorithm                                **/
/*************************************************************************/

#include "kmeans.h"

#ifndef FLT_MAX
#define FLT_MAX 3.40282347e+38
#endif

#ifdef DONT_READ_DATA
  #define LOAD_DATA()  {                        \
   }
  #define INIT_DATA()   {                       \
  }
#else
  #define LOAD_DATA()  {                        \
    point=Data[0];                              \
    Data = (float *)((char*)Data + pitchData);  \
   }
  #define INIT_DATA()   {                       \
    Data = dData + pt_i;                        \
  }
#endif


#define INIT_DIST(len)  {       \
    switch(len){                \
      case 16:  dist[15]=0.0f;  \
      case 15:  dist[14]=0.0f;  \
      case 14:  dist[13]=0.0f;  \
      case 13:  dist[12]=0.0f;  \
      case 12:  dist[11]=0.0f;  \
      case 11:  dist[10]=0.0f;  \
      case 10:  dist[9] =0.0f;  \
      case 9:   dist[8] =0.0f;  \
      case 8:   dist[7] =0.0f;  \
      case 7:   dist[6] =0.0f;  \
      case 6:   dist[5] =0.0f;  \
      case 5:   dist[4] =0.0f;  \
      case 4:   dist[3] =0.0f;  \
      case 3:   dist[2] =0.0f;  \
      case 2:   dist[1] =0.0f;  \
      case 1:   dist[0] =0.0f;  \
    };                          \
}


#define UPDATE_DIST(pt_attr, len)   {                                                   \
  dist[0] += (pt_attr - dCentroid[centIndx])*(pt_attr - dCentroid[centIndx]);           \
  if(len > 1)                                                                           \
    dist[1] += (pt_attr - dCentroid[centIndx+1])*(pt_attr - dCentroid[centIndx+1]);     \
  if(len > 2)                                                                           \
    dist[2] += (pt_attr - dCentroid[centIndx+2])*(pt_attr - dCentroid[centIndx+2]);     \
  if(len > 3)                                                                           \
    dist[3] += (pt_attr - dCentroid[centIndx+3])*(pt_attr - dCentroid[centIndx+3]);     \
  if(len > 4)                                                                           \
    dist[4] += (pt_attr - dCentroid[centIndx+4])*(pt_attr - dCentroid[centIndx+4]);     \
  if(len > 5)                                                                           \
    dist[5] += (pt_attr - dCentroid[centIndx+5])*(pt_attr - dCentroid[centIndx+5]);     \
  if(len > 6)                                                                           \
    dist[6] += (pt_attr - dCentroid[centIndx+6])*(pt_attr - dCentroid[centIndx+6]);     \
  if(len > 7)                                                                           \
    dist[7] += (pt_attr - dCentroid[centIndx+7])*(pt_attr - dCentroid[centIndx+7]);     \
  if(len > 8)                                                                           \
    dist[8] += (pt_attr - dCentroid[centIndx+8])*(pt_attr - dCentroid[centIndx+8]);     \
  if(len > 9)                                                                           \
    dist[9] += (pt_attr - dCentroid[centIndx+9])*(pt_attr - dCentroid[centIndx+9]);     \
  if(len > 10)                                                                          \
    dist[10] += (pt_attr - dCentroid[centIndx+10])*(pt_attr - dCentroid[centIndx+10]);  \
  if(len > 11)                                                                          \
    dist[11] += (pt_attr - dCentroid[centIndx+11])*(pt_attr - dCentroid[centIndx+11]);  \
  if(len > 12)                                                                          \
    dist[12] += (pt_attr - dCentroid[centIndx+12])*(pt_attr - dCentroid[centIndx+12]);  \
  if(len > 13)                                                                          \
    dist[13] += (pt_attr - dCentroid[centIndx+13])*(pt_attr - dCentroid[centIndx+13]);  \
  if(len > 14)                                                                          \
    dist[14] += (pt_attr - dCentroid[centIndx+14])*(pt_attr - dCentroid[centIndx+14]);  \
  if(len > 15)                                                                          \
    dist[15] += (pt_attr - dCentroid[centIndx+15])*(pt_attr - dCentroid[centIndx+15]);  \
}

#define UPDATE_CLOSEST(len)   {       \
  if(dist[0] < min_dist){             \
    min_dist = dist[0];               \
    closest = centIndx;               \
  }                                   \
  if(len>1 && dist[1] < min_dist){    \
    min_dist = dist[1];               \
    closest = centIndx+1;             \
  }                                   \
  if(len>2 && dist[2] < min_dist){    \
    min_dist = dist[2];               \
    closest = centIndx+2;             \
  }                                   \
  if(len>3 && dist[3] < min_dist){    \
    min_dist = dist[3];               \
    closest = centIndx+3;             \
  }                                   \
  if(len>4 && dist[4] < min_dist){    \
    min_dist = dist[4];               \
    closest = centIndx+4;             \
  }                                   \
  if(len>5 && dist[5] < min_dist){    \
    min_dist = dist[5];               \
    closest = centIndx+5;             \
  }                                   \
  if(len>6 && dist[6] < min_dist){    \
    min_dist = dist[6];               \
    closest = centIndx+6;             \
  }                                   \
  if(len>7 && dist[7] < min_dist){    \
    min_dist = dist[7];               \
    closest = centIndx+7;             \
  }                                   \
  if(len>8 && dist[8] < min_dist){    \
    min_dist = dist[8];               \
    closest = centIndx+8;             \
  }                                   \
  if(len>9 && dist[9] < min_dist){    \
    min_dist = dist[9];               \
    closest = centIndx+9;             \
  }                                   \
  if(len>10 && dist[10] < min_dist){  \
    min_dist = dist[10];              \
    closest = centIndx+10;            \
  }                                   \
  if(len>11 && dist[11] < min_dist){  \
    min_dist = dist[11];              \
    closest = centIndx+11;            \
  }                                   \
  if(len>12 && dist[12] < min_dist){  \
    min_dist = dist[12];              \
    closest = centIndx+12;            \
  }                                   \
  if(len>13 && dist[13] < min_dist){  \
    min_dist = dist[13];              \
    closest = centIndx+13;            \
  }                                   \
  if(len>14 && dist[14] < min_dist){  \
    min_dist = dist[14];              \
    closest = centIndx+14;            \
  }                                   \
  if(len>15 && dist[15] < min_dist){  \
    min_dist = dist[15];              \
    closest = centIndx+15;            \
  }                                   \
}

__constant__ float dCentroid[12000];   // The max possible size is 65536 bytes. We are leaving some empty space for compiler to use.

template<int distLen>
__device__ int getClosest( int k, int dimensions, float *Data, size_t pitchData, int closest, float min_dist)
{
  float dist[distLen];
  float point;

  INIT_DIST(distLen)

  int centIndx = k - distLen;
  int remDim = dimensions;
  #pragma unroll 4
  for(remDim; remDim>0; remDim--){
    LOAD_DATA()
    UPDATE_DIST(point,distLen)
    centIndx+=k;
  }

  centIndx = k - distLen;
  UPDATE_CLOSEST(distLen)

  return closest;
}

template <int distLen>
__global__ void assign_labels( int n, int k, int dimensions, float *dData, int *Index, size_t pitchData)
{
  float dist[distLen];

  float point;
  const int pt_i = blockIdx.x * blockDim.x + threadIdx.x;
  float* Data = dData;

  if(pt_i < n) {

    float min_dist = FLT_MAX;
    int closest = 0;
    int remCent = k;
    for( remCent; remCent >= distLen; remCent -= distLen )
    {
      INIT_DATA()
      INIT_DIST(distLen)

      int centIndx = k - remCent;
      int remDim = dimensions;
      #pragma unroll 4
      for(remDim; remDim>0; remDim--){
        LOAD_DATA()
        UPDATE_DIST(point,distLen)
        centIndx +=k;
      }

      centIndx = k - remCent;
      UPDATE_CLOSEST(distLen)
    }

    INIT_DATA();
    switch(remCent){
      case 15:  closest = getClosest<15>(k,dimensions,Data,pitchData,closest,min_dist);
                break;
      case 14:  closest = getClosest<14>(k,dimensions,Data,pitchData,closest,min_dist);
                break;
      case 13:  closest = getClosest<13>(k,dimensions,Data,pitchData,closest,min_dist);
                break;
      case 12:  closest = getClosest<12>(k,dimensions,Data,pitchData,closest,min_dist);
                break;
      case 11:  closest = getClosest<11>(k,dimensions,Data,pitchData,closest,min_dist);
                break;
      case 10:  closest = getClosest<10>(k,dimensions,Data,pitchData,closest,min_dist);
                break;
      case 9:   closest = getClosest<9>(k,dimensions,Data,pitchData,closest,min_dist);
                break;
      case 8:   closest = getClosest<8>(k,dimensions,Data,pitchData,closest,min_dist);
                break;
      case 7:   closest = getClosest<7>(k,dimensions,Data,pitchData,closest,min_dist);
                break;
      case 6:   closest = getClosest<6>(k,dimensions,Data,pitchData,closest,min_dist);
                break;
      case 5:   closest = getClosest<5>(k,dimensions,Data,pitchData,closest,min_dist);
                break;
      case 4:   closest = getClosest<4>(k,dimensions,Data,pitchData,closest,min_dist);
                break;
      case 3:   closest = getClosest<3>(k,dimensions,Data,pitchData,closest,min_dist);
                break;
      case 2:   closest = getClosest<2>(k,dimensions,Data,pitchData,closest,min_dist);
                break;
      case 1:   closest = getClosest<1>(k,dimensions,Data,pitchData,closest,min_dist);
                break;
    }

    Index[pt_i] = closest;

  }
}


#define INVOKE_ASSIGN_LABELS_DEV(i)                                                         \
  cudaFuncSetCacheConfig(assign_labels<i>, cudaFuncCachePreferL1);                          \
  assign_labels<i><<<dim3(blockDim),dim3(threadDim)>>>( n, k, d, dData, Index, pitchData);  \
  break;


void callAssignLabels(int n, int k, int d, float *dData, int *Index, size_t pitchData, int distLen, int blockDim,
                      int threadDim)  {
    switch(distLen)
    {
      case 1:    INVOKE_ASSIGN_LABELS_DEV(1)
      case 2:    INVOKE_ASSIGN_LABELS_DEV(2)
      case 3:    INVOKE_ASSIGN_LABELS_DEV(3)
      case 4:    INVOKE_ASSIGN_LABELS_DEV(4)
      case 5:    INVOKE_ASSIGN_LABELS_DEV(5)
      case 6:    INVOKE_ASSIGN_LABELS_DEV(6)
      case 7:    INVOKE_ASSIGN_LABELS_DEV(7)
      case 8:    INVOKE_ASSIGN_LABELS_DEV(8)
      case 9:    INVOKE_ASSIGN_LABELS_DEV(9)
      case 10:   INVOKE_ASSIGN_LABELS_DEV(10)
      case 11:   INVOKE_ASSIGN_LABELS_DEV(11)
      case 12:   INVOKE_ASSIGN_LABELS_DEV(12)
      case 13:   INVOKE_ASSIGN_LABELS_DEV(13)
      case 14:   INVOKE_ASSIGN_LABELS_DEV(14)
      case 15:   INVOKE_ASSIGN_LABELS_DEV(15)
      case 16:   INVOKE_ASSIGN_LABELS_DEV(16)
      default:  printf("Only distLen ranging from 1 to 16 is supported.\n");
                exit(EXIT_FAILURE);
    }
}
