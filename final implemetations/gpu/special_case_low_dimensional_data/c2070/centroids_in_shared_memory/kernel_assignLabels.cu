/*************************************************************************/
/**   File:         kmeans_clustering.c                                 **/
/**   Description:  Implementation of find cluster for k-means          **/
/**                 clustering algorithm                                **/
/*************************************************************************/

#include "kmeans.h"

#ifndef FLT_MAX
#define FLT_MAX 3.40282347e+38
#endif


#ifdef ACCESS_CENT_COALESCED
  #define EXPRESSION(i)  {                                                    \
    dist+=(point[i]-dCentroid[centIndx])*(point[i]-dCentroid[centIndx]);      \
    centIndx += k;                                                            \
   }
#else
  #define EXPRESSION(i)  {                                                    \
    dist+=(point[i]-dCentroid[centIndx+i])*(point[i]-dCentroid[centIndx+i]);  \
   }
#endif

#ifdef ACCESS_DATA_COALESCED
  #define INIT_VARS(i)  {                       \
    point[i]=Data[0];                           \
    Data = (float *)((char*)Data + pitchData);  \
   }
#else
  #define INIT_VARS(i)  {                       \
    point[i]=Data[i];                           \
   }
#endif



template <int dimensions>
__global__ void assign_labels( int n, int k, float *gCentroid, float *dData, int *Index, size_t pitchData)
{

  float point[dimensions];
  __shared__ float dCentroid[4000];

#ifdef ACCESS_CENT_COALESCED
  float *tmp_gCentroid = gCentroid;
  float *tmp_dCentroid = dCentroid;
  for(int attr_i = 0; attr_i<dimensions; attr_i++)
  {
    for (int i = threadIdx.x; i<k; i+=blockDim.x){
      tmp_dCentroid[i] = tmp_gCentroid[i];
    }
    tmp_gCentroid += k;
    tmp_dCentroid += k;
  }
#else
  for (int i = threadIdx.x; i<k*dimensions; i+=blockDim.x){
    dCentroid[i] = gCentroid[i];
  }
#endif
  __syncthreads();

  const int pt_i = blockIdx.x * blockDim.x + threadIdx.x;

  if(pt_i < n) {
  float* Data = dData;
#ifdef ACCESS_DATA_COALESCED
  Data = Data + pt_i;       // coalesced access from global memory
#else
  Data = (float*)((char*)Data + pt_i*pitchData);  // non-coalesced access from global memory
#endif


  INIT_VARS(0);

  if(dimensions>1)
    INIT_VARS(1)

  if(dimensions>2)
    INIT_VARS(2)

  if(dimensions>3)
    INIT_VARS(3)

  if(dimensions>4)
    INIT_VARS(4)

  if(dimensions>5)
    INIT_VARS(5)

  if(dimensions>6)
    INIT_VARS(6)

  if(dimensions>7)
    INIT_VARS(7)

  if(dimensions>8)
    INIT_VARS(8)

  if(dimensions>9)
    INIT_VARS(9)

  if(dimensions>10)
    INIT_VARS(10)

  if(dimensions>11)
    INIT_VARS(11)

  if(dimensions>12)
    INIT_VARS(12)

  if(dimensions>13)
    INIT_VARS(13)

  if(dimensions>14)
    INIT_VARS(14)

  if(dimensions>15)
    INIT_VARS(15)

  if(dimensions>16)
    INIT_VARS(16)

  if(dimensions>17)
    INIT_VARS(17)

  if(dimensions>18)
    INIT_VARS(18)

  if(dimensions>19)
    INIT_VARS(19)

  if(dimensions>20)
    INIT_VARS(20)

  if(dimensions>21)
    INIT_VARS(21)

  int centIndx = 0;

  float min_dist = FLT_MAX;
  int closest = 0;
  #pragma unroll 32
  for(int cent_i=0; cent_i< k; cent_i++)
  {
    float dist=0;

    EXPRESSION(0)
    if(dimensions>1)
      EXPRESSION(1)

    if(dimensions>2)
      EXPRESSION(2)

    if(dimensions>3)
      EXPRESSION(3)

    if(dimensions>4)
      EXPRESSION(4)

    if(dimensions>5)
      EXPRESSION(5)

    if(dimensions>6)
      EXPRESSION(6)

    if(dimensions>7)
      EXPRESSION(7)

    if(dimensions>8)
      EXPRESSION(8)

    if(dimensions>9)
      EXPRESSION(9)

    if(dimensions>10)
      EXPRESSION(10)

    if(dimensions>11)
      EXPRESSION(11)

    if(dimensions>12)
      EXPRESSION(12)

    if(dimensions>13)
      EXPRESSION(13)

    if(dimensions>14)
      EXPRESSION(14)

    if(dimensions>15)
      EXPRESSION(15)

    if(dimensions>16)
      EXPRESSION(16)

    if(dimensions>17)
      EXPRESSION(17)

    if(dimensions>18)
      EXPRESSION(18)

    if(dimensions>19)
      EXPRESSION(19)

    if(dimensions>20)
      EXPRESSION(20)

    if(dimensions>21)
      EXPRESSION(21)

    if(dist < min_dist) {
      min_dist = dist;
      closest = cent_i;
    }

#ifdef ACCESS_CENT_COALESCED
    centIndx = cent_i+1;
#else
    //Centroid = Centroid + dimensions;
    centIndx = centIndx + dimensions;
#endif
  }

  Index[pt_i] = closest;
  }
}


#define INVOKE_ASSIGN_LABELS_DEV(i)                                                                 \
  cudaFuncSetCacheConfig(assign_labels<i>, cudaFuncCachePreferL1);                                  \
  assign_labels<i><<<dim3(blockDim),dim3(threadDim)>>>( n, k, gCentroid, dData, Index, pitchData);  \
  break;


void callAssignLabels(int n, int k, float *gCentroid, float *dData, int *Index, size_t pitchData, int d, int blockDim,
                      int threadDim)  {
    switch(d)
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
      case 17:   INVOKE_ASSIGN_LABELS_DEV(17)
      case 18:   INVOKE_ASSIGN_LABELS_DEV(18)
      case 19:   INVOKE_ASSIGN_LABELS_DEV(19)
      case 20:   INVOKE_ASSIGN_LABELS_DEV(20)
      case 21:   INVOKE_ASSIGN_LABELS_DEV(21)
      case 22:   INVOKE_ASSIGN_LABELS_DEV(22)
      default:  printf("Only data points with dimension ranging from 1 to 22 are supported.\n");
                exit(EXIT_FAILURE);
    }
}
