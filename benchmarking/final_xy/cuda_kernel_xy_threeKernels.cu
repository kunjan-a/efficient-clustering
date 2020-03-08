#include <stdio.h>
#include <settings.h>

#define CONFLICT_FREE_INDEX(n) ( (n) + ((n) >> LOG_NUM_BANKS))


////////////////////////////////////////////////////////////////////////////////
//!Number of blocks (MAX_BLOCKS defines the upper limit)
//! @param n number of points
////////////////////////////////////////////////////////////////////////////////
__device__ __host__ int get_num_blocks(int n){
    return 1 + ( (n-1) >> LOG_BLOCKDIM );                                                               //Must be atleast equal to number of GPU cores. Can be more with lesser threads per block if there are too many synchronize calls required so that there is minimum relative blocking among threads.

}

////////////////////////////////////////////////////////////////////////////////
//!Number of blocks rounded to nearest multiple of HalfWarp i.e. ceil(num_blocks/HalfWarp)*HalfWarp
//! @param n number of points
////////////////////////////////////////////////////////////////////////////////
__device__ __host__ unsigned int get_mem_length(int n){
    int blocks = get_num_blocks(n);
    return ((((blocks-1)>>LOG_HALF_WARP) + 1) << LOG_HALF_WARP);                                        //MemLength gives number of blocks rounded to nearest multiple of HalfWarp i.e. ceil(num_blocks/HalfWarp)*HalfWarp
}



#if TRY_ATOMIC_CNTR_IN_LOCAL_REDCN == 1
__device__ int inline reduceThreads(const unsigned int checkFor, float *const __restrict__ sumx, float *const __restrict__ sumy,
                                    const float *const __restrict__ s_thrd_x, const float *const __restrict__ s_thrd_y,
                                    const int *const __restrict__ s_thrd_k, unsigned int * const __restrict__ points_checked)
{
    int count=0;

    {
//TODO: Unroll the following for loop, if unrolling does not cause memory spilling.
        for (int i = 0; (*points_checked) < BLOCKDIM && i < BLOCKDIM; ++i)
        {
            int indx=i+checkFor;
            if (indx >= BLOCKDIM)
                indx -= BLOCKDIM;
            if (checkFor == s_thrd_k[indx])                                                          // Coalesced access from shared mem.
            {
                // Only one thread from all the warps wud enter in, all other warps wud move to the next value of i.
                // In all WarpSize -1 threads wait here every time one of the threads meets the if-condition for some i.
                // Max. no. of warps that can be simultaneously in this if-cond while executing for diff. values of i is: 2*(broad. shared access) + 3*floating pt. arithmetic instr

                *sumx = (*sumx) + s_thrd_x[indx];
                *sumy = (*sumy) + s_thrd_y[indx];
                count++;
                atomicInc(points_checked,BLOCKDIM);
            }
        }
    }
    return count;
}
#else


__device__ int inline reduceThreads(const unsigned int checkFor, float *const __restrict__ sumx, float *const __restrict__ sumy,
                                    const float *const __restrict__ s_thrd_x, const float *const __restrict__ s_thrd_y,
                                    const int *const __restrict__ s_thrd_k)
{
    int count=0;

    {
//TODO: Unroll the following for loop, if unrolling does not cause memory spilling.
        for (int i = 0; i < BLOCKDIM; ++i)
        {
#if USE_BRDCST == 0
            int indx=i+checkFor;
            if (indx >= BLOCKDIM)
                indx -= BLOCKDIM;
            if (checkFor == s_thrd_k[indx])                                                          // Coalesced access from shared mem.
            {
                // Only one thread from all the warps wud enter in, all other warps wud move to the next value of i.
                // In all WarpSize -1 threads wait here every time one of the threads meets the if-condition for some i.
                // Max. no. of warps that can be simultaneously in this if-cond while ececuting for diff. values of i is: 2*(broad. shared access) + 3*floating pt. arithmetic instr

                *sumx = (*sumx) + s_thrd_x[indx];
                *sumy = (*sumy) + s_thrd_y[indx];
                count++;
            }
#else
        // Broadcast access from shared mem. This line shud be executed by all threads in warp simultaneously.
            if (checkFor == s_thrd_k[i])
            {
                *sumx = (*sumx) + s_thrd_x[i];
                *sumy = (*sumy) + s_thrd_y[i];
                count++;
            }
#endif
        }
    }

    return count;
}
#endif


/*__device__ float distance(const float &x1, const float &y1, const float &x2, const float &y2)
{
    return sqrtf((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1));
}*/


__global__ void
cluster_withNewReduction(const int n, const int k, const float *const __restrict__ d_dataptx, const float *const __restrict__ d_datapty,
                         float *const __restrict__ d_centroidx, float *const __restrict__ d_centroidy, float *const __restrict__ d_sumx,
                         float *const __restrict__ d_sumy, int *const __restrict__ d_count)
{
    // shared memory
    // Stores x-coord of all clusters followed by y-coord of all clusters i.e. x and y coord of a cluster are separated by k.
    extern __shared__ float s_centroid[];

    __shared__ float s_thrd[(BLOCKDIM<<1)];                                               // Store the x-coord of data point processed, for all thrds followed by y-coord.
    // No point of going for short int as each bank in shared mem. is 32-bit wide and only one thread
    // can access one bank at a time, so no. of accesses wont be reduced
    __shared__ volatile signed int s_thrd_k[(BLOCKDIM)];                                           // Store the nearest cluster of point checked by this thrd.

    volatile float *const s_centroidx = s_centroid;                                                                  // x-coord of 1st cluster's centroid
    volatile float *const s_centroidy = s_centroid+k;                                                                // y-coord of 1st cluster's centroid

    volatile float *const s_thrd_x = s_thrd;                                                                // x-coord of points checked by thrds
    volatile float *const s_thrd_y = s_thrd+BLOCKDIM;                                                       // y-coord of points checked by thrds
    // We have declard all of these as volatile bcoz we will be reading them in each iteration
    // and we want new values to be read instead of the old values stored in registers by the previous iterations.

    float ptx,pty;
    int closestNum;

    // If we've T threads per block then index of ith thread in kth block is k*T + i. So we are maintaining an order among all threads of all blocks.
    // This is same as blockIdx.x * BLOCKDIM + threadId.x
    const unsigned int index = (blockIdx.x  << LOG_BLOCKDIM) + threadIdx.x;
    const unsigned int mem_length = get_mem_length(n);

    //Assign the values of the global memory centroids to the shared memory centroids for all k clusters
    //coalesced access
    __syncthreads();                                                                                        //Ensure all the threads in the same warp of a block start simultaneously for next instruction
    for (int i = threadIdx.x; i < k; i+=BLOCKDIM)
    {
        s_centroidx[i] = d_centroidx[i];
        s_centroidy[i] = d_centroidy[i];
    }
    __syncthreads();                                                                                        // BLOCKDIM-k%BLOCKDIM threads would have waited for maximum one iteration of above loop over here while others executes the above lines


    float centroid_dist,new_centroid_dist;
    if (index<n)                                                                                            //Find centroid nearest to the datapoint at location index
    {
        ptx = d_dataptx[index];
        pty = d_datapty[index];
        s_thrd_x[threadIdx.x]=ptx;
        s_thrd_y[threadIdx.x]=pty;

        closestNum = 0;
        centroid_dist=distance( ptx, pty, s_centroidx[closestNum], s_centroidy[closestNum]);
        for (int i =  1; i < k; ++i)
        {
            new_centroid_dist=distance( ptx, pty, s_centroidx[i], s_centroidy[i]);
            if (  new_centroid_dist < centroid_dist )
            {
                centroid_dist=new_centroid_dist;
                closestNum = i;
            }
        }
         // Store the index of the centroid closest to the datapoint.
        s_thrd_k[threadIdx.x]=closestNum;
    }else
        s_thrd_k[threadIdx.x]=-1;

#if TRY_ATOMIC_CNTR_IN_LOCAL_REDCN == 1
        __shared__ unsigned int points_checked;
        if(threadIdx.x == 0)
          points_checked=0;
#endif
        __syncthreads();                                                                                    // Wait for all thrds to finish setting closest k value


    float sumx=0;
    float sumy =0;
    int count=0;
    for(int checkFor=threadIdx.x;checkFor<k;checkFor+=BLOCKDIM)
    {
        sumx=0;
        sumy =0;
        count=0;
#if TRY_ATOMIC_CNTR_IN_LOCAL_REDCN == 1
        count += reduceThreads(checkFor, &sumx, &sumy, (float *)s_thrd_x, (float *)s_thrd_y, (int *)s_thrd_k,&points_checked);
#else
        count += reduceThreads(checkFor, &sumx, &sumy, (float *)s_thrd_x, (float *)s_thrd_y, (int *)s_thrd_k);
#endif


        d_count[ checkFor * mem_length + blockIdx.x ] = count;
        d_sumx[ checkFor * mem_length + blockIdx.x ] = sumx;
        d_sumy[ checkFor * mem_length + blockIdx.x ] = sumy;

    }

}

#define REDUCE_EXP(tid, stride) if (blockSize >= (stride<<1)) { if (tid < stride) { s_sumx[tid] = *sumx = *sumx + s_sumx[tid + stride];\
                                                                                    s_sumy[tid] = *sumy = *sumy + s_sumy[tid + stride];\
                                                                                    s_count[tid] = *count = *count + s_count[tid + stride];\
                                                                                  } __syncthreads();\
                                                              }

#define REDUCE_EXP_NOSYNC(tid, stride) if (blockSize >= (stride<<1)){   s_sumx[tid] = *sumx = *sumx + s_sumx[tid + stride];\
                                                                        s_sumy[tid] = *sumy = *sumy + s_sumy[tid + stride];\
                                                                        s_count[tid] = *count = *count + s_count[tid + stride];\
                                                                    }

#define REDUCE_EXP_BIT(tid, stride) if (blockSize >= (stride<<1)) { if (tid < stride) { s_sumx[tid] = *sumx = *sumx + s_sumx[tid | stride];\
                                                                                    s_sumy[tid] = *sumy = *sumy + s_sumy[tid | stride];\
                                                                                    s_count[tid] = *count = *count + s_count[tid | stride];\
                                                                                  } __syncthreads();\
                                                              }

#define REDUCE_EXP_NOSYNC_BIT(tid, stride) if (blockSize >= (stride<<1)){   s_sumx[tid] = *sumx = *sumx + s_sumx[tid | stride];\
                                                                        s_sumy[tid] = *sumy = *sumy + s_sumy[tid | stride];\
                                                                        s_count[tid] = *count = *count + s_count[tid | stride];\
                                                                    }

#define REDUCE_EXP_LESS_SHARED(tid, stride) if (blockSize >= (stride<<1)) { if (tid < stride) { *sumx = *sumx + s_sumx[tid + stride];\
                                                                                    *sumy = *sumy + s_sumy[tid + stride];\
                                                                                    *count = *count + s_count[tid + stride];\
                                                                                    if (tid >= (stride>>1) ) { s_sumx[tid] = *sumx;\
                                                                                            s_sumy[tid] = *sumy;\
                                                                                            s_count[tid] = *count;\
                                                                                          } __syncthreads();\
                                                                                  };\
                                                              }

#define REDUCE_EXP_NOSYNC_LESS_SHARED(tid, stride) if (blockSize >= (stride<<1)){ if (tid < stride) { *sumx = *sumx + s_sumx[tid + stride];\
                                                                                    *sumy = *sumy + s_sumy[tid + stride];\
                                                                                    *count = *count + s_count[tid + stride];\
                                                                                  };\
                                                                                  if (tid >= (stride>>1) ) { s_sumx[tid] = *sumx;\
                                                                                    s_sumy[tid] = *sumy;\
                                                                                    s_count[tid] = *count;\
                                                                                  } __syncthreads();\
                                                                    }

template <unsigned int blockSize>
__device__ inline void
reduce(float *const __restrict__ sumx, float *const __restrict__ sumy, int *const __restrict__ count,
       volatile float *const __restrict__ s_sumx, volatile float *const __restrict__ s_sumy,
       volatile int *const __restrict__ s_count)
{
    __syncthreads();
//    float sumx,sumy;
//    int count;

//    if (blockSize >= 1024) { if (threadIdx.x< 512) {
//                                                    s_sumx[threadIdx.x] = *sumx = *sumx + s_sumx[threadIdx.x+ 512];
//                                                    s_sumy[threadIdx.x] = *sumy = *sumy + s_sumy[threadIdx.x+ 512];
//                                                    s_count[threadIdx.x] = *count = *count + s_count[threadIdx.x+ 512];
//                                                  } __syncthreads();
//                          }
//
//    if (blockSize >= 512) { if (threadIdx.x< 256) {
//                                                    s_sumx[threadIdx.x] = *sumx = *sumx + s_sumx[threadIdx.x+ 256];
//                                                    s_sumy[threadIdx.x] = *sumy = *sumy + s_sumy[threadIdx.x+ 256];
//                                                    s_count[threadIdx.x] = *count = *count + s_count[threadIdx.x+ 256];
//                                                  } __syncthreads();
//                          }
//
//    if (blockSize >= 256) { if (threadIdx.x< 128) {
//                                                    s_sumx[threadIdx.x] = sumx = sumx + s_sumx[threadIdx.x+ 128];
//                                                    s_sumy[threadIdx.x] = sumy = sumy + s_sumy[threadIdx.x+ 128];
//                                                    s_count[threadIdx.x] = count = count + s_count[threadIdx.x+ 128];
//                                                  } __syncthreads();
//                          }
//
//    if (blockSize >= 128) { if (threadIdx.x< 64) {
//                                                    s_sumx[threadIdx.x] = sumx = sumx + s_sumx[threadIdx.x+ 64];
//                                                    s_sumy[threadIdx.x] = sumy = sumy + s_sumy[threadIdx.x+ 64];
//                                                    s_count[threadIdx.x] = count = count + s_count[threadIdx.x+ 64];
//                                                  } __syncthreads();
//                          }

    // do reduction in shared mem
#if optimizeTree == 0
    REDUCE_EXP(threadIdx.x, 512)
    REDUCE_EXP(threadIdx.x, 256)
    REDUCE_EXP(threadIdx.x, 128)
    REDUCE_EXP(threadIdx.x, 64)
#else
//    REDUCE_EXP_BIT(threadIdx.x, 512)
//    REDUCE_EXP_BIT(threadIdx.x, 256)
//    REDUCE_EXP_BIT(threadIdx.x, 128)
//    REDUCE_EXP_BIT(threadIdx.x, 64)
//
    REDUCE_EXP_LESS_SHARED(threadIdx.x, 512)
    REDUCE_EXP_LESS_SHARED(threadIdx.x, 256)
    REDUCE_EXP_LESS_SHARED(threadIdx.x, 128)
    REDUCE_EXP_LESS_SHARED(threadIdx.x, 64)
#endif
    if (threadIdx.x < 32)
    {
//        if (blockSize >= 64){
//                              s_sumx[threadIdx.x] = sumx = sumx + s_sumx[threadIdx.x + 32];
//                              s_sumy[threadIdx.x] = sumy = sumy + s_sumy[threadIdx.x + 32];
//                              s_count[threadIdx.x] = count = count + s_count[threadIdx.x + 32];
//                            }
//
//        if (blockSize >= 32){
//                              s_sumx[threadIdx.x] = sumx = sumx + s_sumx[threadIdx.x + 16];
//                              s_sumy[threadIdx.x] = sumy = sumy + s_sumy[threadIdx.x + 16];
//                              s_count[threadIdx.x] = count = count + s_count[threadIdx.x + 16];
//                            }
//
//        if (blockSize >= 16){
//                              s_sumx[threadIdx.x] = sumx = sumx + s_sumx[threadIdx.x + 8];
//                              s_sumy[threadIdx.x] = sumy = sumy + s_sumy[threadIdx.x + 8];
//                              s_count[threadIdx.x] = count = count + s_count[threadIdx.x + 8];
//                            }
//
//        if (blockSize >= 8){
//                              s_sumx[threadIdx.x] = sumx = sumx + s_sumx[threadIdx.x + 4];
//                              s_sumy[threadIdx.x] = sumy = sumy + s_sumy[threadIdx.x + 4];
//                              s_count[threadIdx.x] = count = count + s_count[threadIdx.x + 4];
//                            }
//
//        if (blockSize >= 4){
//                              s_sumx[threadIdx.x] = sumx = sumx + s_sumx[threadIdx.x + 2];
//                              s_sumy[threadIdx.x] = sumy = sumy + s_sumy[threadIdx.x + 2];
//                              s_count[threadIdx.x] = count = count + s_count[threadIdx.x + 2];
//                            }
//
//        if (blockSize >= 2){
//                              s_sumx[threadIdx.x] = sumx = sumx + s_sumx[threadIdx.x + 1];
//                              s_sumy[threadIdx.x] = sumy = sumy + s_sumy[threadIdx.x + 1];
//                              s_count[threadIdx.x] = count = count + s_count[threadIdx.x + 1];
//                            }

        REDUCE_EXP_NOSYNC(threadIdx.x, 32)
        REDUCE_EXP_NOSYNC(threadIdx.x, 16)
        REDUCE_EXP_NOSYNC(threadIdx.x,  8)
        REDUCE_EXP_NOSYNC(threadIdx.x,  4)
        REDUCE_EXP_NOSYNC(threadIdx.x,  2)
        REDUCE_EXP_NOSYNC(threadIdx.x,  1)

//        REDUCE_EXP_NOSYNC_BIT(threadIdx.x, 32)
//        REDUCE_EXP_NOSYNC_BIT(threadIdx.x, 16)
//        REDUCE_EXP_NOSYNC_BIT(threadIdx.x,  8)
//        REDUCE_EXP_NOSYNC_BIT(threadIdx.x,  4)
//        REDUCE_EXP_NOSYNC_BIT(threadIdx.x,  2)
//        REDUCE_EXP_NOSYNC_BIT(threadIdx.x,  1)

//        REDUCE_EXP_NOSYNC_LESS_SHARED(threadIdx.x, 32)
//        REDUCE_EXP_NOSYNC_LESS_SHARED(threadIdx.x, 16)
//        REDUCE_EXP_NOSYNC_LESS_SHARED(threadIdx.x,  8)
//        REDUCE_EXP_NOSYNC_LESS_SHARED(threadIdx.x,  4)
//        REDUCE_EXP_NOSYNC_LESS_SHARED(threadIdx.x,  2)
//        REDUCE_EXP_NOSYNC_LESS_SHARED(threadIdx.x,  1)

    }

}



__global__ void
cluster_withTreeReduction(const int n, const int k, const float *const __restrict__ d_dataptx, const float *const __restrict__ d_datapty,
                          float *const __restrict__ d_centroidx, float *const __restrict__ d_centroidy, float *const __restrict__ d_sumx,
                          float *const __restrict__ d_sumy, int *const __restrict__ d_count)
{
    // shared memory
    extern __shared__ float s_centroid[];                                                                   //Stores x-coord of all clusters followed by y-coord of all clusters i.e. x and y coord of a cluster are separated by k.

                                                                                                            //We have multiplied the number of threads by 2 to ensure that length is >= ceil(log(no. of threads)), to help find bank-conflict free index
    __shared__  volatile float s_sumx[BLOCKDIM<<1];                                                         //Stores the sum of x-coordinates of all the points for each thread of this block
    __shared__  volatile float s_sumy[BLOCKDIM<<1];                                                         //Stores the sum of y-coordinates of all the points for each thread of this block
    __shared__  volatile int s_count[BLOCKDIM<<1];                                                          //Stores the no. of points in for each thread of this block

    const unsigned int index = (blockIdx.x << LOG_BLOCKDIM) + threadIdx.x;                                  //If we've T threads per block then index of ith thread in kth block is k*T + i. So we are maintaining an order among all threads of all blocks.
                                                                                                            // Same as blockIdx.x * blockDim.x + threadId.x

    const unsigned int mem_length = get_mem_length(n);

    float *const s_centroidx = s_centroid;                                                                  //x-coord of 1st cluster's centroid
    float *const s_centroidy = s_centroid+k;                                                                //y-coord of 1st cluster's centroid
    float ptx,pty;
    int closestNum;


    //Assign the values of the global memory centroids to the shared memory centroids for all k clusters
    //coalesced access
    __syncthreads();                                                                                        //Ensure all the threads in the same warp of a block start simultaneously for next instruction
    for (int i = threadIdx.x; i < k; i+=BLOCKDIM)
      {
          s_centroidx[i] = d_centroidx[i];
          s_centroidy[i] = d_centroidy[i];
      }
    __syncthreads();                                                                                        // BLOCKDIM-k%BLOCKDIM threads would have waited for maximum one iteration of above loop over here while others executes the above lines



    float centroid_dist,new_centroid_dist;
    if (index<n)                                                                                            //Find centroid nearest to the datapoint at location index
    {
        ptx = d_dataptx[index];
        pty = d_datapty[index];
        closestNum = 0;
        centroid_dist=distance( ptx, pty, s_centroidx[closestNum], s_centroidy[closestNum]);
        for (int i =  1; i < k; ++i)
        {
            new_centroid_dist=distance( ptx, pty, s_centroidx[i], s_centroidy[i]);
            if (  new_centroid_dist < centroid_dist )
            {
                centroid_dist=new_centroid_dist;
                closestNum = i;
            }
        }
    }else
      closestNum = -1;

    __syncthreads();

    float sumx,sumy;
    int count;
    for (int i = 0; i < k; ++ i)
    {
        if (closestNum==i)
        {
#if optimizeTree == 0
            s_count[threadIdx.x] = count = 1;
            s_sumx[threadIdx.x] = sumx = ptx;
            s_sumy[threadIdx.x] = sumy = pty;
#else
            count = 1;
            sumx = ptx;
            sumy = pty;
#endif
        }
        else
        {
#if optimizeTree == 0
            s_count[threadIdx.x] = count = 0;
            s_sumx[threadIdx.x] = s_sumy[threadIdx.x] = sumx = sumy = 0.0f;
#else
            count = 0;
            sumx = sumy = 0.0f;
#endif
        }
#if optimizeTree == 1
        if(threadIdx.x >= BLOCKDIM>>1){
            s_count[threadIdx.x] = count;
            s_sumx[threadIdx.x] = sumx;
            s_sumy[threadIdx.x] = sumy;
        }
#endif
        switch (BLOCKDIM)
        {
          case 1024:
              reduce<1024>(&sumx, &sumy, &count, s_sumx, s_sumy, s_count); break;
          case 512:
              reduce<512>(&sumx, &sumy, &count, s_sumx, s_sumy, s_count); break;
          case 256:
              reduce<256>(&sumx, &sumy, &count, s_sumx, s_sumy, s_count); break;
          case 128:
              reduce<128>(&sumx, &sumy, &count, s_sumx, s_sumy, s_count); break;
          case 64:
              reduce<64>(&sumx, &sumy, &count, s_sumx, s_sumy, s_count); break;
          case 32:
              reduce<32>(&sumx, &sumy, &count, s_sumx, s_sumy, s_count); break;
          case 16:
              reduce<16>(&sumx, &sumy, &count, s_sumx, s_sumy, s_count); break;
          case  8:
              reduce<8>(&sumx, &sumy, &count, s_sumx, s_sumy, s_count); break;
          case  4:
              reduce<4>(&sumx, &sumy, &count, s_sumx, s_sumy, s_count); break;
          case  2:
              reduce<2>(&sumx, &sumy, &count, s_sumx, s_sumy, s_count); break;
          case  1:
              reduce<1>(&sumx, &sumy, &count, s_sumx, s_sumy, s_count); break;
        }

        __syncthreads();

        if (threadIdx.x==0)
        {
            d_count[ i * mem_length + blockIdx.x ] = count;//s_count[0];
            d_sumx[ i * mem_length + blockIdx.x ] = sumx;//s_sumx[0];
            d_sumy[ i * mem_length + blockIdx.x ] = sumy;//s_sumy[0];
        }
    }
}


__global__ void
cluster(int n, int k, const float *const d_dataptx, const float *const d_datapty, float *d_centroidx, float *d_centroidy, float *d_sumx, float *d_sumy, int *d_count)
{
    // shared memory
    extern __shared__ float s_centroid[];                                                                   //Stores x-coord of all clusters followed by y-coord of all clusters i.e. x and y coord of a cluster are separated by k.

                                                                                                            //We have multiplied the number of threads by 2 to ensure that length is >= ceil(log(no. of threads)), to help find bank-conflict free index
    __shared__  float s_sumx[BLOCKDIM<<1];                                                                  //Stores the sum of x-coordinates of all the points for each thread of this block
    __shared__  float s_sumy[BLOCKDIM<<1];                                                                  //Stores the sum of y-coordinates of all the points for each thread of this block
    __shared__  float s_count[BLOCKDIM<<1];                                                                 //Stores the no. of points in for each thread of this block

    const unsigned int index = (blockIdx.x << LOG_BLOCKDIM) + threadIdx.x;                                  //If we've T threads per block then index of ith thread in kth block is k*T + i. So we are maintaining an order among all threads of all blocks.
                                                                                                            // Same as blockIdx.x * blockDim.x + threadId.x

    const unsigned int mem_length = get_mem_length(n);

    float *const s_centroidx = s_centroid;                                                                  //x-coord of 1st cluster's centroid
    float *const s_centroidy = s_centroid+k;                                                                //y-coord of 1st cluster's centroid
    float ptx,pty;
    int closestNum;


    //Assign the values of the global memory centroids to the shared memory centroids for all k clusters
    //coalesced access
    __syncthreads();                                                                                        //Ensure all the threads in the same warp of a block start simultaneously for next instruction
    for (int i = 0; i < k; i+=BLOCKDIM)
        if ( i + threadIdx.x < k)
        {
            s_centroidx[i+threadIdx.x] = d_centroidx[i+threadIdx.x];
            s_centroidy[i+threadIdx.x] = d_centroidy[i+threadIdx.x];
        }
    __syncthreads();                                                                                        // BLOCKDIM-k%BLOCKDIM threads would have waited for maximum one iteration of above loop over here while others executes the above lines



    float centroid_dist,new_centroid_dist;
    if (index<n)                                                                                            //Find centroid nearest to the datapoint at location index
    {
        ptx = d_dataptx[index];
        pty = d_datapty[index];
        closestNum = 0;
        centroid_dist=distance( ptx, pty, s_centroidx[closestNum], s_centroidy[closestNum]);
        for (int i =  1; i < k; ++i)
        {
            new_centroid_dist=distance( ptx, pty, s_centroidx[i], s_centroidy[i]);
            if (  new_centroid_dist < centroid_dist )
            {
                centroid_dist=new_centroid_dist;
                closestNum = i;
            }
        }
    }



    for (int i = 0; i < k; ++ i)
    {
        __syncthreads();
        if (index<n && closestNum==i)
        {
            s_count[CONFLICT_FREE_INDEX(threadIdx.x)] = 1;
            s_sumx[CONFLICT_FREE_INDEX(threadIdx.x)] = ptx;
            s_sumy[CONFLICT_FREE_INDEX(threadIdx.x)] = pty;
        }
        else
        {
            s_count[CONFLICT_FREE_INDEX(threadIdx.x)] = 0;
            s_sumx[CONFLICT_FREE_INDEX(threadIdx.x)] = s_sumy[CONFLICT_FREE_INDEX(threadIdx.x)] = 0.0f;
        }

        for (int depth = 0; depth < LOG_BLOCKDIM; ++depth)
        {
            __syncthreads();
            if (threadIdx.x < 1<<(LOG_BLOCKDIM-depth-1) )
            {
                s_count[CONFLICT_FREE_INDEX( threadIdx.x<<(depth+1) )] += s_count[CONFLICT_FREE_INDEX( (threadIdx.x<<(depth+1)) + (1<<depth) )];
                s_sumx[CONFLICT_FREE_INDEX( threadIdx.x<<(depth+1) )] += s_sumx[CONFLICT_FREE_INDEX( (threadIdx.x<<(depth+1)) + (1<<depth) )];
                s_sumy[CONFLICT_FREE_INDEX( threadIdx.x<<(depth+1) )] += s_sumy[CONFLICT_FREE_INDEX( (threadIdx.x<<(depth+1)) + (1<<depth) )];
            }
        }

        __syncthreads();
        if (threadIdx.x==0)
        {
            d_count[ i * mem_length + blockIdx.x ] = s_count[0];
            d_sumx[ i * mem_length + blockIdx.x ] = s_sumx[0];
            d_sumy[ i * mem_length + blockIdx.x ] = s_sumy[0];
        }
    }
}

__global__ void
summation(int n,int k,int len, float *d_sumx, float *d_sumy, int *d_count, float *d_sumx_aux, float *d_sumy_aux, int *d_count_aux)
{
    //shared
    __shared__  float s_sumx[BLOCKDIM<<2];
    __shared__  float s_sumy[BLOCKDIM<<2];
    __shared__  int s_count[BLOCKDIM<<2];

    const unsigned int mem_length = get_mem_length(n);//(( (n-1) >> (LOG_BLOCKDIM + LOG_HALF_WARP)) + 1) << LOG_HALF_WARP;

    for (int i = 0; i < k; ++i)
    {
        unsigned int index = i * mem_length + (blockIdx.x<<(LOG_BLOCKDIM+1)) + threadIdx.x;
        __syncthreads();
        if ((blockIdx.x<<(LOG_BLOCKDIM+1)) + threadIdx.x<len)
        {
            s_sumx[CONFLICT_FREE_INDEX( threadIdx.x )] = d_sumx[index];
            s_sumy[CONFLICT_FREE_INDEX( threadIdx.x )] = d_sumy[index];
            s_count[CONFLICT_FREE_INDEX( threadIdx.x )] = d_count[index];
        }
        else
        {
            s_sumx[CONFLICT_FREE_INDEX( threadIdx.x )] = 0.0f;
            s_sumy[CONFLICT_FREE_INDEX( threadIdx.x )] = 0.0f;
            s_count[CONFLICT_FREE_INDEX( threadIdx.x )] = 0;
        }
        __syncthreads();
        if ((blockIdx.x<<(LOG_BLOCKDIM+1)) + threadIdx.x+BLOCKDIM < len)
        {
            s_sumx[CONFLICT_FREE_INDEX( threadIdx.x+BLOCKDIM )] = d_sumx[index+BLOCKDIM];
            s_sumy[CONFLICT_FREE_INDEX( threadIdx.x+BLOCKDIM )] = d_sumy[index+BLOCKDIM];
            s_count[CONFLICT_FREE_INDEX( threadIdx.x+BLOCKDIM )] = d_count[index+BLOCKDIM];
        }
        else
        {
            s_sumx[CONFLICT_FREE_INDEX( threadIdx.x+BLOCKDIM )] = 0.0f;
            s_sumy[CONFLICT_FREE_INDEX( threadIdx.x+BLOCKDIM )] = 0.0f;
            s_count[CONFLICT_FREE_INDEX( threadIdx.x+BLOCKDIM )] = 0;
        }

        //addition
        for (int depth = 0; depth < LOG_BLOCKDIM+1; ++depth)
        {
            __syncthreads();
            if (threadIdx.x < 1<<(LOG_BLOCKDIM-depth) )
            {
                s_count[CONFLICT_FREE_INDEX( threadIdx.x<<(depth+1) )] += s_count[CONFLICT_FREE_INDEX( (threadIdx.x<<(depth+1)) + (1<<depth) )];
                s_sumx[CONFLICT_FREE_INDEX( threadIdx.x<<(depth+1) )] += s_sumx[CONFLICT_FREE_INDEX( (threadIdx.x<<(depth+1)) + (1<<depth) )];
                s_sumy[CONFLICT_FREE_INDEX( threadIdx.x<<(depth+1) )] += s_sumy[CONFLICT_FREE_INDEX( (threadIdx.x<<(depth+1)) + (1<<depth) )];
            }
        }

        //return
        __syncthreads();
        if (threadIdx.x==0)
        {
            d_sumx_aux[ i * mem_length + blockIdx.x ] = s_sumx[0];
            d_sumy_aux[ i * mem_length + blockIdx.x ] = s_sumy[0];
            d_count_aux[ i * mem_length + blockIdx.x ] = s_count[0];
        }
    }

}

__global__ void
computecentroid(int n,int k, float threshold, float *d_centroidx, float *d_centroidy, float *d_sumx, float *d_sumy, int *d_count, bool* d_repeat)
{
    __shared__ int s_flag[BLOCKDIM<<1];

    const unsigned int mem_length = get_mem_length(n);//(( (n-1) >> (LOG_BLOCKDIM + LOG_HALF_WARP)) + 1) << LOG_HALF_WARP;
    const unsigned int index = (blockIdx.x << LOG_BLOCKDIM) + threadIdx.x;
    float sumx, sumy, centroidx,centroidy;
    int count;

    __syncthreads();
    if (index<k)
    {
        sumx = d_sumx[index*mem_length];
        sumy = d_sumy[index*mem_length];
        count=d_count[index*mem_length];
        centroidx = d_centroidx[index];
        centroidy = d_centroidy[index];
        if(count){
          sumx = __fdividef( sumx, __int2float_rn(count));
          sumy = __fdividef( sumy, __int2float_rn(count));
        }
        s_flag[CONFLICT_FREE_INDEX( threadIdx.x )] = (distance( sumx, sumy, centroidx, centroidy) < threshold * __int2float_rn(n)) ? 0 : 1;
        //printf("%d %f %d\n",threadIdx.x,d_sumx[threadIdx.x],d_count[threadIdx.x*mem_length]);
    }
    else
    {
        s_flag[CONFLICT_FREE_INDEX( threadIdx.x )] = 0.0f;
    }
    __syncthreads();
    if (threadIdx.x<k)
    {
#if DONT_CHNG_CENTROIDS == 0
        d_centroidx[threadIdx.x] = sumx;
        d_centroidy[threadIdx.x] = sumy;
#endif
    }

    for (int depth = 0; depth < LOG_BLOCKDIM; ++depth)
    {
        __syncthreads();
        if (threadIdx.x < 1<<(LOG_BLOCKDIM-depth-1) )
        {
            s_flag[CONFLICT_FREE_INDEX( threadIdx.x<<(depth+1) )]+=s_flag[CONFLICT_FREE_INDEX( (threadIdx.x<<(depth+1)) + (1<<depth) )];
        }
    }

    __syncthreads();
    if (threadIdx.x==0 && s_flag[0]>0)
    {
        *d_repeat=true;
    }
}

__global__ void
findCluster( int n, int k, const float *const d_dataptx, const float *const d_datapty, float *d_centroidx, float *d_centroidy, int *d_clusterno )
{
    // shared memory
    extern __shared__ float s_centroid[];

    unsigned int index = (blockIdx.x << LOG_BLOCKDIM) + threadIdx.x;
    int closestNum;
    float *const s_centroidx = s_centroid;
    float *const s_centroidy = s_centroid+k;
    float ptx,pty;

    __syncthreads();
    for (int i = 0; i < k; i+=BLOCKDIM)
        if ( i + threadIdx.x < k)
        {
            s_centroidx[i+threadIdx.x] = d_centroidx[i+threadIdx.x];
            s_centroidy[i+threadIdx.x] = d_centroidy[i+threadIdx.x];
        }

    __syncthreads();
    if (index<n)
    {
        ptx = d_dataptx[index];
        pty = d_datapty[index];
        closestNum = 0;
        for (int i =  1; i < k; ++i)
        {
            if ( distance( ptx, pty, s_centroidx[i], s_centroidy[i]) <
                    distance( ptx, pty, s_centroidx[closestNum], s_centroidy[closestNum]) )
            {
                closestNum = i;
            }
        }
    }

    __syncthreads();
    if (index<n)
        d_clusterno[index] = closestNum;
}


