#include <settings_xy.h>

////////////////////////////////////////////////////////////////////////////////
//!Number of blocks (MAX_BLOCKS defines the upper limit)
//! @param n number of points
////////////////////////////////////////////////////////////////////////////////
__device__ __host__ int get_num_blocks(int n)
{
    //Must be atleast equal to number of GPU cores.
    //Can be more with lesser threads per block if there are too many synchronize calls required so that there is minimum relative blocking among threads.
    return 1 + ( ( ( 1 + (n-1)/POINTS_PER_THREAD ) - 1 ) >> LOG_BLOCKDIM );

}



////////////////////////////////////////////////////////////////////////////////
//!Length of the array required for barrier synchronization of blocks
//! @param numBlocks number of blocks
////////////////////////////////////////////////////////////////////////////////
__device__ __host__ int get_barrier_synch_array_length(int numBlocks)
{
    int length=0;

    while(numBlocks > 1)
    {
        length += numBlocks;
        numBlocks = (numBlocks -1 ) >> LOG_BLOCKDIM +1;
    }

    return length;

}


////////////////////////////////////////////////////////////////////////////////
//!Number of blocks rounded to nearest multiple of HalfWarp i.e. ceil(num_blocks/HalfWarp)*HalfWarp
//! @param n number of points
////////////////////////////////////////////////////////////////////////////////
__device__ __host__ unsigned int get_mem_length(int n)
{
    int blocks = get_num_blocks(n);
    return ((((blocks-1)>>LOG_HALF_WARP) + 1) << LOG_HALF_WARP);                                        //MemLength gives number of blocks rounded to nearest multiple of HalfWarp i.e. ceil(num_blocks/HalfWarp)*HalfWarp
}


////////////////////////////////////////////////////////////////////////////////
//!Initialize all elements of an array with default values
////////////////////////////////////////////////////////////////////////////////
__global__ void initialize(int length, int *const RESTRICT array, int value)
{
    int index = (blockIdx.x << LOG_BLOCKDIM)+threadIdx.x;
    if(index < length)
        array[index]=value;

}

////////////////////////////////////////////////////////////////////////////////
//!Initialize all elements of an array with default values
////////////////////////////////////////////////////////////////////////////////
__global__ void initialize(int length, float *const RESTRICT array, float value)
{
    int index = (blockIdx.x << LOG_BLOCKDIM)+threadIdx.x;
    if(index < length)
        array[index]=value;

}


#if POINTS_PER_THREAD == 1    //load_data_points
__device__ int inline load_data_points(const unsigned int n,float * const RESTRICT ptx, float * const RESTRICT pty,
                                       const unsigned int index, const float *const RESTRICT d_dataptx,
                                       const float *const RESTRICT d_datapty)
{

  #if TRY_CACHE_BW_DEV_AND_SHARED == 1
    __syncthreads();
  #endif

    if (index < n )
    {
        *ptx=d_dataptx[index];
        *pty=d_datapty[index];
        return 1;
    }
    return 0;
}


__device__ int inline load_data_points(const unsigned int n,float * const RESTRICT ptx, float * const RESTRICT pty,
                                       const unsigned int index, const float *const RESTRICT d_dataptx,
                                       const float *const RESTRICT d_datapty, float *const RESTRICT s_thrd_x,
                                       float *const RESTRICT s_thrd_y)
{

  #if TRY_CACHE_BW_DEV_AND_SHARED == 1
    __syncthreads();
  #endif

    if (index < n )
    {
        *ptx=d_dataptx[index];
        *pty=d_datapty[index];

        s_thrd_x[threadIdx.x]=*ptx;
        s_thrd_y[threadIdx.x]=*pty;
        return 1;
    }
    return 0;
}
#else


__device__ int inline get_my_point_num(const int points_remaining)
{

    if(points_remaining <= 0)
        return 0;
    else if ((POINTS_PER_THREAD << LOG_BLOCKDIM) > points_remaining)               //This is the last block and not all threads might be processing POINTS_PER_THREAD points.
    {
        int my_point_num = points_remaining >> LOG_BLOCKDIM;                       //Next three lines take care of remainder if points_remaining is not divisible by BLOCKDIM
        if( (my_point_num << LOG_BLOCKDIM) + threadIdx.x < points_remaining)       //Load balance s.t. diff. b/w no. of pts processed by any two threads of this block is <= 1.
            my_point_num++;
        return my_point_num;
    }

    return POINTS_PER_THREAD;

}

__device__ void inline load_data_points_given_point_num(const unsigned int my_point_num, float ptx[POINTS_PER_THREAD],
                                                        float pty[POINTS_PER_THREAD], const unsigned int index,
                                                        const float *const RESTRICT d_dataptx, const float *const RESTRICT d_datapty)
{
#if TRY_CACHE_BW_DEV_AND_SHARED == 1
    __syncthreads();
#endif

    switch(my_point_num)
    {
    case POINTS_PER_THREAD:
        #pragma unroll 2
        for(int i=0; i<POINTS_PER_THREAD; i++)
        {
            ptx[i]=d_dataptx[index+(i<<LOG_BLOCKDIM)];
            pty[i]=d_datapty[index+(i<<LOG_BLOCKDIM)];
        }
        break;

    case 0:
        break;

    default:              //Only load as many points as are allotted to this thread.
        #pragma unroll 2
        for(int i=0; i<POINTS_PER_THREAD; i++)
        {
            if(i<my_point_num)
            {
              ptx[i]=d_dataptx[index+(i<<LOG_BLOCKDIM)];
              pty[i]=d_datapty[index+(i<<LOG_BLOCKDIM)];
            }
        }
    }

#if TRY_CACHE_BW_DEV_AND_SHARED == 1
    __syncthreads();
#endif
}


__device__ void inline load_data_points_given_point_num(const unsigned int my_point_num, float ptx[POINTS_PER_THREAD],
                                                        float pty[POINTS_PER_THREAD], const unsigned int index,
                                                        const float *const RESTRICT d_dataptx, const float *const RESTRICT d_datapty,
                                                        float *const RESTRICT s_thrd_x, float *const RESTRICT s_thrd_y)
{
#if TRY_CACHE_BW_DEV_AND_SHARED == 1
    __syncthreads();
#endif

    switch(my_point_num)
    {
    case POINTS_PER_THREAD:
        #pragma unroll 2
        for(int i=0; i<POINTS_PER_THREAD; i++)
        {
            ptx[i]=d_dataptx[index+(i<<LOG_BLOCKDIM)];
            pty[i]=d_datapty[index+(i<<LOG_BLOCKDIM)];

            s_thrd_x[threadIdx.x + (i<<LOG_BLOCKDIM)]=ptx[i];
            s_thrd_y[threadIdx.x + (i<<LOG_BLOCKDIM)]=pty[i];
        }
        break;

    case 0:
        break;

    default:              //Only load as many points as are allotted to this thread.
        #pragma unroll 2
        for(int i=0; i<POINTS_PER_THREAD; i++)
        {
            if(i<my_point_num)
            {
              ptx[i]=d_dataptx[index+(i<<LOG_BLOCKDIM)];
              pty[i]=d_datapty[index+(i<<LOG_BLOCKDIM)];

              s_thrd_x[threadIdx.x + (i<<LOG_BLOCKDIM)]=ptx[i];
              s_thrd_y[threadIdx.x + (i<<LOG_BLOCKDIM)]=pty[i];
            }
        }
    }

#if TRY_CACHE_BW_DEV_AND_SHARED == 1
    __syncthreads();
#endif
}


__device__ int inline load_data_points(const unsigned int n, float ptx[POINTS_PER_THREAD], float pty[POINTS_PER_THREAD], const unsigned int index,
                                       const float *const RESTRICT d_dataptx, const float *const RESTRICT d_datapty)
{
    const int points_remaining = n - (blockIdx.x << LOG_BLOCKDIM)*POINTS_PER_THREAD;                      //Total number of points to be processed by this and later blocks.

    int my_point_num=get_my_point_num(points_remaining);

    load_data_points_given_point_num(my_point_num, ptx, pty, index, d_dataptx, d_datapty);

    return my_point_num;
}


__device__ int inline load_data_points(const unsigned int n, float ptx[POINTS_PER_THREAD], float pty[POINTS_PER_THREAD], const unsigned int index,
                                       const float *const RESTRICT d_dataptx, const float *const RESTRICT d_datapty,
                                       float *const RESTRICT s_thrd_x, float *const RESTRICT s_thrd_y)
{
    const int points_remaining = n - (blockIdx.x << LOG_BLOCKDIM)*POINTS_PER_THREAD;                      //Total number of points to be processed by this and later blocks.

    int my_point_num=get_my_point_num(points_remaining);

    load_data_points_given_point_num(my_point_num, ptx, pty, index, d_dataptx, d_datapty, s_thrd_x, s_thrd_y);

    return my_point_num;
}


__device__ int inline load_data_points(const unsigned int n, const unsigned int curr_point_cycle, const unsigned int total_point_cycles,
                                       float ptx[POINTS_PER_THREAD], float pty[POINTS_PER_THREAD], const unsigned int index,
                                       const float *const RESTRICT d_dataptx, const float *const RESTRICT d_datapty)
{
    int my_point_num=POINTS_PER_THREAD;

    if(curr_point_cycle+1 == total_point_cycles)                                                            // This is the last point cycle
    {
        const int points_remaining = n - (gridDim.x*curr_point_cycle + blockIdx.x)*(POINTS_PER_THREAD << LOG_BLOCKDIM);               //Total number of points to be processed by this and later blocks.
        my_point_num=get_my_point_num(points_remaining);
    }

    load_data_points_given_point_num(my_point_num, ptx, pty, index, d_dataptx, d_datapty);

    return my_point_num;
}


__device__ int inline load_data_points(const unsigned int n, const unsigned int curr_point_cycle, const unsigned int total_point_cycles,
                                       float ptx[POINTS_PER_THREAD], float pty[POINTS_PER_THREAD], const unsigned int index,
                                       const float *const RESTRICT d_dataptx, const float *const RESTRICT d_datapty,
                                       float *const RESTRICT s_thrd_x, float *const RESTRICT s_thrd_y)
{
    int my_point_num=POINTS_PER_THREAD;

    if(curr_point_cycle+1 == total_point_cycles)                                                            // This is the last point cycle
    {
        const int points_remaining = n - (gridDim.x*curr_point_cycle + blockIdx.x)*(POINTS_PER_THREAD << LOG_BLOCKDIM);               //Total number of points to be processed by this and later blocks.
        my_point_num=get_my_point_num(points_remaining);
    }

    load_data_points_given_point_num(my_point_num, ptx, pty, index, d_dataptx, d_datapty, s_thrd_x, s_thrd_y);

    return my_point_num;
}

#endif


__device__ void inline load_centroids_in_shared(const unsigned int k, float *const RESTRICT s_centroidx, float *const RESTRICT s_centroidy,
        volatile const float *const RESTRICT d_centroidx, volatile const float *const RESTRICT d_centroidy)
{
    // Assign the values of the global memory centroids to the shared memory centroids for all k clusters
    // coalesced access
#if TRY_CACHE_BW_DEV_AND_SHARED == 1
    __syncthreads();
#endif
    // TODO: Make this assignment go at top even b4 data point loads as syncthreads is essential after this and
    // if points are loaded b4 this then it might cause us to wait for all data points to be loaded by all the threads in the block.
    // But by pushing this code to top, we can then load points after the syncthreads and then assign closest centroids b4 needing the next synthreads.
    // Thus, more parallelism might be available.
    for(int i=threadIdx.x; i<k; i+=BLOCKDIM)
    {
        s_centroidx[i] = d_centroidx[i];
        s_centroidy[i] = d_centroidy[i];
    }

}


#if POINTS_PER_THREAD == 1    //find_closest_centroid
__device__ signed int inline find_closest_centroid( const unsigned int n, const unsigned int k,const float ptx, const float pty, const unsigned int index,
        volatile const float *const RESTRICT s_centroidx, volatile const float *const RESTRICT s_centroidy)
{

    if (index < n)
    {
        int closestCentroid=0;

        // Find centroid nearest to the datapoint at location: index.
        float centroid_dist,new_centroid_dist;
        centroid_dist=distance( ptx, pty, s_centroidx[closestCentroid], s_centroidy[closestCentroid]);
        for (int i =  1; i < k; ++i)
        {
            // Broadcast access from shared mem. This line shud be executed by all threads in warp simultaneously.
            new_centroid_dist=distance( ptx, pty, s_centroidx[i], s_centroidy[i]);
            if (  new_centroid_dist < centroid_dist )
            {
                centroid_dist=new_centroid_dist;
                closestCentroid = i;
            }
        }
        return closestCentroid;
    }
    else
        return -1;                                                                           // Put an invalid k value.

}
#else
__device__ void inline find_closest_centroid(const unsigned int k,const float ptx[POINTS_PER_THREAD], const float pty[POINTS_PER_THREAD],
                                             volatile const float *const RESTRICT s_centroidx, volatile const float *const RESTRICT s_centroidy,
                                             const int my_point_num, int closestCentroid[POINTS_PER_THREAD])
{
    int startIndex=0;

    // Find centroid nearest to the datapoint at location: index.
    float centroid_dist[POINTS_PER_THREAD],new_centroid_dist[POINTS_PER_THREAD];

    float centroidx, centroidy;
    switch(my_point_num)
    {
    case POINTS_PER_THREAD:
        centroidx=s_centroidx[startIndex];
        centroidy=s_centroidy[startIndex];

        #pragma unroll 2
        for(int i=0; i< POINTS_PER_THREAD ; i++)
        {
            closestCentroid[i]=startIndex;
            centroid_dist[i]=distance( ptx[i], pty[i], centroidx, centroidy);
        }

        for (int centroid =  1; centroid < k; ++centroid)
        {

            //Broadcast access from shared mem. This line shud be executed by all threads in warp simultaneously.
            centroidx=s_centroidx[centroid];
            centroidy=s_centroidy[centroid];

            #pragma unroll 2
            for(int i=0; i<POINTS_PER_THREAD; i++)
            {
                new_centroid_dist[i]=distance( ptx[i], pty[i], centroidx, centroidy);

                if (  new_centroid_dist[i] < centroid_dist[i] )
                {
                    centroid_dist[i]=new_centroid_dist[i];
                    closestCentroid[i] = centroid;
                }
            }
        }
        break;

    case 0:
       #pragma unroll 2
        for(int i=0; i<POINTS_PER_THREAD; i++)
            closestCentroid[i] = -1;

        break;

    default:
        centroidx=s_centroidx[startIndex];
        centroidy=s_centroidy[startIndex];
       #pragma unroll 2
        for(int i=0; i<POINTS_PER_THREAD; i++)
        {
            if(i<my_point_num)
            {
              closestCentroid[i]=startIndex;
              centroid_dist[i]=distance( ptx[i], pty[i], centroidx, centroidy);
            }else{
              closestCentroid[i] = -1;
            }
        }

        for (int centroid =  1; centroid < k; ++centroid)
        {
            //Broadcast access from shared mem. This line shud be executed by all threads in warp simultaneously.
            centroidx=s_centroidx[centroid];
            centroidy=s_centroidy[centroid];

            #pragma unroll 2
            for(int i=0; i<POINTS_PER_THREAD; i++)
            {
                if(i<my_point_num)
                {
                    new_centroid_dist[i]=distance( ptx[i], pty[i], centroidx, centroidy);
                    if (  new_centroid_dist[i] < centroid_dist[i] )
                    {
                        centroid_dist[i]=new_centroid_dist[i];
                        closestCentroid[i] = centroid;
                    }
                }
            }
        }
    }
}
#endif


#if POINTS_PER_THREAD == 1    //store_nearest_in_register
__device__ int inline store_nearest_in_register(const unsigned int n, const unsigned int k,const float ptx, const float pty,
        const unsigned int index, VOLATILE_STORE const float *const RESTRICT s_centroidx, VOLATILE_STORE const float *const RESTRICT s_centroidy)
{
    // Store the index of the centroid closest to the datapoint.
      return find_closest_centroid( n, k, ptx, pty, index, s_centroidx, s_centroidy);
}

#else

__device__ void inline store_nearest_in_register(const unsigned int k,const float ptx[POINTS_PER_THREAD], const float pty[POINTS_PER_THREAD],
        VOLATILE_STORE const float *const RESTRICT s_centroidx, VOLATILE_STORE const float *const RESTRICT s_centroidy,
        int closest[POINTS_PER_THREAD], const int my_point_num)
{
    // Find and store closest centroid if this thread holds a data-point, else store -1
    switch(my_point_num)
    {
    case 0:
        #pragma unroll 2
        for(int i=0; i<POINTS_PER_THREAD; i++)
            closest[i]=-1;       // Put an invalid k value.

        break;
    default:
        find_closest_centroid( k, ptx, pty, s_centroidx, s_centroidy, my_point_num, closest);
    }
}
#endif

#if POINTS_PER_THREAD == 1    //store_nearest_in_shared
__device__ void inline store_nearest_in_shared(const unsigned int n, const unsigned int k,const float ptx, const float pty,
        const unsigned int index, VOLATILE_STORE const float *const RESTRICT s_centroidx, VOLATILE_STORE const float *const RESTRICT s_centroidy,
        int *const RESTRICT s_thrd_k)
{
    // Store the index of the centroid closest to the datapoint.
      s_thrd_k[threadIdx.x]=find_closest_centroid( n, k, ptx, pty, index, s_centroidx, s_centroidy);
}

#else

__device__ void inline store_nearest_in_shared(const unsigned int k,const float ptx[POINTS_PER_THREAD], const float pty[POINTS_PER_THREAD],
        VOLATILE_STORE const float *const RESTRICT s_centroidx, VOLATILE_STORE const float *const RESTRICT s_centroidy,
        int *const RESTRICT s_thrd_k, const int my_point_num)
{
    // Find and store closest centroid if this thread holds a data-point, else store -1
    switch(my_point_num)
    {
    case 0:
        #pragma unroll 2
        for(int i=0; i<POINTS_PER_THREAD; i++)
            s_thrd_k[threadIdx.x + (i<<LOG_BLOCKDIM)]=-1;       // Put an invalid k value.

        break;
    default:
        int closestCentroid[POINTS_PER_THREAD];
        find_closest_centroid( k, ptx, pty, s_centroidx, s_centroidy, my_point_num, closestCentroid);

        // Store closestNum as the index of the closest centroid to the datapoint.
        #pragma unroll 2
        for(int i=0; i<POINTS_PER_THREAD; i++)
            s_thrd_k[threadIdx.x + (i<<LOG_BLOCKDIM)]=closestCentroid[i];

    }
}
#endif


#if POINTS_PER_THREAD == 1    //store_nearest_in_global
__device__ void inline store_nearest_in_global(const unsigned int n, const unsigned int k,const float ptx,
        const float pty, const unsigned int index, VOLATILE_STORE const float *const RESTRICT s_centroidx,
        VOLATILE_STORE const float *const RESTRICT s_centroidy, int *const RESTRICT d_clusterno)
{
    // Find and store closest centroid if this thread holds a data-point, else store -1

  #if TRY_CACHE_BW_DEV_AND_SHARED == 1
    __syncthreads();
  #endif

      if(index<n)
      // Store the index of the centroid closest to the datapoint.
        d_clusterno[index]=find_closest_centroid( n, k, ptx, pty, index, s_centroidx, s_centroidy);

}
#else


__device__ void inline store_nearest_in_global(const unsigned int k, const float ptx[POINTS_PER_THREAD], const float pty[POINTS_PER_THREAD],
        const unsigned int index, VOLATILE_STORE const float *const RESTRICT s_centroidx, VOLATILE_STORE const float *const RESTRICT s_centroidy,
        int *const RESTRICT d_clusterno, const int my_point_num)
{
    // Find and store closest centroid if this thread holds a data-point, else store -1
    int closestCentroid[POINTS_PER_THREAD];

    if(my_point_num > 0)
        find_closest_centroid(k, ptx, pty, s_centroidx, s_centroidy, my_point_num, closestCentroid);

#if TRY_CACHE_BW_DEV_AND_SHARED == 1
    __syncthreads();
#endif

    switch(my_point_num)
    {
    case POINTS_PER_THREAD:
        // Store closestNum as the index of the closest centroid to the datapoint.
        #pragma unroll 2
        for(int i=0; i<POINTS_PER_THREAD; i++)
            d_clusterno[index + (i<<LOG_BLOCKDIM)]=closestCentroid[i];
        break;
    case 0:
        break;
    default:
        // Store closestNum as the index of the closest centroid to the datapoint.
        #pragma unroll 2
        for(int i=0; i<POINTS_PER_THREAD; i++){
            if(i<my_point_num)
              d_clusterno[index + (i<<LOG_BLOCKDIM)]=closestCentroid[i];
        }
    }

}
#endif


__device__ int inline reduceThreads_single(const unsigned int k, float *const RESTRICT sumx, float *const RESTRICT sumy,
                                    volatile const float *const RESTRICT s_thrd_x, volatile const float *const RESTRICT s_thrd_y,
                                    volatile const int *const RESTRICT s_thrd_k)
{
    int count=0;
    if (threadIdx.x < k)
    {
#if TRY_ATOMIC_CNTR_IN_LOCAL_REDCN == 1
        __shared__ unsigned int points_checked;
        if(threadIdx.x==0)
          points_checked=0;
        __syncthreads();
//TODO: Unroll the following for loop, if unrolling does not cause memory spilling.
        for (int i = 0; points_checked < BLOCKDIM*POINTS_PER_THREAD && i < BLOCKDIM*POINTS_PER_THREAD; ++i)
        {
            int indx=i+threadIdx.x;
            if (indx >= BLOCKDIM*POINTS_PER_THREAD)
                indx -= BLOCKDIM*POINTS_PER_THREAD;
            if (threadIdx.x == s_thrd_k[indx])                                                          // Coalesced access from shared mem.
            {
                // Only one thread from all the warps wud enter in, all other warps wud move to the next value of i.
                // In all WarpSize -1 threads wait here every time one of the threads meets the if-condition for some i.
                // Max. no. of warps that can be simultaneously in this if-cond while executing for diff. values of i is: 2*(broad. shared access) + 3*floating pt. arithmetic instr

                *sumx = (*sumx) + s_thrd_x[indx];
                *sumy = (*sumy) + s_thrd_y[indx];
                count++;
                atomicInc(&points_checked,BLOCKDIM*POINTS_PER_THREAD);
            }
        }
#else
//TODO: Unroll the following for loop, if unrolling does not cause memory spilling.
        for (int i = 0; i < BLOCKDIM*POINTS_PER_THREAD; ++i)
        {
        // Broadcast access from shared mem. This line shud be executed by all threads in warp simultaneously.
            if (threadIdx.x == s_thrd_k[i])
            {
                *sumx = (*sumx) + s_thrd_x[i];
                *sumy = (*sumy) + s_thrd_y[i];
                count++;
            }
        }

#endif

    }

    return count;
}


#if POINTS_PER_THREAD == 1    //reduceValues_multi
__device__ int inline reduceValues_multi(const unsigned int k, float *const RESTRICT sumx, float *const RESTRICT sumy,
                                    volatile const float *const RESTRICT s_thrd_x, volatile const float *const RESTRICT s_thrd_y,
                                    volatile const int *const RESTRICT s_thrd_k)
{

    int count=0;

    int limit=k;
    while(limit <= BLOCKDIM){
      limit=(limit<<1);
    }
    limit=(limit>>1);

    if (threadIdx.x < limit)
    {
        int checkFor=threadIdx.x%k;
        int index=threadIdx.x;
        int pointsToCheck=k;

        if(BLOCKDIM < ((index >> 5)<<5) + FULL_WARP-1 + pointsToCheck)
        {
          for (int i = 0; i < pointsToCheck; ++i)
          {
              if(index >= BLOCKDIM)
              {
                index=index - BLOCKDIM;
              }
          // Coalesced access from shared mem.
              if (checkFor == s_thrd_k[index])
              {
                  *sumx = (*sumx) + s_thrd_x[index];
                  *sumy = (*sumy) + s_thrd_y[index];
                  count++;
              }
              index++;
          }
        }else
        {
          for (int i = 0; i < pointsToCheck; ++i)
          {
          // Coalesced access from shared mem.
              if (checkFor == s_thrd_k[index])
              {
                  *sumx = (*sumx) + s_thrd_x[index];
                  *sumy = (*sumy) + s_thrd_y[index];
                  count++;
              }
              index++;
          }
        }


        int points_remaining=BLOCKDIM-limit;
        int threadsPerCluster=points_remaining/k;

        index=limit+threadIdx.x;
        if(threadIdx.x/k==threadsPerCluster)
          pointsToCheck=points_remaining-threadsPerCluster*k;
        else
        {
          if(threadIdx.x/k > threadsPerCluster)
            pointsToCheck=0;
        }

        if(BLOCKDIM < ((index >> 5)<<5) + FULL_WARP-1 + pointsToCheck)
        {
          for (int i = 0; i < pointsToCheck; ++i)
          {
              if(index >= BLOCKDIM)
              {
                index=index - BLOCKDIM;
              }
          // Coalesced access from shared mem.
              if (checkFor == s_thrd_k[index])
              {
                  *sumx = (*sumx) + s_thrd_x[index];
                  *sumy = (*sumy) + s_thrd_y[index];
                  count++;
              }
              index++;
          }
        }else
        {
          for (int i = 0; i < pointsToCheck; ++i)
          {
          // Coalesced access from shared mem.
              if (checkFor == s_thrd_k[index])
              {
                  *sumx = (*sumx) + s_thrd_x[index];
                  *sumy = (*sumy) + s_thrd_y[index];
                  count++;
              }
              index++;
          }
        }

    }

    return count;
}
#else
__device__ int inline reduceValues_multi(const unsigned int k, float *const RESTRICT sumx, float *const RESTRICT sumy,
                                    volatile const float *const RESTRICT s_thrd_x, volatile const float *const RESTRICT s_thrd_y,
                                    volatile const int *const RESTRICT s_thrd_k)
{

    int count=0;

    int limit=k;
    while(limit <= BLOCKDIM){
      limit=(limit<<1);
    }
    limit=(limit>>1);

    if (threadIdx.x < limit)
    {
        int checkFor=threadIdx.x%k;
        int index=threadIdx.x;
        int pointsToCheck=k;
        int currCycle;

        for( currCycle = 0; currCycle<BLOCKDIM*POINTS_PER_THREAD/limit; currCycle++){
            index=currCycle*limit + threadIdx.x;
            if(BLOCKDIM*POINTS_PER_THREAD < ((index >> 5)<<5) + FULL_WARP-1 + pointsToCheck)
            {
              for (int i = 0; i < pointsToCheck; ++i)
              {
                  if(index >= BLOCKDIM*POINTS_PER_THREAD)
                  {
                    index=index - BLOCKDIM*POINTS_PER_THREAD;
                  }
              // Coalesced access from shared mem.
                  if (checkFor == s_thrd_k[index])
                  {
                      *sumx = (*sumx) + s_thrd_x[index];
                      *sumy = (*sumy) + s_thrd_y[index];
                      count++;
                  }
                  index++;
              }
            }else
            {
              for (int i = 0; i < pointsToCheck; ++i)
              {
              // Coalesced access from shared mem.
                  if (checkFor == s_thrd_k[index])
                  {
                      *sumx = (*sumx) + s_thrd_x[index];
                      *sumy = (*sumy) + s_thrd_y[index];
                      count++;
                  }
                  index++;
              }
            }
        }

        int points_remaining=BLOCKDIM*POINTS_PER_THREAD-limit*currCycle;
        int threadsPerCluster=points_remaining/k;

        index=currCycle*limit+threadIdx.x;
        if(threadIdx.x/k==threadsPerCluster)
          pointsToCheck=points_remaining-threadsPerCluster*k;
        else
        {
          if(threadIdx.x/k > threadsPerCluster)
            pointsToCheck=0;
        }

        if(BLOCKDIM*POINTS_PER_THREAD < ((index >> 5)<<5) + FULL_WARP-1 + pointsToCheck)
        {
          for (int i = 0; i < pointsToCheck; ++i)
          {
              if(index >= BLOCKDIM*POINTS_PER_THREAD)
              {
                index   = index - BLOCKDIM*POINTS_PER_THREAD;
              }
          // Coalesced access from shared mem.
              if (checkFor == s_thrd_k[index])
              {
                  *sumx = (*sumx) + s_thrd_x[index];
                  *sumy = (*sumy) + s_thrd_y[index];
                  count++;
              }
              index++;
          }
        }else
        {
          for (int i = 0; i < pointsToCheck; ++i)
          {
          // Coalesced access from shared mem.
              if (checkFor == s_thrd_k[index])
              {
                  *sumx = (*sumx) + s_thrd_x[index];
                  *sumy = (*sumy) + s_thrd_y[index];
                  count++;
              }
              index++;
          }
        }

    }

    return count;
}
#endif


__device__ void inline reduceThreads_multi(const unsigned int k, float *const RESTRICT sumx, float *const RESTRICT sumy,
                                    int *const RESTRICT count,volatile float *const RESTRICT s_thrd_x,
                                    volatile float *const RESTRICT s_thrd_y, volatile int *const RESTRICT s_thrd_k)
{

    int limit=k;
    while(limit <= BLOCKDIM){
      limit=(limit<<1);
    }
    limit=(limit>>1);

    const int tid=threadIdx.x;
    if(tid<limit)
    {
      s_thrd_x[tid]=*sumx;
      s_thrd_y[tid]=*sumy;
      s_thrd_k[tid]=*count;
    }
    __syncthreads();


    int stride;
    for(stride = (limit>>1); stride>=k && stride>FULL_WARP; stride=(stride>>1))
    {
       if (tid < stride)
       {
         s_thrd_x[tid] = *sumx = *sumx + s_thrd_x[tid + stride];
         s_thrd_y[tid] = *sumy = *sumy + s_thrd_y[tid + stride];
         s_thrd_k[tid] = *count = *count + s_thrd_k[tid + stride];
       }
       __syncthreads();
    }

    if(tid<FULL_WARP) // We can skip this part and while doing global reduction do it for all 32 threads instead of just k threads as anyway memory transfers happening wud be same
                      // Most probably it wont make much difference as this reduction is invoked only once per iteration
    {
        for(; stride>=k ; stride=(stride>>1))
        {
             s_thrd_x[tid] = *sumx = *sumx + s_thrd_x[tid + stride];
             s_thrd_y[tid] = *sumy = *sumy + s_thrd_y[tid + stride];
             s_thrd_k[tid] = *count = *count + s_thrd_k[tid + stride];
        }
    }

}


#define REDUCE_EXP(tid, stride) if (blockSize >= (stride<<1)) { if (tid < stride) { s_thrd_x[tid] = *sumx = *sumx + s_thrd_x[tid + stride];\
                                                                                    s_thrd_y[tid] = *sumy = *sumy + s_thrd_y[tid + stride];\
                                                                                    s_thrd_k[tid] = *count = *count + s_thrd_k[tid + stride];\
                                                                                  } __syncthreads();\
                                                              }

#define REDUCE_EXP_NOSYNC(tid, stride) if (blockSize >= (stride<<1)){   s_thrd_x[tid] = *sumx = *sumx + s_thrd_x[tid + stride];\
                                                                        s_thrd_y[tid] = *sumy = *sumy + s_thrd_y[tid + stride];\
                                                                        s_thrd_k[tid] = *count = *count + s_thrd_k[tid + stride];\
                                                                    }

template <unsigned int blockSize>
__device__ inline void
reduce(float *const RESTRICT sumx, float *const RESTRICT sumy, int *const RESTRICT count,
       volatile float *const RESTRICT s_thrd_x, volatile float *const RESTRICT s_thrd_y,
       volatile int *const RESTRICT s_thrd_k)
{
    __syncthreads();

    // do reduction in shared mem
    REDUCE_EXP(threadIdx.x, 512)
    REDUCE_EXP(threadIdx.x, 256)
    REDUCE_EXP(threadIdx.x, 128)
    REDUCE_EXP(threadIdx.x, 64)

    if (threadIdx.x < 32)
    {
        REDUCE_EXP_NOSYNC(threadIdx.x, 32)
        REDUCE_EXP_NOSYNC(threadIdx.x, 16)
        REDUCE_EXP_NOSYNC(threadIdx.x,  8)
        REDUCE_EXP_NOSYNC(threadIdx.x,  4)
        REDUCE_EXP_NOSYNC(threadIdx.x,  2)
        REDUCE_EXP_NOSYNC(threadIdx.x,  1)
    }

}


#if POINTS_PER_THREAD == 1    //reduceThreads_tree
__device__ int inline reduceThreads_tree(const unsigned int k, float *const RESTRICT sumx, float *const RESTRICT sumy,
                                    volatile float *const RESTRICT s_thrd_x, volatile float *const RESTRICT s_thrd_y,
                                    volatile int *const RESTRICT s_thrd_k, const float ptx, const float pty, const int closest)
{
    int count=0;

    int tmp_count;
    float tmp_sumx,tmp_sumy;
    for (int i = 0; i < k; ++ i)
    {
        tmp_count=0;
        tmp_sumx=tmp_sumy=0.0f;
        if (closest==i)
        {
            s_thrd_k[threadIdx.x] = tmp_count = 1;
            s_thrd_x[threadIdx.x] = tmp_sumx = ptx;
            s_thrd_y[threadIdx.x] = tmp_sumy = pty;
        }
        else
        {
            s_thrd_k[threadIdx.x] = tmp_count = 0;
            s_thrd_x[threadIdx.x] = s_thrd_y[threadIdx.x] = tmp_sumx = tmp_sumy = 0.0f;
        }
        switch (BLOCKDIM)
        {
          case 1024:
              reduce<1024>(&tmp_sumx, &tmp_sumy, &tmp_count, s_thrd_x, s_thrd_y, s_thrd_k); break;
          case 512:
              reduce<512>(&tmp_sumx, &tmp_sumy, &tmp_count, s_thrd_x, s_thrd_y, s_thrd_k); break;
          case 256:
              reduce<256>(&tmp_sumx, &tmp_sumy, &tmp_count, s_thrd_x, s_thrd_y, s_thrd_k); break;
          case 128:
              reduce<128>(&tmp_sumx, &tmp_sumy, &tmp_count, s_thrd_x, s_thrd_y, s_thrd_k); break;
          case 64:
              reduce<64>(&tmp_sumx, &tmp_sumy, &tmp_count, s_thrd_x, s_thrd_y, s_thrd_k); break;
          case 32:
              reduce<32>(&tmp_sumx, &tmp_sumy, &tmp_count, s_thrd_x, s_thrd_y, s_thrd_k); break;
          case 16:
              reduce<16>(&tmp_sumx, &tmp_sumy, &tmp_count, s_thrd_x, s_thrd_y, s_thrd_k); break;
          case  8:
              reduce<8>(&tmp_sumx, &tmp_sumy, &tmp_count, s_thrd_x, s_thrd_y, s_thrd_k); break;
          case  4:
              reduce<4>(&tmp_sumx, &tmp_sumy, &tmp_count, s_thrd_x, s_thrd_y, s_thrd_k); break;
          case  2:
              reduce<2>(&tmp_sumx, &tmp_sumy, &tmp_count, s_thrd_x, s_thrd_y, s_thrd_k); break;
          case  1:
              reduce<1>(&tmp_sumx, &tmp_sumy, &tmp_count, s_thrd_x, s_thrd_y, s_thrd_k); break;
        }

        __syncthreads();

        if (threadIdx.x==i)
        {
            *sumx += s_thrd_x[0];
            *sumy += s_thrd_y[0];
            count = s_thrd_k[0];
        }

        __syncthreads();
    }


    return count;
}
#else
__device__ int inline reduceThreads_tree(const unsigned int k, float *const RESTRICT sumx, float *const RESTRICT sumy,
                                    volatile float *const RESTRICT s_thrd_x, volatile float *const RESTRICT s_thrd_y,
                                    volatile int *const RESTRICT s_thrd_k, const float ptx[POINTS_PER_THREAD],
                                    const float pty[POINTS_PER_THREAD], const int closest[POINTS_PER_THREAD])
{
    int count=0;

    int tmp_count;
    float tmp_sumx,tmp_sumy;
    for (int i = 0; i < k; ++ i)
    {
        tmp_count=0;
        tmp_sumx=tmp_sumy=0.0f;

        #pragma unroll 2
        for(int j=0;j<POINTS_PER_THREAD; j++)
        {
            if (closest[j]==i)
          {
              tmp_count++;
              tmp_sumx+= ptx[j];
              tmp_sumy+= pty[j];
          }
        }

        s_thrd_k[threadIdx.x] = tmp_count;
        s_thrd_x[threadIdx.x] = tmp_sumx;
        s_thrd_y[threadIdx.x] = tmp_sumy;

        switch (BLOCKDIM)
        {
          case 1024:
              reduce<1024>(&tmp_sumx, &tmp_sumy, &tmp_count, s_thrd_x, s_thrd_y, s_thrd_k); break;
          case 512:
              reduce<512>(&tmp_sumx, &tmp_sumy, &tmp_count, s_thrd_x, s_thrd_y, s_thrd_k); break;
          case 256:
              reduce<256>(&tmp_sumx, &tmp_sumy, &tmp_count, s_thrd_x, s_thrd_y, s_thrd_k); break;
          case 128:
              reduce<128>(&tmp_sumx, &tmp_sumy, &tmp_count, s_thrd_x, s_thrd_y, s_thrd_k); break;
          case 64:
              reduce<64>(&tmp_sumx, &tmp_sumy, &tmp_count, s_thrd_x, s_thrd_y, s_thrd_k); break;
          case 32:
              reduce<32>(&tmp_sumx, &tmp_sumy, &tmp_count, s_thrd_x, s_thrd_y, s_thrd_k); break;
          case 16:
              reduce<16>(&tmp_sumx, &tmp_sumy, &tmp_count, s_thrd_x, s_thrd_y, s_thrd_k); break;
          case  8:
              reduce<8>(&tmp_sumx, &tmp_sumy, &tmp_count, s_thrd_x, s_thrd_y, s_thrd_k); break;
          case  4:
              reduce<4>(&tmp_sumx, &tmp_sumy, &tmp_count, s_thrd_x, s_thrd_y, s_thrd_k); break;
          case  2:
              reduce<2>(&tmp_sumx, &tmp_sumy, &tmp_count, s_thrd_x, s_thrd_y, s_thrd_k); break;
          case  1:
              reduce<1>(&tmp_sumx, &tmp_sumy, &tmp_count, s_thrd_x, s_thrd_y, s_thrd_k); break;
        }

        __syncthreads();

        if (threadIdx.x==i)
        {
            *sumx += s_thrd_x[0];
            *sumy += s_thrd_y[0];
            count = s_thrd_k[0];
        }

        __syncthreads();
    }


    return count;
}
#endif


__device__ void inline reduceBlocks(const int k,const int goalVal,unsigned int blocks,const unsigned int storageIndex, float *const RESTRICT sumx,
                                    float *const RESTRICT sumy,int *const RESTRICT count,volatile float *const RESTRICT d_sumx,
                                    volatile float *const RESTRICT d_sumy, volatile int *const RESTRICT d_count,unsigned int reductionDist,
                                    volatile int *const RESTRICT Arrayin)//, int *d_reducedCounts, clock_t *const d_timeVar)
{
    // Perform reduction till you have no one left on your right to reduce with.
    // At that stage store your own reduced values in global mem. and return.
    // Only blockIdx 0, wont store its reduced values as it will directly use them to find centroids.
    // Hence, we wait till blockIdx 0 is the only block left in this loop.
    while(blocks > 1)
    {

        if((blockIdx.x & ((reductionDist<<1) -1)) == 0)//% (reductionDist<<1) == 0 )                                 // (x % 2^k == 0) is same as (x & ((2^k)-1) == 0)
        {
            unsigned int reduceWithBlockIdx=blockIdx.x+reductionDist;
            if(reduceWithBlockIdx < gridDim.x)
            {
                if(threadIdx.x==0)
                {
                    while(Arrayin[reduceWithBlockIdx]!=goalVal);
                }
                __syncthreads();                                                                                     // All threads wait for the block with which we have to reduce to finish its own global reduction

                if(threadIdx.x < k)                                                                                  // Perform reduction for each cluster
                {
                    const unsigned int reduceWithStorageIndex = (reduceWithBlockIdx>>1)*k + threadIdx.x;

                    *sumx=(*sumx)+d_sumx[reduceWithStorageIndex];
                    *sumy=(*sumy)+d_sumy[reduceWithStorageIndex];
                    *count=*count+d_count[reduceWithStorageIndex];

                }
#if TRY_CACHE_BW_DEV_AND_SHARED == 1
                __syncthreads();
#endif
                blocks=blocks - (blocks>>1);
                reductionDist=reductionDist<<1;
                continue;
            }
        }

        {
            // I didnt reduce with anyone. So its time I store my own value and wait for first block to finish which will
            // indicate that whole reduction is finished
#if TRY_CACHE_BW_DEV_AND_SHARED == 1
            __syncthreads();
#endif
            if(threadIdx.x < k)                                                                                   // Store the reduced values in the global mem.
            {
                d_sumx[storageIndex]=*sumx;
                d_sumy[storageIndex]=*sumy;
                d_count[storageIndex]=*count;
            }

            __threadfence();                                                                             // threadfence has to be b4 syncthreads so that all threads have made public their changes.
            __syncthreads();                                                                             // Ensure sum values are seen by other blocks b4 Arrayin is seen by them.

            if(threadIdx.x==0)
            {
                Arrayin[blockIdx.x]=goalVal;
            }
            break;

        }
    }
}


__device__ void inline reduceBlocks_and_storeNewCentroids(const int k,const int numIter,unsigned int blocks,const unsigned int storageIndex,
        float *const RESTRICT sumx,float *const RESTRICT sumy,int *const RESTRICT count,
        float *const RESTRICT d_sumx, float *const RESTRICT d_sumy, int *const RESTRICT d_count,
        volatile int *const RESTRICT d_syncArr, float *const RESTRICT d_centroidx,
        float *const RESTRICT d_centroidy)
{
    unsigned int reducnDist=1;
    reduceBlocks(k,numIter,gridDim.x,storageIndex,sumx,sumy,count,d_sumx,d_sumy,d_count,reducnDist,d_syncArr);  // Local reduction done for all centroids.

    //Global reduction done for all blocks. Now store the new centroid values in global memory
    if(blockIdx.x==0)
    {
        if(threadIdx.x < k)
        {
#if APPROACH == 155
            d_sumx[storageIndex]=*sumx;
            d_sumy[storageIndex]=*sumy;
            d_count[storageIndex]=*count;
#endif
            if(*count != 0)
            {
#if DONT_CHNG_CENTROIDS == 0
                d_centroidx[threadIdx.x] = __fdividef( *sumx, __int2float_rn(*count));
                d_centroidy[threadIdx.x] = __fdividef( *sumy, __int2float_rn(*count));
#endif
            }
        }

        __threadfence();                                                                                      //Ensure sum values are seen by other blocks b4 Arrayin is seen by them.
        __syncthreads();

        // Declare reduction done by thread with greatest index so that if no. of centroids is less than threads,
        // then by the time last thread sets the global value other threads from other warps would load the new
        // centroid values into the shared memory.
        if(threadIdx.x == BLOCKDIM -1)
        {
            d_syncArr[0]=numIter;                                                                                   // Declares global reduction over when it has stored new centroids in global mem.
        }
        //No need for syncthreads() as other threads can go on and load the new centroid values into shared mem. from device mem.
    }
    else
    {
        if(threadIdx.x==0)
        {
            while(d_syncArr[0]!=numIter);                                                                           // All blocks wait for blockIdx 0, to store new centroids.
        }
        __syncthreads();                                                                                          // One threads checks value from global and others just wait for it.
    }
}


// Right now this code only works when
// 1. Number of threads >= k
// 2. Number of points can be > threads*blocks, i.e each thread may have to process more than one point till the shared mem allows.
// 3. All the centroids can be accomodated in shared mem. at once.
__global__ void __launch_bounds__(BLOCKDIM, MIN_NUM_BLOCKS_PER_MULTIPROC)
cluster_single(int n, int k,int *const RESTRICT max_iter, const float *const RESTRICT d_dataptx, const float *const RESTRICT d_datapty,
        float *const RESTRICT d_centroidx, float *const RESTRICT d_centroidy, int *const RESTRICT d_syncArr,
        float *const RESTRICT d_sumx, float *const RESTRICT d_sumy, int *const RESTRICT d_count,int *const RESTRICT d_clusterno)//, int *const d_reducedCounts, clock_t *const d_timeVar)//, int *const d_dbgIter)
{
    // shared memory
    // Stores x-coord of all clusters followed by y-coord of all clusters and so on.
    extern __shared__ float s_centroid[];

    // No point of going for short int as each bank in shared mem. is 32-bit wide and only one thread
    // can access one bank at a time, so no. of accesses wont be reduced
    __shared__ signed int s_thrd_k[(BLOCKDIM)*POINTS_PER_THREAD];                                           // Store the nearest cluster of point checked by this thrd.

    __shared__ float s_thrd[(BLOCKDIM<<1)*POINTS_PER_THREAD];                                               // Store the x-coord of data point processed, for all thrds followed by y-coord.

    float *const s_centroidx = s_centroid;                                                                  // x-coord of 1st cluster's centroid
    float *const s_centroidy = s_centroid+k;                                                                // y-coord of 1st cluster's centroid

    float *const s_thrd_x = s_thrd;                                                                         // x-coord of points checked by thrds
    float *const s_thrd_y = s_thrd+(POINTS_PER_THREAD<<LOG_BLOCKDIM);                                       // y-coord of points checked by thrds


  #if POINTS_PER_THREAD == 1
      float ptx,pty;
  #else
      float ptx[POINTS_PER_THREAD],pty[POINTS_PER_THREAD];
  #endif

    // If we've T threads per block then index of ith thread in kth block is k*T + i. So we are maintaining an order among all threads of all blocks.
    // This is same as blockIdx.x * BLOCKDIM + threadId.x
    const unsigned int index = (POINTS_PER_THREAD << LOG_BLOCKDIM)*blockIdx.x + threadIdx.x;

  #if POINTS_PER_THREAD == 1
    load_data_points(n,&ptx, &pty, index, d_dataptx, d_datapty, s_thrd_x, s_thrd_y);                            //Number of points to be processed by this thread.
  #else
    const int my_point_num = load_data_points(n,ptx, pty, index, d_dataptx, d_datapty, s_thrd_x, s_thrd_y);     //Number of points to be processed by this thread.
  #endif

    bool repeat=true;
    int numIter;
    for ( numIter = 0; repeat && (numIter < (*max_iter)); ++numIter )
    {
        load_centroids_in_shared(k, s_centroidx, s_centroidy, d_centroidx, d_centroidy);
        __syncthreads();


        // Find closest centroid if this thread holds a data-point and set it in shared mem.
  #if POINTS_PER_THREAD == 1
          store_nearest_in_shared(n, k, ptx, pty, index, s_centroidx, s_centroidy, s_thrd_k);
  #else
          store_nearest_in_shared(k, ptx, pty, s_centroidx, s_centroidy, s_thrd_k, my_point_num);
  #endif
        __syncthreads();                                                                                    // Wait for all thrds to finish setting closest k value


        // Do parallel reduction with each thread doing reduction for a distinct centroid.
        const unsigned int storageIndex = (blockIdx.x>>1)*k + threadIdx.x;                                        // The sumx, sumy and count value will be stored at this index in global mem.
        int count=0;
        float sumx=0;
        float sumy =0;

        count += reduceThreads_single(k, &sumx, &sumy, s_thrd_x, s_thrd_y, s_thrd_k);

        reduceBlocks_and_storeNewCentroids(k,numIter,gridDim.x,storageIndex,&sumx,&sumy,&count,d_sumx,d_sumy,d_count,d_syncArr, d_centroidx, d_centroidy);//, d_reducedCounts, d_timeVar);//,d_dbgIter,*max_iter);
    }

    // Final centroids have been found. Now set the cluster no. in d_clusterno array in device memory.
    {
        load_centroids_in_shared(k, s_centroidx, s_centroidy, d_centroidx, d_centroidy);
        __syncthreads();

        // Find closest centroid if this thread holds a data-point
  #if POINTS_PER_THREAD == 1
        store_nearest_in_global(n, k, ptx, pty, index, s_centroidx, s_centroidy, d_clusterno);
  #else
        store_nearest_in_global(k, ptx, pty, index, s_centroidx, s_centroidy, d_clusterno, my_point_num);
  #endif

    }

    if(blockIdx.x == 0 && threadIdx.x == 0)
        *max_iter = numIter;
}


// Right now this code only works when
// 1. Number of threads >= k
// 2. Number of points can be > threads*blocks, i.e each thread may have to process more than one point till the shared mem allows.
// 3. All the centroids can be accomodated in shared mem. at once.
__global__ void __launch_bounds__(BLOCKDIM, MIN_NUM_BLOCKS_PER_MULTIPROC)
cluster_multi(int n, int k,int *const RESTRICT max_iter, const float *const RESTRICT d_dataptx, const float *const RESTRICT d_datapty,
        float *const RESTRICT d_centroidx, float *const RESTRICT d_centroidy, int *const RESTRICT d_syncArr,
        float *const RESTRICT d_sumx, float *const RESTRICT d_sumy, int *const RESTRICT d_count,int *const RESTRICT d_clusterno)//, int *const d_reducedCounts, clock_t *const d_timeVar)//, int *const d_dbgIter)
{
    // shared memory
    // Stores x-coord of all clusters followed by y-coord of all clusters and so on.
    extern __shared__ float s_centroid[];

    // No point of going for short int as each bank in shared mem. is 32-bit wide and only one thread
    // can access one bank at a time, so no. of accesses wont be reduced
    __shared__ signed int s_thrd_k[(BLOCKDIM)*POINTS_PER_THREAD];                                           // Store the nearest cluster of point checked by this thrd.

    __shared__ float s_thrd[(BLOCKDIM<<1)*POINTS_PER_THREAD];                                               // Store the x-coord of data point processed, for all thrds followed by y-coord.

    float *const s_centroidx = s_centroid;                                                                  // x-coord of 1st cluster's centroid
    float *const s_centroidy = s_centroid+k;                                                                // y-coord of 1st cluster's centroid

    float *const s_thrd_x = s_thrd;                                                                         // x-coord of points checked by thrds
    float *const s_thrd_y = s_thrd+(POINTS_PER_THREAD<<LOG_BLOCKDIM);                                       // y-coord of points checked by thrds

    __shared__ float s_thrd_reduce[BLOCKDIM<<1];                                                            // Store the x-coord of sum values reduced, followed by y-coord.
    float *const s_thrd_reduce_x = s_thrd_reduce;                                                                  // x-coord of sum values reduced by thrds
    float *const s_thrd_reduce_y = s_thrd_reduce+(1<<LOG_BLOCKDIM);                                                // y-coord of sum values reduced by thrds

  #if POINTS_PER_THREAD == 1
      float ptx,pty;
  #else
      float ptx[POINTS_PER_THREAD],pty[POINTS_PER_THREAD];
  #endif

    // If we've T threads per block then index of ith thread in kth block is k*T + i. So we are maintaining an order among all threads of all blocks.
    // This is same as blockIdx.x * BLOCKDIM + threadId.x
    const unsigned int index = (POINTS_PER_THREAD << LOG_BLOCKDIM)*blockIdx.x + threadIdx.x;

  #if POINTS_PER_THREAD == 1
    load_data_points(n,&ptx, &pty, index, d_dataptx, d_datapty, s_thrd_x, s_thrd_y);                            //Number of points to be processed by this thread.
  #else
    const int my_point_num = load_data_points(n,ptx, pty, index, d_dataptx, d_datapty, s_thrd_x, s_thrd_y);     //Number of points to be processed by this thread.
  #endif

    bool repeat=true;
    int numIter;
    for ( numIter = 0; repeat && (numIter < (*max_iter)); ++numIter )
    {
        load_centroids_in_shared(k, s_centroidx, s_centroidy, d_centroidx, d_centroidy);
        __syncthreads();


        // Find closest centroid if this thread holds a data-point and set it in shared mem.
  #if POINTS_PER_THREAD == 1
          store_nearest_in_shared(n, k, ptx, pty, index, s_centroidx, s_centroidy, s_thrd_k);
  #else
          store_nearest_in_shared(k, ptx, pty, s_centroidx, s_centroidy, s_thrd_k, my_point_num);
  #endif
        __syncthreads();                                                                                    // Wait for all thrds to finish setting closest k value


        // Do parallel reduction with each thread doing reduction for a distinct centroid.
        const unsigned int storageIndex = (blockIdx.x>>1)*k + threadIdx.x;                                        // The sumx, sumy and count value will be stored at this index in global mem.
        int count=0;
        float sumx=0;
        float sumy =0;

        count += reduceValues_multi(k, &sumx, &sumy, s_thrd_x, s_thrd_y, s_thrd_k);
        __syncthreads();
        reduceThreads_multi(k, &sumx, &sumy, &count, s_thrd_reduce_x, s_thrd_reduce_y, s_thrd_k);
        reduceBlocks_and_storeNewCentroids(k,numIter,gridDim.x,storageIndex,&sumx,&sumy,&count,d_sumx,d_sumy,d_count,d_syncArr, d_centroidx, d_centroidy);//, d_reducedCounts, d_timeVar);//,d_dbgIter,*max_iter);
    }

    // Final centroids have been found. Now set the cluster no. in d_clusterno array in device memory.
    {
        load_centroids_in_shared(k, s_centroidx, s_centroidy, d_centroidx, d_centroidy);
        __syncthreads();

        // Find closest centroid if this thread holds a data-point
  #if POINTS_PER_THREAD == 1
        store_nearest_in_global(n, k, ptx, pty, index, s_centroidx, s_centroidy, d_clusterno);
  #else
        store_nearest_in_global(k, ptx, pty, index, s_centroidx, s_centroidy, d_clusterno, my_point_num);
  #endif

    }

    if(blockIdx.x == 0 && threadIdx.x == 0)
        *max_iter = numIter;
}


// Right now this code only works when
// 1. Number of threads >= k
// 2. Number of points can be > threads*blocks, i.e each thread may have to process more than one point till the shared mem allows.
// 3. All the centroids can be accomodated in shared mem. at once.
__global__ void __launch_bounds__(BLOCKDIM, MIN_NUM_BLOCKS_PER_MULTIPROC)
cluster_tree(int n, int k,int *const RESTRICT max_iter, const float *const RESTRICT d_dataptx, const float *const RESTRICT d_datapty,
        float *const RESTRICT d_centroidx, float *const RESTRICT d_centroidy, int *const RESTRICT d_syncArr,
        float *const RESTRICT d_sumx, float *const RESTRICT d_sumy, int *const RESTRICT d_count,int *const RESTRICT d_clusterno)//, int *const d_reducedCounts, clock_t *const d_timeVar)//, int *const d_dbgIter)
{
    // shared memory
    // Stores x-coord of all clusters followed by y-coord of all clusters and so on.
    extern __shared__ float s_centroid[];

    // No point of going for short int as each bank in shared mem. is 32-bit wide and only one thread
    // can access one bank at a time, so no. of accesses wont be reduced
    __shared__ signed int s_thrd_k[BLOCKDIM];                                                          // Store the nearest cluster of point checked by this thrd.

    __shared__ float s_thrd[BLOCKDIM<<1];                                                               // Store the x-coord of data point processed, for all thrds followed by y-coord.

    float *const s_centroidx = s_centroid;                                                                  // x-coord of 1st cluster's centroid
    float *const s_centroidy = s_centroid+k;                                                                // y-coord of 1st cluster's centroid

    float *const s_thrd_x = s_thrd;                                                                         // x-coord of points checked by thrds
    float *const s_thrd_y = s_thrd+(1<<LOG_BLOCKDIM);                                       // y-coord of points checked by thrds


  #if POINTS_PER_THREAD == 1
      float ptx,pty;
  #else
      float ptx[POINTS_PER_THREAD],pty[POINTS_PER_THREAD];
  #endif

    // If we've T threads per block then index of ith thread in kth block is k*T + i. So we are maintaining an order among all threads of all blocks.
    // This is same as blockIdx.x * BLOCKDIM + threadId.x
    const unsigned int index = (POINTS_PER_THREAD << LOG_BLOCKDIM)*blockIdx.x + threadIdx.x;

  #if POINTS_PER_THREAD == 1
    load_data_points(n,&ptx, &pty, index, d_dataptx, d_datapty);                            //Number of points to be processed by this thread.
  #else
    const int my_point_num = load_data_points(n,ptx, pty, index, d_dataptx, d_datapty);     //Number of points to be processed by this thread.
  #endif

    bool repeat=true;
    int numIter;
    for ( numIter = 0; repeat && (numIter < (*max_iter)); ++numIter )
    {
        load_centroids_in_shared(k, s_centroidx, s_centroidy, d_centroidx, d_centroidy);
        __syncthreads();


        // Find closest centroid if this thread holds a data-point and set it in shared mem.
  #if POINTS_PER_THREAD == 1
          int closest;
          closest=store_nearest_in_register(n, k, ptx, pty, index, s_centroidx, s_centroidy);
  #else
          int closest[POINTS_PER_THREAD];
          store_nearest_in_register(k, ptx, pty, s_centroidx, s_centroidy, closest, my_point_num);
  #endif
        __syncthreads();                                                                                    // Wait for all thrds to finish setting closest k value


        // Do parallel reduction with each thread doing reduction for a distinct centroid.
        const unsigned int storageIndex = (blockIdx.x>>1)*k + threadIdx.x;                                        // The sumx, sumy and count value will be stored at this index in global mem.
        int count=0;
        float sumx=0;
        float sumy =0;

        count += reduceThreads_tree(k, &sumx, &sumy, s_thrd_x, s_thrd_y, s_thrd_k, ptx, pty, closest);


        reduceBlocks_and_storeNewCentroids(k,numIter,gridDim.x,storageIndex,&sumx,&sumy,&count,d_sumx,d_sumy,d_count,d_syncArr, d_centroidx, d_centroidy);//, d_reducedCounts, d_timeVar);//,d_dbgIter,*max_iter);
    }

    // Final centroids have been found. Now set the cluster no. in d_clusterno array in device memory.
    {
        load_centroids_in_shared(k, s_centroidx, s_centroidy, d_centroidx, d_centroidy);
        __syncthreads();

        // Find closest centroid if this thread holds a data-point
  #if POINTS_PER_THREAD == 1
        store_nearest_in_global(n, k, ptx, pty, index, s_centroidx, s_centroidy, d_clusterno);
  #else
        store_nearest_in_global(k, ptx, pty, index, s_centroidx, s_centroidy, d_clusterno, my_point_num);
  #endif

    }

    if(blockIdx.x == 0 && threadIdx.x == 0)
        *max_iter = numIter;
}


// Right now this code only works when
// 1. Number of threads >= k
// 2. Number of points can be > threads*blocks, i.e each thread may have to process more than one point till the shared mem allows.
// 3. All the centroids can be accomodated in shared mem. at once.
__global__ void __launch_bounds__(BLOCKDIM, MIN_NUM_BLOCKS_PER_MULTIPROC)
cluster1_single(int n, int k,int *const RESTRICT max_iter, const float *const RESTRICT d_dataptx, const float *const RESTRICT d_datapty,
         float *const RESTRICT d_centroidx, float *const RESTRICT d_centroidy, int *const RESTRICT d_syncArr,
         float *const RESTRICT d_sumx, float *const RESTRICT d_sumy, int *const RESTRICT d_count,int *const RESTRICT d_clusterno)//, int *const d_reducedCounts, clock_t *const d_timeVar)//, int *const d_dbgIter)
{
    // shared memory
    // Stores x-coord of all clusters followed by y-coord of all clusters i.e. x and y coord of a cluster are separated by k.
    extern __shared__ float s_centroid[];

    // No point of going for short int as each bank in shared mem. is 32-bit wide and only one thread
    // can access one bank at a time, so no. of accesses wont be reduced
    __shared__ signed int s_thrd_k[(BLOCKDIM)*POINTS_PER_THREAD];                                           // Store the nearest cluster of point checked by this thrd.

    __shared__ float s_thrd[(BLOCKDIM<<1)*POINTS_PER_THREAD];                                               // Store the x-coord of data point processed, for all thrds followed by y-coord.

    float *const s_centroidx = s_centroid;                                                                  // x-coord of 1st cluster's centroid
    float *const s_centroidy = s_centroid+k;                                                                // y-coord of 1st cluster's centroid

    float *const s_thrd_x = s_thrd;                                                                         // x-coord of points checked by thrds
    float *const s_thrd_y = s_thrd+(POINTS_PER_THREAD<<LOG_BLOCKDIM);                                       // y-coord of points checked by thrds


  #if POINTS_PER_THREAD == 1
    float ptx,pty;
  #else
    float ptx[POINTS_PER_THREAD],pty[POINTS_PER_THREAD];
  #endif

    const unsigned int total_point_cycles=(n-1)/((POINTS_PER_THREAD<<LOG_BLOCKDIM)*gridDim.x) + 1;

    bool repeat=true;
    int numIter;
    for ( numIter = 0; repeat && (numIter < *max_iter); ++numIter )
    {
        load_centroids_in_shared(k, s_centroidx, s_centroidy, d_centroidx, d_centroidy);

        __syncthreads();

        float sumx=0;
        float sumy =0;
        int count=0;
        const unsigned int storageIndex = (blockIdx.x>>1)*k + threadIdx.x;                         // The sumx, sumy and count value will be stored at this index in global mem.
        for( int curr_point_cycle=0; curr_point_cycle < total_point_cycles; curr_point_cycle++)
        {

            // If we've T threads per block then index of ith thread in kth block is k*T + i. So we are maintaining an order among all threads of all blocks.
            // This is same as blockIdx.x * BLOCKDIM + threadId.x
            unsigned int index = (POINTS_PER_THREAD << LOG_BLOCKDIM)*(curr_point_cycle*gridDim.x + blockIdx.x) + threadIdx.x;

  #if POINTS_PER_THREAD == 1
            load_data_points(n,&ptx, &pty, index, d_dataptx, d_datapty, s_thrd_x, s_thrd_y);                            //Number of points to be processed by this thread.
  #else
            int my_point_num = load_data_points(n,curr_point_cycle, total_point_cycles, ptx, pty, index, d_dataptx, d_datapty, s_thrd_x, s_thrd_y);     //Number of points to be processed by this thread.
  #endif


            // Find closest centroid if this thread holds a data-point and set it in shared mem.
  #if POINTS_PER_THREAD == 1
            store_nearest_in_shared(n, k, ptx, pty, index, s_centroidx, s_centroidy, s_thrd_k);
  #else
            store_nearest_in_shared(k, ptx, pty, s_centroidx, s_centroidy, s_thrd_k, my_point_num);
  #endif
            __syncthreads();                                                                                    // Wait for all thrds to finish setting closest k value

            // Do parallel reduction with each thread doing reduction for a distinct centroid.
            count+= reduceThreads_single(k, &sumx, &sumy, s_thrd_x, s_thrd_y, s_thrd_k);

            __syncthreads();
        }

        reduceBlocks_and_storeNewCentroids(k,numIter,gridDim.x,storageIndex,&sumx,&sumy,&count,d_sumx,d_sumy,d_count,d_syncArr, d_centroidx, d_centroidy);//, d_reducedCounts, d_timeVar);//,d_dbgIter,*max_iter);

    }

    // Iterations are over and final centroids have been found. Now set the cluster no. in d_clusterno array in device memory.
    {
        load_centroids_in_shared(k, s_centroidx, s_centroidy, d_centroidx, d_centroidy);
        __syncthreads();
        for( int curr_point_cycle=0; curr_point_cycle < total_point_cycles; curr_point_cycle++)
        {
            // If we've T threads per block then index of ith thread in kth block is k*T + i. So we are maintaining an order among all threads of all blocks.
            // This is same as blockIdx.x * BLOCKDIM + threadId.x
            unsigned int index = (POINTS_PER_THREAD << LOG_BLOCKDIM)*(curr_point_cycle*gridDim.x + blockIdx.x) + threadIdx.x;

  #if POINTS_PER_THREAD == 1
            load_data_points(n,&ptx, &pty, index, d_dataptx, d_datapty, s_thrd_x, s_thrd_y);                            //Number of points to be processed by this thread.
  #else
            const int my_point_num = load_data_points(n,ptx, pty, index, d_dataptx, d_datapty, s_thrd_x, s_thrd_y);     //Number of points to be processed by this thread.
  #endif

            // Find closest centroid if this thread holds a data-point
  #if POINTS_PER_THREAD == 1
            store_nearest_in_global(n, k, ptx, pty, index, s_centroidx, s_centroidy, d_clusterno);
  #else
            store_nearest_in_global(k, ptx, pty, index, s_centroidx, s_centroidy, d_clusterno, my_point_num);
  #endif
        }

    }

    if(blockIdx.x == 0 && threadIdx.x == 0)
        *max_iter = numIter;
}


// Right now this code only works when
// 1. Number of threads >= k
// 2. Number of points can be > threads*blocks, i.e each thread may have to process more than one point till the shared mem allows.
// 3. All the centroids can be accomodated in shared mem. at once.
__global__ void __launch_bounds__(BLOCKDIM, MIN_NUM_BLOCKS_PER_MULTIPROC)
cluster1_multi(int n, int k,int *const RESTRICT max_iter, const float *const RESTRICT d_dataptx, const float *const RESTRICT d_datapty,
         float *const RESTRICT d_centroidx, float *const RESTRICT d_centroidy, int *const RESTRICT d_syncArr,
         float *const RESTRICT d_sumx, float *const RESTRICT d_sumy, int *const RESTRICT d_count,int *const RESTRICT d_clusterno)//, int *const d_reducedCounts, clock_t *const d_timeVar)//, int *const d_dbgIter)
{
    // shared memory
    // Stores x-coord of all clusters followed by y-coord of all clusters i.e. x and y coord of a cluster are separated by k.
    extern __shared__ float s_centroid[];

    // No point of going for short int as each bank in shared mem. is 32-bit wide and only one thread
    // can access one bank at a time, so no. of accesses wont be reduced
    __shared__ signed int s_thrd_k[(BLOCKDIM)*POINTS_PER_THREAD];                                           // Store the nearest cluster of point checked by this thrd.

    __shared__ float s_thrd[(BLOCKDIM<<1)*POINTS_PER_THREAD];                                               // Store the x-coord of data point processed, for all thrds followed by y-coord.

    float *const s_centroidx = s_centroid;                                                                  // x-coord of 1st cluster's centroid
    float *const s_centroidy = s_centroid+k;                                                                // y-coord of 1st cluster's centroid

    float *const s_thrd_x = s_thrd;                                                                         // x-coord of points checked by thrds
    float *const s_thrd_y = s_thrd+(POINTS_PER_THREAD<<LOG_BLOCKDIM);                                       // y-coord of points checked by thrds


  #if POINTS_PER_THREAD == 1
    float ptx,pty;
  #else
    float ptx[POINTS_PER_THREAD],pty[POINTS_PER_THREAD];
  #endif

    const unsigned int total_point_cycles=(n-1)/((POINTS_PER_THREAD<<LOG_BLOCKDIM)*gridDim.x) + 1;

    bool repeat=true;
    int numIter;
    for ( numIter = 0; repeat && (numIter < *max_iter); ++numIter )
    {
        load_centroids_in_shared(k, s_centroidx, s_centroidy, d_centroidx, d_centroidy);

        __syncthreads();

        float sumx=0;
        float sumy =0;
        int count=0;
        const unsigned int storageIndex = (blockIdx.x>>1)*k + threadIdx.x;                         // The sumx, sumy and count value will be stored at this index in global mem.
        for( int curr_point_cycle=0; curr_point_cycle < total_point_cycles; curr_point_cycle++)
        {

            // If we've T threads per block then index of ith thread in kth block is k*T + i. So we are maintaining an order among all threads of all blocks.
            // This is same as blockIdx.x * BLOCKDIM + threadId.x
            unsigned int index = (POINTS_PER_THREAD << LOG_BLOCKDIM)*(curr_point_cycle*gridDim.x + blockIdx.x) + threadIdx.x;

  #if POINTS_PER_THREAD == 1
            load_data_points(n,&ptx, &pty, index, d_dataptx, d_datapty, s_thrd_x, s_thrd_y);                            //Number of points to be processed by this thread.
  #else
            int my_point_num = load_data_points(n,curr_point_cycle, total_point_cycles, ptx, pty, index, d_dataptx, d_datapty, s_thrd_x, s_thrd_y);     //Number of points to be processed by this thread.
  #endif


            // Find closest centroid if this thread holds a data-point and set it in shared mem.
  #if POINTS_PER_THREAD == 1
            store_nearest_in_shared(n, k, ptx, pty, index, s_centroidx, s_centroidy, s_thrd_k);
  #else
            store_nearest_in_shared(k, ptx, pty, s_centroidx, s_centroidy, s_thrd_k, my_point_num);
  #endif
            __syncthreads();                                                                                    // Wait for all thrds to finish setting closest k value

            // Do parallel reduction with each thread doing reduction for a distinct centroid.
            count+= reduceValues_multi(k, &sumx, &sumy, s_thrd_x, s_thrd_y, s_thrd_k);

            __syncthreads();
        }
        reduceThreads_multi(k, &sumx, &sumy, &count, s_thrd_x, s_thrd_y, s_thrd_k);
        reduceBlocks_and_storeNewCentroids(k,numIter,gridDim.x,storageIndex,&sumx,&sumy,&count,d_sumx,d_sumy,d_count,d_syncArr, d_centroidx, d_centroidy);//, d_reducedCounts, d_timeVar);//,d_dbgIter,*max_iter);

    }

    // Iterations are over and final centroids have been found. Now set the cluster no. in d_clusterno array in device memory.
    {
        load_centroids_in_shared(k, s_centroidx, s_centroidy, d_centroidx, d_centroidy);
        __syncthreads();
        for( int curr_point_cycle=0; curr_point_cycle < total_point_cycles; curr_point_cycle++)
        {
            // If we've T threads per block then index of ith thread in kth block is k*T + i. So we are maintaining an order among all threads of all blocks.
            // This is same as blockIdx.x * BLOCKDIM + threadId.x
            unsigned int index = (POINTS_PER_THREAD << LOG_BLOCKDIM)*(curr_point_cycle*gridDim.x + blockIdx.x) + threadIdx.x;

  #if POINTS_PER_THREAD == 1
            load_data_points(n,&ptx, &pty, index, d_dataptx, d_datapty, s_thrd_x, s_thrd_y);                            //Number of points to be processed by this thread.
  #else
            const int my_point_num = load_data_points(n,ptx, pty, index, d_dataptx, d_datapty, s_thrd_x, s_thrd_y);     //Number of points to be processed by this thread.
  #endif

            // Find closest centroid if this thread holds a data-point
  #if POINTS_PER_THREAD == 1
            store_nearest_in_global(n, k, ptx, pty, index, s_centroidx, s_centroidy, d_clusterno);
  #else
            store_nearest_in_global(k, ptx, pty, index, s_centroidx, s_centroidy, d_clusterno, my_point_num);
  #endif
        }

    }

    if(blockIdx.x == 0 && threadIdx.x == 0)
        *max_iter = numIter;
}


// Right now this code only works when
// 1. Number of threads >= k
// 2. Number of points can be > threads*blocks, i.e each thread may have to process more than one point till the shared mem allows.
// 3. All the centroids can be accomodated in shared mem. at once.
__global__ void __launch_bounds__(BLOCKDIM, MIN_NUM_BLOCKS_PER_MULTIPROC)
cluster1_tree(int n, int k,int *const RESTRICT max_iter, const float *const RESTRICT d_dataptx, const float *const RESTRICT d_datapty,
         float *const RESTRICT d_centroidx, float *const RESTRICT d_centroidy, int *const RESTRICT d_syncArr,
         float *const RESTRICT d_sumx, float *const RESTRICT d_sumy, int *const RESTRICT d_count,int *const RESTRICT d_clusterno)//, int *const d_reducedCounts, clock_t *const d_timeVar)//, int *const d_dbgIter)
{
    // shared memory
    // Stores x-coord of all clusters followed by y-coord of all clusters i.e. x and y coord of a cluster are separated by k.
    extern __shared__ float s_centroid[];

    // No point of going for short int as each bank in shared mem. is 32-bit wide and only one thread
    // can access one bank at a time, so no. of accesses wont be reduced
    __shared__ signed int s_thrd_k[BLOCKDIM];                                           // Store the nearest cluster of point checked by this thrd.

    __shared__ float s_thrd[BLOCKDIM<<1];                                               // Store the x-coord of data point processed, for all thrds followed by y-coord.

    float *const s_centroidx = s_centroid;                                                                  // x-coord of 1st cluster's centroid
    float *const s_centroidy = s_centroid+k;                                                                // y-coord of 1st cluster's centroid

    float *const s_thrd_x = s_thrd;                                                                         // x-coord of points checked by thrds
    float *const s_thrd_y = s_thrd+(1<<LOG_BLOCKDIM);                                       // y-coord of points checked by thrds


  #if POINTS_PER_THREAD == 1
    float ptx,pty;
  #else
    float ptx[POINTS_PER_THREAD],pty[POINTS_PER_THREAD];
  #endif

    const unsigned int total_point_cycles=(n-1)/((POINTS_PER_THREAD<<LOG_BLOCKDIM)*gridDim.x) + 1;

    bool repeat=true;
    int numIter;
    for ( numIter = 0; repeat && (numIter < *max_iter); ++numIter )
    {
        load_centroids_in_shared(k, s_centroidx, s_centroidy, d_centroidx, d_centroidy);

        __syncthreads();

        float sumx=0;
        float sumy =0;
        int count=0;
        const unsigned int storageIndex = (blockIdx.x>>1)*k + threadIdx.x;                         // The sumx, sumy and count value will be stored at this index in global mem.
        for( int curr_point_cycle=0; curr_point_cycle < total_point_cycles; curr_point_cycle++)
        {

            // If we've T threads per block then index of ith thread in kth block is k*T + i. So we are maintaining an order among all threads of all blocks.
            // This is same as blockIdx.x * BLOCKDIM + threadId.x
            unsigned int index = (POINTS_PER_THREAD << LOG_BLOCKDIM)*(curr_point_cycle*gridDim.x + blockIdx.x) + threadIdx.x;

  #if POINTS_PER_THREAD == 1
            load_data_points(n,&ptx, &pty, index, d_dataptx, d_datapty);                            //Number of points to be processed by this thread.
  #else
            int my_point_num = load_data_points(n,curr_point_cycle, total_point_cycles, ptx, pty, index, d_dataptx, d_datapty);     //Number of points to be processed by this thread.
  #endif


            // Find closest centroid if this thread holds a data-point and set it in shared mem.
  #if POINTS_PER_THREAD == 1
            int closest;
            closest=store_nearest_in_register(n, k, ptx, pty, index, s_centroidx, s_centroidy);
  #else
          int closest[POINTS_PER_THREAD];
          store_nearest_in_register(k, ptx, pty, s_centroidx, s_centroidy, closest, my_point_num);
  #endif
            __syncthreads();                                                                                    // Wait for all thrds to finish setting closest k value

            // Do parallel reduction with each thread doing reduction for a distinct centroid.
            count += reduceThreads_tree(k, &sumx, &sumy, s_thrd_x, s_thrd_y, s_thrd_k, ptx, pty, closest);
            __syncthreads();
        }

        reduceBlocks_and_storeNewCentroids(k,numIter,gridDim.x,storageIndex,&sumx,&sumy,&count,d_sumx,d_sumy,d_count,d_syncArr, d_centroidx, d_centroidy);//, d_reducedCounts, d_timeVar);//,d_dbgIter,*max_iter);

    }

    // Iterations are over and final centroids have been found. Now set the cluster no. in d_clusterno array in device memory.
    {
        load_centroids_in_shared(k, s_centroidx, s_centroidy, d_centroidx, d_centroidy);
        __syncthreads();
        for( int curr_point_cycle=0; curr_point_cycle < total_point_cycles; curr_point_cycle++)
        {
            // If we've T threads per block then index of ith thread in kth block is k*T + i. So we are maintaining an order among all threads of all blocks.
            // This is same as blockIdx.x * BLOCKDIM + threadId.x
            unsigned int index = (POINTS_PER_THREAD << LOG_BLOCKDIM)*(curr_point_cycle*gridDim.x + blockIdx.x) + threadIdx.x;

  #if POINTS_PER_THREAD == 1
            load_data_points(n,&ptx, &pty, index, d_dataptx, d_datapty);                            //Number of points to be processed by this thread.
  #else
            const int my_point_num = load_data_points(n,ptx, pty, index, d_dataptx, d_datapty);     //Number of points to be processed by this thread.
  #endif

            // Find closest centroid if this thread holds a data-point
  #if POINTS_PER_THREAD == 1
            store_nearest_in_global(n, k, ptx, pty, index, s_centroidx, s_centroidy, d_clusterno);
  #else
            store_nearest_in_global(k, ptx, pty, index, s_centroidx, s_centroidy, d_clusterno, my_point_num);
  #endif
        }

    }

    if(blockIdx.x == 0 && threadIdx.x == 0)
        *max_iter = numIter;
}


__device__ int inline init_data(const int curr_point_cycle, const int total_point_cycles, const int n, float *const RESTRICT ptx,
                                float *const RESTRICT pty, unsigned int *const RESTRICT index, const float *const RESTRICT d_dataptx,
                                const float *const RESTRICT d_datapty, float *const RESTRICT s_thrd_x, float *const RESTRICT s_thrd_y)
{

    // If we've T threads per block then index of ith thread in kth block is k*T + i. So we are maintaining an order among all threads of all blocks.
    // This is same as blockIdx.x * BLOCKDIM + threadId.x
    *index = (POINTS_PER_THREAD << LOG_BLOCKDIM)*(curr_point_cycle*gridDim.x + blockIdx.x) + threadIdx.x;

  #if POINTS_PER_THREAD == 1
    int my_point_num = load_data_points(n,ptx, pty, *index, d_dataptx, d_datapty, s_thrd_x, s_thrd_y);                            //Number of points to be processed by this thread.
  #else
    int my_point_num = load_data_points(n,curr_point_cycle, total_point_cycles, ptx, pty, *index, d_dataptx, d_datapty, s_thrd_x, s_thrd_y);     //Number of points to be processed by this thread.
  #endif

    return my_point_num;
}


__device__ int inline init_data(const int curr_point_cycle, const int total_point_cycles, const int n, float *const RESTRICT ptx,
                                float *const RESTRICT pty, unsigned int *const RESTRICT index, const float *const RESTRICT d_dataptx,
                                const float *const RESTRICT d_datapty)
{

    // If we've T threads per block then index of ith thread in kth block is k*T + i. So we are maintaining an order among all threads of all blocks.
    // This is same as blockIdx.x * BLOCKDIM + threadId.x
    *index = (POINTS_PER_THREAD << LOG_BLOCKDIM)*(curr_point_cycle*gridDim.x + blockIdx.x) + threadIdx.x;

  #if POINTS_PER_THREAD == 1
    int my_point_num = load_data_points(n,ptx, pty, *index, d_dataptx, d_datapty);                            //Number of points to be processed by this thread.
  #else
    int my_point_num = load_data_points(n,curr_point_cycle, total_point_cycles, ptx, pty, *index, d_dataptx, d_datapty);     //Number of points to be processed by this thread.
  #endif

    return my_point_num;
}


__device__ inline void reduceThreads_and_setCentroidVars_single(const int n, const int k, float *const RESTRICT ptx, float *const RESTRICT pty, float *const RESTRICT s_centroidx,
                                                         float *const RESTRICT s_centroidy, float *const RESTRICT s_thrd_x, float *const RESTRICT s_thrd_y,
                                                         int *const RESTRICT s_thrd_k, const unsigned int *const RESTRICT index, const int *const RESTRICT my_point_num,
                                                         float *const RESTRICT sumx, float *const RESTRICT sumy, int *const RESTRICT count)
{
    // Find closest centroid if this thread holds a data-point and set it in shared mem.
  #if POINTS_PER_THREAD == 1
    store_nearest_in_shared(n, k, *ptx, *pty, *index, s_centroidx, s_centroidy, s_thrd_k);
  #else
    store_nearest_in_shared(k, ptx, pty, s_centroidx, s_centroidy, s_thrd_k, *my_point_num);
  #endif
    __syncthreads();                                                                                    // Wait for all thrds to finish setting closest k value

    // Do parallel reduction with each thread doing reduction for a distinct centroid.
    (*count) += reduceThreads_single(k, sumx, sumy, s_thrd_x, s_thrd_y, s_thrd_k);
    __syncthreads();
}


__device__ inline void reduceValues_and_setCentroidVars_multi(const int n, const int k, float *const RESTRICT ptx, float *const RESTRICT pty, float *const RESTRICT s_centroidx,
                                                         float *const RESTRICT s_centroidy, float *const RESTRICT s_thrd_x, float *const RESTRICT s_thrd_y,
                                                         int *const RESTRICT s_thrd_k, const unsigned int *const RESTRICT index, const int *const RESTRICT my_point_num,
                                                         float *const RESTRICT sumx, float *const RESTRICT sumy, int *const RESTRICT count)
{
    // Find closest centroid if this thread holds a data-point and set it in shared mem.
  #if POINTS_PER_THREAD == 1
    store_nearest_in_shared(n, k, *ptx, *pty, *index, s_centroidx, s_centroidy, s_thrd_k);
  #else
    store_nearest_in_shared(k, ptx, pty, s_centroidx, s_centroidy, s_thrd_k, *my_point_num);
  #endif
    __syncthreads();                                                                                    // Wait for all thrds to finish setting closest k value

    // Do parallel reduction with each thread doing reduction for a distinct centroid.
    (*count) += reduceValues_multi(k, sumx, sumy, s_thrd_x, s_thrd_y, s_thrd_k);
    __syncthreads();
}


__device__ inline void reduceThreads_and_setCentroidVars_tree(const int n, const int k, float *const RESTRICT ptx, float *const RESTRICT pty, float *const RESTRICT s_centroidx,
                                                         float *const RESTRICT s_centroidy, float *const RESTRICT s_thrd_x, float *const RESTRICT s_thrd_y,
                                                         int *const RESTRICT s_thrd_k, const unsigned int *const RESTRICT index, const int *const RESTRICT my_point_num,
                                                         float *const RESTRICT sumx, float *const RESTRICT sumy, int *const RESTRICT count)
{
    // Find closest centroid if this thread holds a data-point and set it in shared mem.
  #if POINTS_PER_THREAD == 1
    int closest;
    closest = store_nearest_in_register(n, k, *ptx, *pty, *index, s_centroidx, s_centroidy);
  #else
    int closest[POINTS_PER_THREAD];
    store_nearest_in_register(k, ptx, pty, s_centroidx, s_centroidy, closest, *my_point_num);
  #endif
    __syncthreads();                                                                                    // Wait for all thrds to finish setting closest k value

    // Do parallel reduction with each thread doing reduction for a distinct centroid.
  #if POINTS_PER_THREAD == 1
    (*count) += reduceThreads_tree(k, sumx, sumy, s_thrd_x, s_thrd_y, s_thrd_k, *ptx, *pty, closest);
  #else
    (*count) += reduceThreads_tree(k, sumx, sumy, s_thrd_x, s_thrd_y, s_thrd_k, ptx, pty, closest);
  #endif
    __syncthreads();
}


__device__ void inline init_centroids(const int k, float *const RESTRICT s_centroidx, float *const RESTRICT s_centroidy,
                                      const float *const RESTRICT d_centroidx, const float *const RESTRICT d_centroidy,
                                      float *const RESTRICT sumx, float *const RESTRICT sumy, int *const RESTRICT count)
{

    *sumx=0.0f;
    *sumy=0.0f;
    load_centroids_in_shared(k, s_centroidx, s_centroidy, d_centroidx, d_centroidy);
    *count=0;
    __syncthreads();

}


// Right now this code only works when
// 1. Number of threads >= k
// 2. Number of points can be > threads*blocks, i.e each thread may have to process more than one point till the shared mem allows.
// 3. All the centroids can be accomodated in shared mem. at once.
__global__ void __launch_bounds__(BLOCKDIM, MIN_NUM_BLOCKS_PER_MULTIPROC)
cluster1_load_optimized_single(int n, int k,int *const RESTRICT max_iter, const float *const RESTRICT d_dataptx, const float *const RESTRICT d_datapty,
                       float *const RESTRICT d_centroidx, float *const RESTRICT d_centroidy, int *const RESTRICT d_syncArr,
                       float *const RESTRICT d_sumx, float *const RESTRICT d_sumy, int *const RESTRICT d_count,int *const RESTRICT d_clusterno)
{
    // shared memory
    // Stores x-coord of all clusters followed by y-coord of all clusters i.e. x and y coord of a cluster are separated by k.
    extern __shared__ float s_centroid[];

    // No point of going for short int as each bank in shared mem. is 32-bit wide and only one thread
    // can access one bank at a time, so no. of accesses wont be reduced
    __shared__ signed int s_thrd_k[(BLOCKDIM)*POINTS_PER_THREAD];                                           // Store the nearest cluster of point checked by this thrd.

    __shared__ float s_thrd[(BLOCKDIM<<1)*POINTS_PER_THREAD];                                               // Store the x-coord of data point processed, for all thrds followed by y-coord.

    float *const s_centroidx = s_centroid;                                                                  // x-coord of 1st cluster's centroid
    float *const s_centroidy = s_centroid+k;                                                                // y-coord of 1st cluster's centroid

    float *const s_thrd_x = s_thrd;                                                                         // x-coord of points checked by thrds
    float *const s_thrd_y = s_thrd+(POINTS_PER_THREAD<<LOG_BLOCKDIM);                                       // y-coord of points checked by thrds

  #if POINTS_PER_THREAD == 1
    float ptx,pty;
  #else
    float ptx[POINTS_PER_THREAD],pty[POINTS_PER_THREAD];
  #endif

    const unsigned int total_point_cycles=(n-1)/((POINTS_PER_THREAD<<LOG_BLOCKDIM)*gridDim.x) + 1;

    bool repeat=true;
    int numIter=0;

    int count=0;
    const unsigned int storageIndex = (blockIdx.x>>1)*k + threadIdx.x;                            // The sumx, sumy and count value will be stored at this index in global mem.

    int curr_point_cycle=0;
    unsigned int index;

    float sumx=0;
    float sumy =0;
  #if POINTS_PER_THREAD == 1
    int my_point_num = init_data(curr_point_cycle, total_point_cycles, n, &ptx, &pty, &index, d_dataptx, d_datapty, s_thrd_x, s_thrd_y);
  #else
    int my_point_num = init_data(curr_point_cycle, total_point_cycles, n, ptx, pty, &index, d_dataptx, d_datapty, s_thrd_x, s_thrd_y);
  #endif

    for ( numIter = 0; repeat && (numIter < *max_iter); ++numIter )
    {
        {
            // ************ 0th, 2nd and so on all even iterations ************
            init_centroids(k, s_centroidx, s_centroidy, d_centroidx, d_centroidy, &sumx, &sumy, &count);
            // We already have points loaded for first point_cycle
  #if POINTS_PER_THREAD == 1
            reduceThreads_and_setCentroidVars_single(n, k, &ptx, &pty, s_centroidx, s_centroidy, s_thrd_x, s_thrd_y, s_thrd_k, &index, &my_point_num, &sumx, &sumy, &count);
  #else
            reduceThreads_and_setCentroidVars_single(n, k, ptx, pty, s_centroidx, s_centroidy, s_thrd_x, s_thrd_y, s_thrd_k, &index, &my_point_num, &sumx, &sumy, &count);
  #endif

            for( curr_point_cycle=1; curr_point_cycle < total_point_cycles; curr_point_cycle++)
            {
  #if POINTS_PER_THREAD == 1
                init_data(curr_point_cycle, total_point_cycles, n, &ptx, &pty, &index, d_dataptx, d_datapty, s_thrd_x, s_thrd_y);
                reduceThreads_and_setCentroidVars_single(n, k, &ptx, &pty, s_centroidx, s_centroidy, s_thrd_x, s_thrd_y, s_thrd_k, &index, &my_point_num, &sumx, &sumy, &count);
  #else
                my_point_num = init_data(curr_point_cycle, total_point_cycles, n, ptx, pty, &index, d_dataptx, d_datapty, s_thrd_x, s_thrd_y);
                reduceThreads_and_setCentroidVars_single(n, k, ptx, pty, s_centroidx, s_centroidy, s_thrd_x, s_thrd_y, s_thrd_k, &index, &my_point_num, &sumx, &sumy, &count);
  #endif
            }
            reduceBlocks_and_storeNewCentroids(k,numIter,gridDim.x,storageIndex,&sumx,&sumy,&count,d_sumx,d_sumy,d_count,d_syncArr, d_centroidx, d_centroidy);//, d_reducedCounts, d_timeVar);//,d_dbgIter,*max_iter);
        }

        if(((++numIter) < (*max_iter)) && repeat)
        {
            // ************ 1st, 3rd and so on all odd iterations ************

            init_centroids(k, s_centroidx, s_centroidy, d_centroidx, d_centroidy, &sumx, &sumy, &count);
            // We already have points loaded for last point_cycle
  #if POINTS_PER_THREAD == 1
            reduceThreads_and_setCentroidVars_single(n, k, &ptx, &pty, s_centroidx, s_centroidy, s_thrd_x, s_thrd_y, s_thrd_k, &index, &my_point_num, &sumx, &sumy, &count);
  #else
            reduceThreads_and_setCentroidVars_single(n, k, ptx, pty, s_centroidx, s_centroidy, s_thrd_x, s_thrd_y, s_thrd_k, &index, &my_point_num, &sumx, &sumy, &count);
  #endif
            for( curr_point_cycle=total_point_cycles-2; curr_point_cycle >= 0; curr_point_cycle--)
            {
  #if POINTS_PER_THREAD == 1
                init_data(curr_point_cycle, total_point_cycles, n, &ptx, &pty, &index, d_dataptx, d_datapty, s_thrd_x, s_thrd_y);
                reduceThreads_and_setCentroidVars_single(n, k, &ptx, &pty, s_centroidx, s_centroidy, s_thrd_x, s_thrd_y, s_thrd_k, &index, &my_point_num, &sumx, &sumy, &count);
  #else
                my_point_num = init_data(curr_point_cycle, total_point_cycles, n, ptx, pty, &index, d_dataptx, d_datapty, s_thrd_x, s_thrd_y);
                reduceThreads_and_setCentroidVars_single(n, k, ptx, pty, s_centroidx, s_centroidy, s_thrd_x, s_thrd_y, s_thrd_k, &index, &my_point_num, &sumx, &sumy, &count);
  #endif
            }

            reduceBlocks_and_storeNewCentroids(k,numIter,gridDim.x,storageIndex,&sumx,&sumy,&count,d_sumx,d_sumy,d_count,d_syncArr, d_centroidx, d_centroidy);//, d_reducedCounts, d_timeVar);//,d_dbgIter,*max_iter);

        }

    }

    // Iterations are over and final centroids have been found. Now set the cluster no. in d_clusterno array in device memory.
    {
        load_centroids_in_shared(k, s_centroidx, s_centroidy, d_centroidx, d_centroidy);
        __syncthreads();

        // Store nearest in global for last loaded point.
        // Find closest centroid if this thread holds a data-point
  #if POINTS_PER_THREAD == 1
        store_nearest_in_global(n, k, ptx, pty, index, s_centroidx, s_centroidy, d_clusterno);
  #else
        store_nearest_in_global(k, ptx, pty, index, s_centroidx, s_centroidy, d_clusterno, my_point_num);
  #endif

        // Assuming last loaded points were of first point cycle
        int nextCycle=1;
        int goTill=total_point_cycles;

        if(curr_point_cycle==total_point_cycles)
        {
            // Last loaded points were of last point cycle
            nextCycle=0;
            goTill=total_point_cycles-1;
        }

        for( int curr_point_cycle=nextCycle; curr_point_cycle < goTill; curr_point_cycle++)
        {
  #if POINTS_PER_THREAD == 1
            init_data(curr_point_cycle, total_point_cycles, n, &ptx, &pty, &index, d_dataptx, d_datapty, s_thrd_x, s_thrd_y);
            // Find closest centroid if this thread holds a data-point
            store_nearest_in_global(n, k, ptx, pty, index, s_centroidx, s_centroidy, d_clusterno);
  #else
            my_point_num=init_data(curr_point_cycle, total_point_cycles, n, ptx, pty, &index, d_dataptx, d_datapty, s_thrd_x, s_thrd_y);
            // Find closest centroid if this thread holds a data-point
            store_nearest_in_global(k, ptx, pty, index, s_centroidx, s_centroidy, d_clusterno, my_point_num);
  #endif
        }

    }

    if(blockIdx.x == 0 && threadIdx.x == 0)
        *max_iter = numIter;
}


// Right now this code only works when
// 1. Number of threads >= k
// 2. Number of points can be > threads*blocks, i.e each thread may have to process more than one point till the shared mem allows.
// 3. All the centroids can be accomodated in shared mem. at once.
__global__ void __launch_bounds__(BLOCKDIM, MIN_NUM_BLOCKS_PER_MULTIPROC)
cluster1_load_optimized_multi(int n, int k,int *const RESTRICT max_iter, const float *const RESTRICT d_dataptx, const float *const RESTRICT d_datapty,
                       float *const RESTRICT d_centroidx, float *const RESTRICT d_centroidy, int *const RESTRICT d_syncArr,
                       float *const RESTRICT d_sumx, float *const RESTRICT d_sumy, int *const RESTRICT d_count,int *const RESTRICT d_clusterno)
{
    // shared memory
    // Stores x-coord of all clusters followed by y-coord of all clusters i.e. x and y coord of a cluster are separated by k.
    extern __shared__ float s_centroid[];

    // No point of going for short int as each bank in shared mem. is 32-bit wide and only one thread
    // can access one bank at a time, so no. of accesses wont be reduced
    __shared__ signed int s_thrd_k[(BLOCKDIM)*POINTS_PER_THREAD];                                           // Store the nearest cluster of point checked by this thrd.

    __shared__ float s_thrd[(BLOCKDIM<<1)*POINTS_PER_THREAD];                                               // Store the x-coord of data point processed, for all thrds followed by y-coord.

    float *const s_centroidx = s_centroid;                                                                  // x-coord of 1st cluster's centroid
    float *const s_centroidy = s_centroid+k;                                                                // y-coord of 1st cluster's centroid

    float *const s_thrd_x = s_thrd;                                                                         // x-coord of points checked by thrds
    float *const s_thrd_y = s_thrd+(POINTS_PER_THREAD<<LOG_BLOCKDIM);                                       // y-coord of points checked by thrds

    __shared__ float s_thrd_reduce[BLOCKDIM<<1];                                                            // Store the x-coord of sum values reduced, followed by y-coord.
    float *const s_thrd_reduce_x = s_thrd_reduce;                                                                  // x-coord of sum values reduced by thrds
    float *const s_thrd_reduce_y = s_thrd_reduce+(1<<LOG_BLOCKDIM);                                                // y-coord of sum values reduced by thrds

  #if POINTS_PER_THREAD == 1
    float ptx,pty;
  #else
    float ptx[POINTS_PER_THREAD],pty[POINTS_PER_THREAD];
  #endif

    const unsigned int total_point_cycles=(n-1)/((POINTS_PER_THREAD<<LOG_BLOCKDIM)*gridDim.x) + 1;

    bool repeat=true;
    int numIter=0;

    int count=0;
    const unsigned int storageIndex = (blockIdx.x>>1)*k + threadIdx.x;                            // The sumx, sumy and count value will be stored at this index in global mem.

    int curr_point_cycle=0;
    unsigned int index;

    float sumx=0;
    float sumy =0;
  #if POINTS_PER_THREAD == 1
    int my_point_num = init_data(curr_point_cycle, total_point_cycles, n, &ptx, &pty, &index, d_dataptx, d_datapty, s_thrd_x, s_thrd_y);
  #else
    int my_point_num = init_data(curr_point_cycle, total_point_cycles, n, ptx, pty, &index, d_dataptx, d_datapty, s_thrd_x, s_thrd_y);
  #endif

    for ( numIter = 0; repeat && (numIter < *max_iter); ++numIter )
    {
        {
            // ************ 0th, 2nd and so on all even iterations ************
            init_centroids(k, s_centroidx, s_centroidy, d_centroidx, d_centroidy, &sumx, &sumy, &count);
            // We already have points loaded for first point_cycle
  #if POINTS_PER_THREAD == 1
            reduceValues_and_setCentroidVars_multi(n, k, &ptx, &pty, s_centroidx, s_centroidy, s_thrd_x, s_thrd_y, s_thrd_k, &index, &my_point_num, &sumx, &sumy, &count);
  #else
            reduceValues_and_setCentroidVars_multi(n, k, ptx, pty, s_centroidx, s_centroidy, s_thrd_x, s_thrd_y, s_thrd_k, &index, &my_point_num, &sumx, &sumy, &count);
  #endif


            for( curr_point_cycle=1; curr_point_cycle < total_point_cycles; curr_point_cycle++)
            {
  #if POINTS_PER_THREAD == 1
                init_data(curr_point_cycle, total_point_cycles, n, &ptx, &pty, &index, d_dataptx, d_datapty, s_thrd_x, s_thrd_y);
                reduceValues_and_setCentroidVars_multi(n, k, &ptx, &pty, s_centroidx, s_centroidy, s_thrd_x, s_thrd_y, s_thrd_k, &index, &my_point_num, &sumx, &sumy, &count);
  #else
                my_point_num = init_data(curr_point_cycle, total_point_cycles, n, ptx, pty, &index, d_dataptx, d_datapty, s_thrd_x, s_thrd_y);
                reduceValues_and_setCentroidVars_multi(n, k, ptx, pty, s_centroidx, s_centroidy, s_thrd_x, s_thrd_y, s_thrd_k, &index, &my_point_num, &sumx, &sumy, &count);
  #endif
            }
            reduceThreads_multi(k, &sumx, &sumy, &count, s_thrd_reduce_x, s_thrd_reduce_y, s_thrd_k);
            reduceBlocks_and_storeNewCentroids(k,numIter,gridDim.x,storageIndex,&sumx,&sumy,&count,d_sumx,d_sumy,d_count,d_syncArr, d_centroidx, d_centroidy);//, d_reducedCounts, d_timeVar);//,d_dbgIter,*max_iter);
        }

        if(((++numIter) < (*max_iter)) && repeat)
        {
            // ************ 1st, 3rd and so on all odd iterations ************

            init_centroids(k, s_centroidx, s_centroidy, d_centroidx, d_centroidy, &sumx, &sumy, &count);
            // We already have points loaded for last point_cycle
  #if POINTS_PER_THREAD == 1
            reduceValues_and_setCentroidVars_multi(n, k, &ptx, &pty, s_centroidx, s_centroidy, s_thrd_x, s_thrd_y, s_thrd_k, &index, &my_point_num, &sumx, &sumy, &count);
  #else
            reduceValues_and_setCentroidVars_multi(n, k, ptx, pty, s_centroidx, s_centroidy, s_thrd_x, s_thrd_y, s_thrd_k, &index, &my_point_num, &sumx, &sumy, &count);
  #endif
            for( curr_point_cycle=total_point_cycles-2; curr_point_cycle >= 0; curr_point_cycle--)
            {
  #if POINTS_PER_THREAD == 1
                init_data(curr_point_cycle, total_point_cycles, n, &ptx, &pty, &index, d_dataptx, d_datapty, s_thrd_x, s_thrd_y);
                reduceValues_and_setCentroidVars_multi(n, k, &ptx, &pty, s_centroidx, s_centroidy, s_thrd_x, s_thrd_y, s_thrd_k, &index, &my_point_num, &sumx, &sumy, &count);
  #else
                my_point_num = init_data(curr_point_cycle, total_point_cycles, n, ptx, pty, &index, d_dataptx, d_datapty, s_thrd_x, s_thrd_y);
                reduceValues_and_setCentroidVars_multi(n, k, ptx, pty, s_centroidx, s_centroidy, s_thrd_x, s_thrd_y, s_thrd_k, &index, &my_point_num, &sumx, &sumy, &count);
  #endif
            }
            reduceThreads_multi(k, &sumx, &sumy, &count, s_thrd_reduce_x, s_thrd_reduce_y, s_thrd_k);
            reduceBlocks_and_storeNewCentroids(k,numIter,gridDim.x,storageIndex,&sumx,&sumy,&count,d_sumx,d_sumy,d_count,d_syncArr, d_centroidx, d_centroidy);//, d_reducedCounts, d_timeVar);//,d_dbgIter,*max_iter);

        }

    }

    // Iterations are over and final centroids have been found. Now set the cluster no. in d_clusterno array in device memory.
    {
        load_centroids_in_shared(k, s_centroidx, s_centroidy, d_centroidx, d_centroidy);
        __syncthreads();

        // Store nearest in global for last loaded point.
        // Find closest centroid if this thread holds a data-point
  #if POINTS_PER_THREAD == 1
        store_nearest_in_global(n, k, ptx, pty, index, s_centroidx, s_centroidy, d_clusterno);
  #else
        store_nearest_in_global(k, ptx, pty, index, s_centroidx, s_centroidy, d_clusterno, my_point_num);
  #endif

        // Assuming last loaded points were of first point cycle
        int nextCycle=1;
        int goTill=total_point_cycles;

        if(curr_point_cycle==total_point_cycles)
        {
            // Last loaded points were of last point cycle
            nextCycle=0;
            goTill=total_point_cycles-1;
        }

        for( int curr_point_cycle=nextCycle; curr_point_cycle < goTill; curr_point_cycle++)
        {
  #if POINTS_PER_THREAD == 1
            init_data(curr_point_cycle, total_point_cycles, n, &ptx, &pty, &index, d_dataptx, d_datapty, s_thrd_x, s_thrd_y);
            // Find closest centroid if this thread holds a data-point
            store_nearest_in_global(n, k, ptx, pty, index, s_centroidx, s_centroidy, d_clusterno);
  #else
            my_point_num=init_data(curr_point_cycle, total_point_cycles, n, ptx, pty, &index, d_dataptx, d_datapty, s_thrd_x, s_thrd_y);
            // Find closest centroid if this thread holds a data-point
            store_nearest_in_global(k, ptx, pty, index, s_centroidx, s_centroidy, d_clusterno, my_point_num);
  #endif
        }

    }

    if(blockIdx.x == 0 && threadIdx.x == 0)
        *max_iter = numIter;
}


// Right now this code only works when
// 1. Number of threads >= k
// 2. Number of points can be > threads*blocks, i.e each thread may have to process more than one point till the shared mem allows.
// 3. All the centroids can be accomodated in shared mem. at once.
__global__ void __launch_bounds__(BLOCKDIM, MIN_NUM_BLOCKS_PER_MULTIPROC)
cluster1_load_optimized_tree(int n, int k,int *const RESTRICT max_iter, const float *const RESTRICT d_dataptx, const float *const RESTRICT d_datapty,
                       float *const RESTRICT d_centroidx, float *const RESTRICT d_centroidy, int *const RESTRICT d_syncArr,
                       float *const RESTRICT d_sumx, float *const RESTRICT d_sumy, int *const RESTRICT d_count,int *const RESTRICT d_clusterno)
{
    // shared memory
    // Stores x-coord of all clusters followed by y-coord of all clusters i.e. x and y coord of a cluster are separated by k.
    extern __shared__ float s_centroid[];

    // No point of going for short int as each bank in shared mem. is 32-bit wide and only one thread
    // can access one bank at a time, so no. of accesses wont be reduced
    __shared__ signed int s_thrd_k[BLOCKDIM];                                           // Store the nearest cluster of point checked by this thrd.

    __shared__ float s_thrd[BLOCKDIM<<1];                                               // Store the x-coord of data point processed, for all thrds followed by y-coord.

    float *const s_centroidx = s_centroid;                                                                  // x-coord of 1st cluster's centroid
    float *const s_centroidy = s_centroid+k;                                                                // y-coord of 1st cluster's centroid

    float *const s_thrd_x = s_thrd;                                                                         // x-coord of points checked by thrds
    float *const s_thrd_y = s_thrd+(1<<LOG_BLOCKDIM);                                       // y-coord of points checked by thrds

  #if POINTS_PER_THREAD == 1
    float ptx,pty;
  #else
    float ptx[POINTS_PER_THREAD],pty[POINTS_PER_THREAD];
  #endif

    const unsigned int total_point_cycles=(n-1)/((POINTS_PER_THREAD<<LOG_BLOCKDIM)*gridDim.x) + 1;

    bool repeat=true;
    int numIter=0;

    int count=0;
    const unsigned int storageIndex = (blockIdx.x>>1)*k + threadIdx.x;                            // The sumx, sumy and count value will be stored at this index in global mem.

    int curr_point_cycle=0;
    unsigned int index;

    float sumx=0;
    float sumy =0;
  #if POINTS_PER_THREAD == 1
    int my_point_num = init_data(curr_point_cycle, total_point_cycles, n, &ptx, &pty, &index, d_dataptx, d_datapty);
  #else
    int my_point_num = init_data(curr_point_cycle, total_point_cycles, n, ptx, pty, &index, d_dataptx, d_datapty);
  #endif

    for ( numIter = 0; repeat && (numIter < *max_iter); ++numIter )
    {
        {
            // ************ 0th, 2nd and so on all even iterations ************
            init_centroids(k, s_centroidx, s_centroidy, d_centroidx, d_centroidy, &sumx, &sumy, &count);
            // We already have points loaded for first point_cycle
  #if POINTS_PER_THREAD == 1
            reduceThreads_and_setCentroidVars_tree(n, k, &ptx, &pty, s_centroidx, s_centroidy, s_thrd_x, s_thrd_y, s_thrd_k, &index, &my_point_num, &sumx, &sumy, &count);
  #else
            reduceThreads_and_setCentroidVars_tree(n, k, ptx, pty, s_centroidx, s_centroidy, s_thrd_x, s_thrd_y, s_thrd_k, &index, &my_point_num, &sumx, &sumy, &count);
  #endif


            for( curr_point_cycle=1; curr_point_cycle < total_point_cycles; curr_point_cycle++)
            {
  #if POINTS_PER_THREAD == 1
                init_data(curr_point_cycle, total_point_cycles, n, &ptx, &pty, &index, d_dataptx, d_datapty);
                reduceThreads_and_setCentroidVars_tree(n, k, &ptx, &pty, s_centroidx, s_centroidy, s_thrd_x, s_thrd_y, s_thrd_k, &index, &my_point_num, &sumx, &sumy, &count);
  #else
                my_point_num = init_data(curr_point_cycle, total_point_cycles, n, ptx, pty, &index, d_dataptx, d_datapty);
                reduceThreads_and_setCentroidVars_tree(n, k, ptx, pty, s_centroidx, s_centroidy, s_thrd_x, s_thrd_y, s_thrd_k, &index, &my_point_num, &sumx, &sumy, &count);
  #endif
            }
            reduceBlocks_and_storeNewCentroids(k,numIter,gridDim.x,storageIndex,&sumx,&sumy,&count,d_sumx,d_sumy,d_count,d_syncArr, d_centroidx, d_centroidy);//, d_reducedCounts, d_timeVar);//,d_dbgIter,*max_iter);
        }

        if(((++numIter) < (*max_iter)) && repeat)
        {
            // ************ 1st, 3rd and so on all odd iterations ************

            init_centroids(k, s_centroidx, s_centroidy, d_centroidx, d_centroidy, &sumx, &sumy, &count);
            // We already have points loaded for last point_cycle
  #if POINTS_PER_THREAD == 1
            reduceThreads_and_setCentroidVars_tree(n, k, &ptx, &pty, s_centroidx, s_centroidy, s_thrd_x, s_thrd_y, s_thrd_k, &index, &my_point_num, &sumx, &sumy, &count);
  #else
            reduceThreads_and_setCentroidVars_tree(n, k, ptx, pty, s_centroidx, s_centroidy, s_thrd_x, s_thrd_y, s_thrd_k, &index, &my_point_num, &sumx, &sumy, &count);
  #endif
            for( curr_point_cycle=total_point_cycles-2; curr_point_cycle >= 0; curr_point_cycle--)
            {
  #if POINTS_PER_THREAD == 1
                init_data(curr_point_cycle, total_point_cycles, n, &ptx, &pty, &index, d_dataptx, d_datapty);
                reduceThreads_and_setCentroidVars_tree(n, k, &ptx, &pty, s_centroidx, s_centroidy, s_thrd_x, s_thrd_y, s_thrd_k, &index, &my_point_num, &sumx, &sumy, &count);
  #else
                my_point_num = init_data(curr_point_cycle, total_point_cycles, n, ptx, pty, &index, d_dataptx, d_datapty);
                reduceThreads_and_setCentroidVars_tree(n, k, ptx, pty, s_centroidx, s_centroidy, s_thrd_x, s_thrd_y, s_thrd_k, &index, &my_point_num, &sumx, &sumy, &count);
  #endif
            }

            reduceBlocks_and_storeNewCentroids(k,numIter,gridDim.x,storageIndex,&sumx,&sumy,&count,d_sumx,d_sumy,d_count,d_syncArr, d_centroidx, d_centroidy);//, d_reducedCounts, d_timeVar);//,d_dbgIter,*max_iter);

        }

    }

    // Iterations are over and final centroids have been found. Now set the cluster no. in d_clusterno array in device memory.
    {
        load_centroids_in_shared(k, s_centroidx, s_centroidy, d_centroidx, d_centroidy);
        __syncthreads();

        // Store nearest in global for last loaded point.
        // Find closest centroid if this thread holds a data-point
  #if POINTS_PER_THREAD == 1
        store_nearest_in_global(n, k, ptx, pty, index, s_centroidx, s_centroidy, d_clusterno);
  #else
        store_nearest_in_global(k, ptx, pty, index, s_centroidx, s_centroidy, d_clusterno, my_point_num);
  #endif

        // Assuming last loaded points were of first point cycle
        int nextCycle=1;
        int goTill=total_point_cycles;

        if(curr_point_cycle==total_point_cycles)
        {
            // Last loaded points were of last point cycle
            nextCycle=0;
            goTill=total_point_cycles-1;
        }

        for( int curr_point_cycle=nextCycle; curr_point_cycle < goTill; curr_point_cycle++)
        {
  #if POINTS_PER_THREAD == 1
            init_data(curr_point_cycle, total_point_cycles, n, &ptx, &pty, &index, d_dataptx, d_datapty);
            // Find closest centroid if this thread holds a data-point
            store_nearest_in_global(n, k, ptx, pty, index, s_centroidx, s_centroidy, d_clusterno);
  #else
            my_point_num=init_data(curr_point_cycle, total_point_cycles, n, ptx, pty, &index, d_dataptx, d_datapty);
            // Find closest centroid if this thread holds a data-point
            store_nearest_in_global(k, ptx, pty, index, s_centroidx, s_centroidy, d_clusterno, my_point_num);
  #endif
        }

    }

    if(blockIdx.x == 0 && threadIdx.x == 0)
        *max_iter = numIter;
}

