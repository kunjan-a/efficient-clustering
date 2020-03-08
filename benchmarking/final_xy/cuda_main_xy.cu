#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <cutil.h>
#include <settings_xy.h>
//#include <cuda_kernel_xy.cu>
//#include <cuda_kernel_xy_tree.cu>
//#include <cuda_kernel_xy_multi.cu>
#include <cuda_kernel_xy_combined.cu>

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
__host__ void mean_by_CUDA( int n, int max_iter, int k, float threshold, float *h_dataptx, float *h_datapty);

////////////////////////////////////////////////////////////////////////////////
//! Global Memory size required in bytes rounded to nearest multiple of mem. segment size.
//! The rules followed by 1.2 and 1.3 CUDA cards are:
//! The following protocol is used to determine the memory transactions
//! necessary to service all threads in a half-warp:
//!
//! Find the memory segment that contains the address requested by the lowest
//! numbered active thread. The segment size depends on the size of the words
//! accessed by the threads:
//! 32 bytes for 1-byte words,
//! 64 bytes for 2-byte words,
//! 128 bytes for 4-, 8- and 16-byte words.
//!
//! Find all other active threads whose requested address lies in the same segment.
//! Reduce the transaction size, if possible:
//! If the transaction size is 128 bytes and only the lower or upper half is used,
//! reduce the transaction size to 64 bytes;
//! If the transaction size is 64 bytes (originally or after reduction from 128
//! bytes) and only the lower or upper half is used, reduce the transaction size
//! to 32 bytes.
//!
//! Carry out the transaction and mark the serviced threads as inactive.
//!
//! Repeat until all threads in the half-warp are serviced.
//! @param n size required
////////////////////////////////////////////////////////////////////////////////
template<class T>
__device__ __host__ unsigned int get_gmem_length(int n)
{
    int size=sizeof(T);
    int logSegLength=7;
    if(size==1)
    {
        logSegLength=5;
    }
    else
    {
        if(size==2)
            logSegLength=6;
    }
    // gives bytes rounded to nearest multiple of segment size i.e. ceil(mem_required/segmenSize)*segmentSize
    return ((((n*size-1)>>logSegLength) + 1) <<logSegLength);
}

////////////////////////////////////////////////////////////////////
//! We store the points in d arrays each of length (n + n%128), to ensure that all global memory accesses are aligned to transaction size.
//!
//! If points more than device memory then we can issue the kmeans code multiple times for each iteration and
//! finally add up the individual counts and sums from each such invocation to get new centroids. We have not coded for such case right now.
//!
//! Another way could be to store the points in following manner:
//! Store 1st co-ordinate of 'p' points, followed by their 2nd co-ordinate, followed by their 3rd co-ordinate and so on,
//! so that when first co-ordinate is read the 2nd co-ordinate comes into cache (assuming there is cache prefetching).
//! Here the number 'p' can be 'w', or 't', or t*b',
//! where b' is the number of blocks that can run simultaneously i.e. m*MIN_NUM_BLOCKS_PER_MULTIPROCESSOR.
///////////////////////////////////////////////////////////////////
__host__ unsigned int load_points_in_device(const unsigned int n, float* &d_dataptx, float* &d_datapty,float* &h_dataptx, float* &h_datapty)
{

    float *d_data;
    unsigned int memReq=get_gmem_length<float>(n);
    // TODO: Check if n is too large for global mempry to store at once.

    // Allcate device memory for data points and centroids. Store x-coord of all points followed by y-coord of all points
    // followed by x-coord of centroids followed by y-coord of centroids

    printf( "device memory required for storing %d points is: %d\n", n,(memReq<<1));
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_data, (memReq<<1)));

    unsigned int lenPerDim_d_datapt = memReq/sizeof(float);
    d_dataptx=d_data;                                                                          // all coord of datapoints
    d_datapty=d_dataptx + lenPerDim_d_datapt;
    printf("d_data:%u, d_dataptx:%u, d_datapty: %u \n",d_data,d_dataptx,d_datapty);

    CUDA_SAFE_CALL( cudaMemcpy( d_dataptx, h_dataptx, n*sizeof(float), cudaMemcpyHostToDevice) );           //copy host memory to device
    CUDA_SAFE_CALL( cudaMemcpy( d_datapty, h_datapty, n*sizeof(float), cudaMemcpyHostToDevice) );           //copy host memory to device

    return lenPerDim_d_datapt;
}


////////////////////////////////////////////////////////////////////
//! We select the first k points as the initial centroids.
//! We might as well take the points at k random positions but this decision would not effect
//! the difference b/w performance of openmp and CUDA as both will run for equal number of iterations for the same choice of k initial centroids.
//! In fact because of this instead of terminating kmeans when the cluster centres stop changing in subsequent iterations,
//! we let it run for a fixed number of iterations for both CUDA and openmp as the amount of work done in each iteration is same.
//!
//! We store the centroids in the device memory again in d arrays each of size (k + k%128)
//!
//! We could store them in constant memory too, provided it fits in its small size, to make use of constant cache.
//! But we invoke the clustering kernel only once and CUDA restricts changing contents of constant memory from inside a kernel.
//! Instead if we call the kernel separaely for each iteration then we can replace the old centroid values in the constant memory
//! with new centroids found at the end of the iteration.
///////////////////////////////////////////////////////////////////
__host__ unsigned int load_centroids_in_device(const unsigned int k, float * &d_centroidx, float * &d_centroidy, float * &d_dataptx, float * &d_datapty)
{

    float *d_data;
    unsigned int memReq=get_gmem_length<float>(k);

    printf( "device memory required for storing %d centroids is: %d\n", k,(memReq<<1));
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_data, (memReq<<1)));

    unsigned int lenPerDim_d_centroid=memReq/sizeof(float);
    d_centroidx=d_data;                                                                        // x-coord of centroids
    d_centroidy=d_data+lenPerDim_d_centroid;                                                            // y-coord of centroids
    printf("d_data:%u, d_centroidx: %u, d_centroidy:%u \n",d_data,d_centroidx,d_centroidy);

    CUDA_SAFE_CALL( cudaMemcpy( d_centroidx, d_dataptx, sizeof(float)*k, cudaMemcpyDeviceToDevice) );   //Initialize the first k data points as the k centroids.
    CUDA_SAFE_CALL( cudaMemcpy( d_centroidy, d_datapty, sizeof(float)*k, cudaMemcpyDeviceToDevice) );

    return lenPerDim_d_centroid;
}


////////////////////////////////////////////////////////////////////
//! The external reduction between blocks requires synchronisation among blocks.
//! We store an array of synchronisation vars which store the value of the iteration for which the block has finished reduction.
//! The initial value is -1 for all the blocks.
//! @return Base address of the synchronisation array which was created in the device memory
///////////////////////////////////////////////////////////////////
__host__ int* create_block_synchronisn_array(const unsigned int num_blocks)
{
    int *d_data1;
    const int syncArrLength = num_blocks;     //not using get_barrier_synch_array_length(num_blocks) as we dont need threadFenceReduction any more
    printf( "length of barrier synchronization array: %d\n", syncArrLength);
    unsigned int memReq=get_gmem_length<int>(num_blocks);
    printf( "device memory required for storing %d synchronizer vars is: %d\n", num_blocks,memReq);
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_data1, memReq));
    int *d_syncArr=d_data1;
    printf("d_data1:%u, d_syncArr: %u \n",d_data1,d_syncArr);

    cudaThreadSynchronize();                                                                            //cudaMalloc is asynchronous. So it may return back to host even before the above allocations were over.

// << grid size i.e. no. of blocks, block size i.e. no. of threads,  no. of bytes in shared mem. that is dynamically allocated per block for this call in addition to the statically allocated mem. >>
    initialize <<< dim3(1 + ((syncArrLength - 1) >> LOG_BLOCKDIM) ), dim3(BLOCKDIM), 0 >>> (syncArrLength,d_syncArr, -1);
    cudaThreadSynchronize();
    CUT_CHECK_ERROR("Kernel execution failed");                                                         // check if kernel execution generated an error
    return d_syncArr;
}


////////////////////////////////////////////////////////////////////
//! Allocate space for storing the sum and count for each cluster for every block.
//! For each block only first k threads store the reduced result in device memory.
//! Also each block reuses the device memory space of the block it reduced with.
//!
//! The sum and count values are initialised as 0 for all clusters of all blocks.
//! @return Base address of the count array which was created in the device memory.
///////////////////////////////////////////////////////////////////
__host__ int* allocate_initialise_reduction_result(const unsigned int num_blocks, const unsigned int k, float * &d_sumx, float * &d_sumy )
{
// For each block only first k threads store the reduced result in device memory.
// Also each block reuses the device memory space of the block it reduced with.
    int reduction_array_len = k*((num_blocks+1)>>1);

    float *d_data;
    unsigned int memReq=get_gmem_length<float>(reduction_array_len);
    printf( "device memory required for storing reduction result for centroid co-ordinates of k*(block_num/2) clusters (%d) is: %d\n", reduction_array_len,(memReq<<1));
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_data, (memReq<<1)));

    unsigned int lenPerDim_d_sum=memReq/sizeof(float);
    d_sumx=d_data;                                                                           //Stores the per coordinate sum of all the points in each cluster for each block
    d_sumy=d_sumx+lenPerDim_d_sum;
    printf("d_data:%u, d_sumx: %u, d_sumy:%u \n",d_data,d_sumx,d_sumy);


    int* d_data1;
    memReq=get_gmem_length<int>(reduction_array_len);
    printf( "device memory required for storing reduction result of count of points in k*(block_num/2) clusters (%d ) is: %d\n", reduction_array_len,memReq);
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_data1, memReq));
    int *const d_count=d_data1;                                                                           //Stores the no. of points in each cluster for each block
    printf("d_data1:%u, d_count: %u \n",d_data1,d_count);

    // Initialize the sum variables to 0. Actually we should have initialised only reduction_array_len number of values for each attribute,
    // but since that would have meant calling initialise kernel 'number  of attribue' times, we set to 0 even the extra address
    // that we had allocated for each dimension to align with transactino size.
    initialize <<< dim3(1 + ((reduction_array_len - 1) >> LOG_BLOCKDIM)), dim3(BLOCKDIM), 0 >>> (reduction_array_len,d_sumx, 0);
    cudaThreadSynchronize();
    CUT_CHECK_ERROR("Kernel execution failed");                      // check if kernel execution generated an error
    initialize <<< dim3(1 + ((reduction_array_len - 1) >> LOG_BLOCKDIM)), dim3(BLOCKDIM), 0 >>> (reduction_array_len,d_sumy, 0);
    cudaThreadSynchronize();
    CUT_CHECK_ERROR("Kernel execution failed");                      // check if kernel execution generated an error
    initialize <<< dim3(1 + ((reduction_array_len - 1) >> LOG_BLOCKDIM)), dim3(BLOCKDIM), 0 >>> (reduction_array_len,d_count, 0);
    cudaThreadSynchronize();
    CUT_CHECK_ERROR("Kernel execution failed");                      // check if kernel execution generated an error

    return d_count;
}


////////////////////////////////////////////////////////////////////
//! Allocate space for storing the final cluster number every point belongs to.
//! It also initialises cluster no. for every point as 0 i.e. first cluster
//! @return Base address of the array which was created in the device memory.
///////////////////////////////////////////////////////////////////
__host__ int* allocate_initialise_clusterno_array(const unsigned int n)
{
    int * d_data1;
    unsigned int memReq=get_gmem_length<int>(n);
    printf( "device memory required for storing the cluster nos of %d data points is: %d\n", n,memReq);
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_data1, memReq ));
    int *const d_clusterno=d_data1;                                                                       //stores the cluster no. to which the data-point belongs
    printf("d_data1:%u, d_clusterno: %u \n",d_data1,d_clusterno);

    cudaThreadSynchronize();                                                                            //cudaMalloc is asynchronous. So it may return back to host even before the above allocations were over.

// << grid size i.e. no. of blocks, block size i.e. no. of threads,  no. of bytes in shared mem. that is dynamically allocated per block for this call in addition to the statically allocated mem. >>
    initialize <<< dim3(1 + ((n - 1) >> LOG_BLOCKDIM) ), dim3(BLOCKDIM), 0 >>> (n,d_clusterno, 0);
    cudaThreadSynchronize();
    CUT_CHECK_ERROR("Kernel execution failed");                                                         // check if kernel execution generated an error
    return d_clusterno;
}


////////////////////////////////////////////////////////////////////////////////
//! Program main
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv)
{
    int n,max_iter,k;
    float threshold;

    float *h_dataptx=NULL;
    float *h_datapty=NULL;

    try
    {
        readInput( n, max_iter, k, threshold, h_dataptx, h_datapty); //reads from a file nmaed by macro INPUTFILE in the pwd

        CUT_DEVICE_INIT(argc, argv); //Checks if system has cuda devices and reads cmdline arg for cuda device to be used (by default 0). If the device is CUDA enabled then prints the device name else prints error and exits code.

        mean_by_CUDA(n, max_iter, k, threshold, h_dataptx, h_datapty);
    }
    catch (...)
    {
        if (h_dataptx!=NULL)
            free(h_dataptx);
        if (h_datapty!=NULL)
            free(h_datapty);

        h_dataptx=NULL;
        h_datapty=NULL;
    }

        if (h_dataptx!=NULL)
            free(h_dataptx);
        if (h_datapty!=NULL)
            free(h_datapty);

        h_dataptx=NULL;
        h_datapty=NULL;

   // CUT_EXIT(argc, argv); //Prompt for press enter to exit and then flush error and out stream. To avoid it pass "--noprompt" on cmdline.
}

#if APPROACH == 1
////////////////////////////////////////////////////////////////////////////////
//!Compare kmeans on CPU and GPU
//! @param n number of points
//! @param h_max_iter number of iterations
//! @param k number of clusters
////////////////////////////////////////////////////////////////////////////////
__host__ void mean_by_CUDA( int n, int h_max_iter, int k, float threshold, float *h_dataptx, float *h_datapty)
{
    unsigned int timer = 0;
    CUT_SAFE_CALL( cutCreateTimer( &timer));
    CUT_SAFE_CALL( cutStartTimer( timer));

    printf("Shared mem. req. on each multiprocessor is: \n MIN_NUM_BLOCKS_PER_MULTIPROC * ( BLOCKDIM *POINTS_PER_THREAD*12  +  k*8)\n");
    printf("%d * ( %d*%d*12 + %d*8 ) = %d bytes\n", MIN_NUM_BLOCKS_PER_MULTIPROC, BLOCKDIM, POINTS_PER_THREAD, k, MIN_NUM_BLOCKS_PER_MULTIPROC * ( BLOCKDIM * POINTS_PER_THREAD * 12  +  k*8));

    bool load_points_in_cycles=false;
    int temp=get_num_blocks(n);
//    temp=1;
    if ((MIN_NUM_BLOCKS_PER_MULTIPROC * NUM_MULTIPROC) < temp)                                             //We cannot guarantee that all blocks will be started together, so this may lead to deadlock.
    {
      printf("\n%d blocks are needed whereas we can only allow a maximum of %d blocks to ensure that all blocks are started together.",temp,MIN_NUM_BLOCKS_PER_MULTIPROC*NUM_MULTIPROC);
      printf("\n We will load points in cycles to do work in %d blocks only.",MIN_NUM_BLOCKS_PER_MULTIPROC*NUM_MULTIPROC);
      printf("\nPlease increase number of threads or points per thread if you want to avoid this.\n");
      load_points_in_cycles=true;
      temp=MIN_NUM_BLOCKS_PER_MULTIPROC*NUM_MULTIPROC;
    }
    const int num_blocks=temp;

    printf( "no. of blocks: %d\n", num_blocks);
    //TODO: Right now there is no check if no. of blocks is more than the limit of 64k.
    //In that case we should be handling more than one point per thread by invoking the local reduction step multiple times.

    float *d_dataptx=NULL;                                                                          // x-coord of datapoints
    float *d_datapty=NULL;                                                                          // y-coord of datapoints
    unsigned int lenPerDim_d_datapt = load_points_in_device(n, d_dataptx, d_datapty,h_dataptx, h_datapty);

    //cudaMemcpy is asynchronous. Required to make sure above memcopy is complete before we copy some of those values as centroids.
    cudaThreadSynchronize();

    float *d_centroidx=NULL;                                                                        // x-coord of centroids
    float *d_centroidy=NULL;                                                                        // y-coord of centroids
    unsigned int lenPerDim_d_centroid = load_centroids_in_device(k, d_centroidx, d_centroidy, d_dataptx, d_datapty);


    int *const d_syncArr=create_block_synchronisn_array(num_blocks);


    //Allocate device memory for result
    float *d_sumx=NULL;                                                                           //Stores the sum of x-coordinates of all the points in each cluster for each block
    float *d_sumy=NULL;                                                                           //Stores the sum of y-coordinates of all the points in each cluster for each block
    int *const d_count=allocate_initialise_reduction_result(num_blocks, k, d_sumx, d_sumy);      //Stores the no. of points in each cluster for each block


    int *const d_clusterno=allocate_initialise_clusterno_array(n);                               //stores the cluster no. to which the data-point belongs


    int *d_max_iter;
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_max_iter, sizeof(int) ));
    cudaThreadSynchronize();                                                                     //cudaMalloc is asynchronous. So it may return back to host even before the above allocations were over.
    CUDA_SAFE_CALL( cudaMemcpy( d_max_iter, &h_max_iter, sizeof(int), cudaMemcpyHostToDevice) );


    CUT_SAFE_CALL( cutStopTimer( timer));
    printf( "GPU version mem. allocation and copy time: %f (ms)\n", cutGetTimerValue( timer));
    CUT_SAFE_CALL( cutResetTimer( timer));


#if PROFILE_TIME == 1
    cudaEvent_t start_cluster,stop_cluster;
    float time,time_cluster;
    time=time_cluster=0;

    cudaEventCreate(&start_cluster);
    cudaEventCreate(&stop_cluster);
#endif


    CUT_SAFE_CALL( cutStartTimer( timer));

#if PROFILE_TIME == 1
    cudaEventRecord( start_cluster, 0 );
#endif
// << grid size i.e. no. of blocks, block size i.e. no. of threads,  no. of bytes in shared mem. that is dynamically allocated per block for this call in addition to the statically allocated mem. >>
    if(load_points_in_cycles)//load_points_in_cycles)
      //cluster1_load_optimized <<< dim3(num_blocks), dim3(BLOCKDIM), k*(sizeof(float)<<1) >>> (n, k, d_max_iter, d_dataptx,d_datapty,d_centroidx,d_centroidy,d_syncArr,d_sumx, d_sumy, d_count,d_clusterno);//, d_reducedCounts, d_timeVar);//,d_dbgIter);
      cluster1 <<< dim3(num_blocks), dim3(BLOCKDIM), k*(sizeof(float)<<1) >>> (n, k, d_max_iter, d_dataptx,d_datapty,d_centroidx,d_centroidy,d_syncArr,d_sumx, d_sumy, d_count,d_clusterno);//, d_reducedCounts, d_timeVar);//,d_dbgIter);
    else
      cluster <<< dim3(num_blocks), dim3(BLOCKDIM), k*(sizeof(float)<<1) >>> (n, k, d_max_iter, d_dataptx,d_datapty,d_centroidx,d_centroidy,d_syncArr,d_sumx, d_sumy, d_count,d_clusterno);//, d_reducedCounts, d_timeVar);//,d_dbgIter);
//  testBRDCST <<< dim3(num_blocks), dim3(BLOCKDIM), k*(sizeof(float)) >>> (n, k, d_dataptx);

#if PROFILE_TIME == 1
    cudaEventRecord( stop_cluster, 0 );
    cudaEventSynchronize(stop_cluster);
    cudaEventElapsedTime( &time, start_cluster, stop_cluster );
    time_cluster+=time;
    time=0;
#else
    cudaThreadSynchronize();
#endif
    // check if kernel execution generated an error
    CUT_CHECK_ERROR("Kernel execution failed");

    if(load_points_in_cycles)
    {
      //findCluster <<<  dim3(num_blocks), dim3(BLOCKDIM), k*(sizeof(float)<<1) >>> (n, k, d_dataptx, d_datapty, d_centroidx, d_centroidy, d_clusterno);
      cudaThreadSynchronize();
      // check if kernel execution generated an error
      CUT_CHECK_ERROR("Kernel execution failed");
    }
    CUT_SAFE_CALL( cutStopTimer( timer));
    printf( "GPU version processing time: %f (ms)\n", cutGetTimerValue( timer));

#if PROFILE_TIME == 1
    printf("Total time in kernel: cluster = %f(ms)\n",time_cluster);

    cudaEventDestroy(start_cluster);
    cudaEventDestroy(stop_cluster);
#endif
    CUT_SAFE_CALL( cutResetTimer( timer));
    CUT_SAFE_CALL( cutStartTimer( timer));

    // check if kernel execution generated an error
    CUT_CHECK_ERROR("Kernel execution failed");

    float *h_centroidx,*h_centroidy;
    h_centroidx=NULL;
    h_centroidy=NULL;
    int *h_clusterno=NULL;
    try
    {
        h_centroidx = (float*) malloc(sizeof(float)*k);
        h_centroidy = (float*) malloc(sizeof(float)*k);

        CUDA_SAFE_CALL( cudaMemcpy( h_centroidx, d_centroidx, sizeof(float)*k, cudaMemcpyDeviceToHost) );
        CUDA_SAFE_CALL( cudaMemcpy( h_centroidy, d_centroidy, sizeof(float)*k, cudaMemcpyDeviceToHost) );

        h_clusterno = (int*) malloc(sizeof(int)*n);

        CUDA_SAFE_CALL( cudaMemcpy( h_clusterno, d_clusterno, sizeof(int)*n, cudaMemcpyDeviceToHost) );
        CUDA_SAFE_CALL( cudaMemcpy( &h_max_iter, d_max_iter, sizeof(int), cudaMemcpyDeviceToHost) );


        CUDA_SAFE_CALL(cudaFree(d_dataptx));
        CUDA_SAFE_CALL(cudaFree(d_centroidx));
        CUDA_SAFE_CALL(cudaFree(d_clusterno));
        CUDA_SAFE_CALL(cudaFree(d_sumx) );
        CUDA_SAFE_CALL(cudaFree(d_count) );
        CUDA_SAFE_CALL(cudaFree(d_syncArr) );

        CUT_SAFE_CALL( cutStopTimer( timer));
        printf( "GPU version copying results back and deallocation time: %f (ms)\n", cutGetTimerValue( timer));
        CUT_SAFE_CALL( cutDeleteTimer( timer));

        printf( "GPU version number of iterations: %d\n", h_max_iter);

        writeOutput(n, k, h_dataptx, h_datapty, h_clusterno, h_centroidx, h_centroidy);
    }
    catch(...)
    {
        if(h_centroidx!=NULL)
            free(h_centroidx);
        if(h_centroidy!=NULL)
            free(h_centroidy);
        if(h_clusterno!=NULL)
            free(h_clusterno);

        h_centroidx=NULL;
        h_centroidy=NULL;
        h_clusterno=NULL;
    }

    if(h_centroidx!=NULL)
        free(h_centroidx);
    if(h_centroidy!=NULL)
        free(h_centroidy);
    if(h_clusterno!=NULL)
        free(h_clusterno);

    h_centroidx=NULL;
    h_centroidy=NULL;
    h_clusterno=NULL;
}
#endif

#if APPROACH == 155
////////////////////////////////////////////////////////////////////////////////
//!Compare kmeans on CPU and GPU
//! @param n number of points
//! @param h_max_iter number of iterations
//! @param k number of clusters
////////////////////////////////////////////////////////////////////////////////
__host__ void mean_by_CUDA( int n, int h_max_iter, int k, float threshold, float *h_dataptx, float *h_datapty)
{
    unsigned int timer = 0;
    CUT_SAFE_CALL( cutCreateTimer( &timer));
    CUT_SAFE_CALL( cutStartTimer( timer));

    printf("Shared mem. req. on each multiprocessor is: \n MIN_NUM_BLOCKS_PER_MULTIPROC * ( BLOCKDIM *POINTS_PER_THREAD*12  +  k*8)\n");
    printf("%d * ( %d*%d*12 + %d*8 ) = %d bytes\n", MIN_NUM_BLOCKS_PER_MULTIPROC, BLOCKDIM, POINTS_PER_THREAD, k, MIN_NUM_BLOCKS_PER_MULTIPROC * ( BLOCKDIM * POINTS_PER_THREAD * 12  +  k*8));

    bool load_points_in_cycles=false;
    int temp=get_num_blocks(n);
//    temp=1;
    if ((MIN_NUM_BLOCKS_PER_MULTIPROC * NUM_MULTIPROC) < temp)                                             //We cannot guarantee that all blocks will be started together, so this may lead to deadlock.
    {
      printf("\n%d blocks are needed whereas we can only allow a maximum of %d blocks to ensure that all blocks are started together.",temp,MIN_NUM_BLOCKS_PER_MULTIPROC*NUM_MULTIPROC);
      printf("\n We will load points in cycles to do work in %d blocks only.",MIN_NUM_BLOCKS_PER_MULTIPROC*NUM_MULTIPROC);
      printf("\nPlease increase number of threads or points per thread if you want to avoid this.\n");
      load_points_in_cycles=true;
      temp=MIN_NUM_BLOCKS_PER_MULTIPROC*NUM_MULTIPROC;
    }
    const int num_blocks=temp;

    printf( "no. of blocks: %d\n", num_blocks);
    //TODO: Right now there is no check if no. of blocks is more than the limit of 64k.
    //In that case we should be handling more than one point per thread by invoking the local reduction step multiple times.

    float *d_dataptx=NULL;                                                                          // x-coord of datapoints
    float *d_datapty=NULL;                                                                          // y-coord of datapoints
    unsigned int lenPerDim_d_datapt = load_points_in_device(n, d_dataptx, d_datapty,h_dataptx, h_datapty);

    //cudaMemcpy is asynchronous. Required to make sure above memcopy is complete before we copy some of those values as centroids.
    cudaThreadSynchronize();

    float *d_centroidx=NULL;                                                                        // x-coord of centroids
    float *d_centroidy=NULL;                                                                        // y-coord of centroids
    unsigned int lenPerDim_d_centroid = load_centroids_in_device(k, d_centroidx, d_centroidy, d_dataptx, d_datapty);


    int *const d_syncArr=create_block_synchronisn_array(num_blocks);


    //Allocate device memory for result
    float *d_sumx=NULL;                                                                           //Stores the sum of x-coordinates of all the points in each cluster for each block
    float *d_sumy=NULL;                                                                           //Stores the sum of y-coordinates of all the points in each cluster for each block
    int *const d_count=allocate_initialise_reduction_result(num_blocks, k, d_sumx, d_sumy);      //Stores the no. of points in each cluster for each block


    int *const d_clusterno=allocate_initialise_clusterno_array(n);                               //stores the cluster no. to which the data-point belongs


    int *d_max_iter;
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_max_iter, sizeof(int) ));
    cudaThreadSynchronize();                                                                     //cudaMalloc is asynchronous. So it may return back to host even before the above allocations were over.
    CUDA_SAFE_CALL( cudaMemcpy( d_max_iter, &h_max_iter, sizeof(int), cudaMemcpyHostToDevice) );


    CUT_SAFE_CALL( cutStopTimer( timer));
    printf( "GPU version mem. allocation and copy time: %f (ms)\n", cutGetTimerValue( timer));
    CUT_SAFE_CALL( cutResetTimer( timer));


#if PROFILE_TIME == 1
    cudaEvent_t start_cluster,stop_cluster;
    float time,time_cluster;
    time=time_cluster=0;

    cudaEventCreate(&start_cluster);
    cudaEventCreate(&stop_cluster);
#endif


    CUT_SAFE_CALL( cutStartTimer( timer));

#if PROFILE_TIME == 1
    cudaEventRecord( start_cluster, 0 );
#endif
// << grid size i.e. no. of blocks, block size i.e. no. of threads,  no. of bytes in shared mem. that is dynamically allocated per block for this call in addition to the statically allocated mem. >>
    if(load_points_in_cycles)//load_points_in_cycles)
    {
      cluster1_load_optimized_tree <<< dim3(num_blocks), dim3(BLOCKDIM), k*(sizeof(float)<<1) >>> (n, k, d_max_iter, d_dataptx,d_datapty,d_centroidx,d_centroidy,d_syncArr,d_sumx, d_sumy, d_count,d_clusterno);//, d_reducedCounts, d_timeVar);//,d_dbgIter);
      //cluster1_tree <<< dim3(num_blocks), dim3(BLOCKDIM), k*(sizeof(float)<<1) >>> (n, k, d_max_iter, d_dataptx,d_datapty,d_centroidx,d_centroidy,d_syncArr,d_sumx, d_sumy, d_count,d_clusterno);//, d_reducedCounts, d_timeVar);//,d_dbgIter);
    }else{
      cluster_tree<<< dim3(num_blocks), dim3(BLOCKDIM), k*(sizeof(float)<<1) >>> (n, k, d_max_iter, d_dataptx,d_datapty,d_centroidx,d_centroidy,d_syncArr,d_sumx, d_sumy, d_count,d_clusterno);//, d_reducedCounts, d_timeVar);//,d_dbgIter);
    }

#if PROFILE_TIME == 1
    cudaEventRecord( stop_cluster, 0 );
    cudaEventSynchronize(stop_cluster);
    cudaEventElapsedTime( &time, start_cluster, stop_cluster );
    time_cluster+=time;
    time=0;
#else
    cudaThreadSynchronize();
#endif
    // check if kernel execution generated an error
    CUT_CHECK_ERROR("Kernel execution failed");

    if(load_points_in_cycles)
    {
      //findCluster <<<  dim3(num_blocks), dim3(BLOCKDIM), k*(sizeof(float)<<1) >>> (n, k, d_dataptx, d_datapty, d_centroidx, d_centroidy, d_clusterno);
      cudaThreadSynchronize();
      // check if kernel execution generated an error
      CUT_CHECK_ERROR("Kernel execution failed");
    }
    CUT_SAFE_CALL( cutStopTimer( timer));
    printf( "GPU version processing time: %f (ms)\n", cutGetTimerValue( timer));

#if PROFILE_TIME == 1
    printf("Total time in kernel: cluster = %f(ms)\n",time_cluster);

    cudaEventDestroy(start_cluster);
    cudaEventDestroy(stop_cluster);
#endif
    CUT_SAFE_CALL( cutResetTimer( timer));
    CUT_SAFE_CALL( cutStartTimer( timer));

    // check if kernel execution generated an error
    CUT_CHECK_ERROR("Kernel execution failed");

    float *h_centroidx,*h_centroidy;
    h_centroidx=NULL;
    h_centroidy=NULL;
    int *h_clusterno=NULL;
    float *h_sumx,*h_sumy;
    h_sumx=NULL;
    h_sumy=NULL;
    int *h_count;
    h_count=NULL;
    try
    {
        h_centroidx = (float*) malloc(sizeof(float)*k);
        h_centroidy = (float*) malloc(sizeof(float)*k);

        CUDA_SAFE_CALL( cudaMemcpy( h_centroidx, d_centroidx, sizeof(float)*k, cudaMemcpyDeviceToHost) );
        CUDA_SAFE_CALL( cudaMemcpy( h_centroidy, d_centroidy, sizeof(float)*k, cudaMemcpyDeviceToHost) );

        h_clusterno = (int*) malloc(sizeof(int)*n);

        CUDA_SAFE_CALL( cudaMemcpy( h_clusterno, d_clusterno, sizeof(int)*n, cudaMemcpyDeviceToHost) );
        CUDA_SAFE_CALL( cudaMemcpy( &h_max_iter, d_max_iter, sizeof(int), cudaMemcpyDeviceToHost) );

        h_sumx = (float *)malloc(sizeof(float)*(k*((num_blocks+1)>>1)));

        h_sumy = (float *)malloc(sizeof(float)*(k*((num_blocks+1)>>1)));

        h_count = (int *)malloc(sizeof(int)*(k*((num_blocks+1)>>1)));

        CUDA_SAFE_CALL( cudaMemcpy( h_sumx, d_sumx, sizeof(float)*(k*((num_blocks+1)>>1)), cudaMemcpyDeviceToHost) );
        CUDA_SAFE_CALL( cudaMemcpy( h_sumy, d_sumy, sizeof(float)*(k*((num_blocks+1)>>1)), cudaMemcpyDeviceToHost) );
        CUDA_SAFE_CALL( cudaMemcpy( h_count, d_count, sizeof(int)*(k*((num_blocks+1)>>1)), cudaMemcpyDeviceToHost) );


        CUDA_SAFE_CALL(cudaFree(d_dataptx));
        CUDA_SAFE_CALL(cudaFree(d_centroidx));
        CUDA_SAFE_CALL(cudaFree(d_clusterno));
        CUDA_SAFE_CALL(cudaFree(d_sumx) );
        CUDA_SAFE_CALL(cudaFree(d_count) );
        CUDA_SAFE_CALL(cudaFree(d_syncArr) );

        CUT_SAFE_CALL( cutStopTimer( timer));
        printf( "GPU version copying results back and deallocation time: %f (ms)\n", cutGetTimerValue( timer));
        CUT_SAFE_CALL( cutDeleteTimer( timer));

        printf( "GPU version number of iterations: %d\n", h_max_iter);

        writeOutput(n, k, h_dataptx, h_datapty, h_clusterno, h_centroidx, h_centroidy);

        printf("\n %d sums and count are:\n",k);
        int sum_count=0;
        for ( int i = 0; i < k; ++i){
              printf("%f %f %d\n", *(h_sumx+i), *(h_sumy+i), *(h_count+i));
              sum_count+=(*(h_count+i));
        }
        if(sum_count!=n)
          printf("\nPanga total count did not match:%d",sum_count);
    }
    catch(...)
    {
        if(h_centroidx!=NULL)
            free(h_centroidx);
        if(h_centroidy!=NULL)
            free(h_centroidy);
        if(h_clusterno!=NULL)
            free(h_clusterno);

        if(h_sumx!=NULL)
            free(h_sumx);
        if(h_sumy!=NULL)
            free(h_sumy);
        if(h_count!=NULL)
            free(h_count);

        h_sumx=NULL;
        h_sumy=NULL;
        h_count=NULL;

        h_centroidx=NULL;
        h_centroidy=NULL;
        h_clusterno=NULL;
    }

    if(h_centroidx!=NULL)
        free(h_centroidx);
    if(h_centroidy!=NULL)
        free(h_centroidy);
    if(h_clusterno!=NULL)
        free(h_clusterno);

    if(h_sumx!=NULL)
        free(h_sumx);
    if(h_sumy!=NULL)
        free(h_sumy);
    if(h_count!=NULL)
        free(h_count);

    h_sumx=NULL;
    h_sumy=NULL;
    h_count=NULL;

    h_centroidx=NULL;
    h_centroidy=NULL;
    h_clusterno=NULL;
}
#endif
