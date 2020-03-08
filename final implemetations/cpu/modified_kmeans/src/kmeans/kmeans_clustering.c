/*************************************************************************/
/**   File:         kmeans_clustering.c                                 **/
/**   Description:  Implementation of regular k-means clustering        **/
/**                 algorithm                                           **/
/*************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <omp.h>

#include "kmeans.h"

#ifndef FLT_MAX
#define FLT_MAX 3.40282347e+38
#endif

extern double wtime(void);

/*----< euclid_dist_2() >----------------------------------------------------*/
/* multi-dimensional spatial Euclid distance square */
__inline
float euclid_dist_2(float *pt1,
                    float *pt2,
                    int    numdims)
{
    int i;
    float ans=0.0;

    for (i=0; i<numdims; i++)
        ans += (pt1[i]-pt2[i]) * (pt1[i]-pt2[i]);

    return SQRT_MACRO(ans);
}

__inline
int find_nearest_point(float  *pt,          /* [nfeatures] */
                       int     nfeatures,
                       float **pts,         /* [npts][nfeatures] */
                       int     npts)
{
    int index, i;
    float max_dist=FLT_MAX;

    /* find the cluster center id with min distance to pt */
    for (i=0; i<npts; i++) {
        float dist;
        dist = euclid_dist_2(pt, pts[i], nfeatures);
        if (dist < max_dist) {
            max_dist = dist;
            index    = i;
        }
    }
    return(index);
}

#if PROFILE_TIME != 1
/*----< kmeans_clustering() >---------------------------------------------*/
void kmeans_clustering( int     is_perform_atomic,  /* in:                        */
                        float **feature,            /* in: [npoints][nfeatures]   */
                        int     nfeatures,
                        int     npoints,
                        int     nclusters,
                        float **clusters,           /* in: [nclusters][nfeatures] */
                        float   threshold,
                        int    *membership,         /* out: [npoints]             */
                        int     max_iterations)
{

    int      i, j, k, index, loop=0;
    int     *new_centers_len; /* [nclusters]: no. of points in each cluster */
    float    delta;
    float  **new_centers;     /* [nclusters][nfeatures] */
    double   timing;

    int      nthreads;
    int    **partial_new_centers_len;
    float ***partial_new_centers;
    int CACHE_ADJUSTMENT = (CACHE_BLOCK_SIZE_BYTES/(nclusters*sizeof(float)) + 1);

    nthreads = omp_get_max_threads();

    for (i=0; i<npoints; i++)
        membership[i] = -1;

    /* need to initialize new_centers_len and new_centers[0] to all 0 */
    new_centers_len = (int*) calloc(nclusters, sizeof(int));

    new_centers    = (float**) malloc(nclusters * sizeof(float*));
    new_centers[0] = (float*)  calloc(nclusters * nfeatures, sizeof(float));
    for (i=1; i<nclusters; i++)
        new_centers[i] = new_centers[i-1] + nfeatures;

    if (!is_perform_atomic)
    {
        partial_new_centers_len    = (int**) malloc(nthreads *CACHE_ADJUSTMENT*
                                     sizeof(int*));
        partial_new_centers_len[0] = (int*)  calloc(nthreads*nclusters*CACHE_ADJUSTMENT,
                                     sizeof(int));
        for (i=1; i<nthreads*CACHE_ADJUSTMENT; i++)
            partial_new_centers_len[i] = partial_new_centers_len[i-1]+nclusters;

        partial_new_centers    =(float***)malloc(nthreads *CACHE_ADJUSTMENT*
                                sizeof(float**));
        partial_new_centers[0] =(float**) malloc(nthreads*nclusters *CACHE_ADJUSTMENT*
                                sizeof(float*));
        for (i=1; i<nthreads*CACHE_ADJUSTMENT; i++)
            partial_new_centers[i] = partial_new_centers[i-1] + nclusters;
        for (i=0; i<nthreads*CACHE_ADJUSTMENT; i++)
            for (j=0; j<nclusters; j++)
                partial_new_centers[i][j] = (float*)calloc(nfeatures,
                                            sizeof(float));
    }

    if (_debug) timing = omp_get_wtime();

#if CHECK_THRESHOLD == 1
    do
    {
        delta = 0.0;

        if (is_perform_atomic)
        {
#pragma omp parallel for \
                    private(i,j,index) \
                    firstprivate(npoints,nclusters,nfeatures) \
                    shared(feature,clusters,membership,new_centers,new_centers_len) \
                    schedule(static) \
                    reduction(+:delta)
            for (i=0; i<npoints; i++)
            {
                /* find the index of nestest cluster centers */
                index = find_nearest_point(feature[i],
                                           nfeatures,
                                           clusters,
                                           nclusters);
                /* if membership changes, increase delta by 1 */
                if (membership[i] != index) delta += 1.0;

                /* assign the membership to object i */
                membership[i] = index;

                /* update new cluster centers : sum of objects located within */
#pragma omp atomic
                new_centers_len[index]++;
                for (j=0; j<nfeatures; j++)
#pragma omp atomic
                    new_centers[index][j] += feature[i][j];
            }
        }
        else
        {
#pragma omp parallel \
                    shared(feature,clusters,membership,partial_new_centers,partial_new_centers_len)
            {
                int tid = omp_get_thread_num()*CACHE_ADJUSTMENT;
#pragma omp for \
                            private(i,j,index) \
                            firstprivate(npoints,nclusters,nfeatures) \
                            schedule(static) \
                            reduction(+:delta)
                for (i=0; i<npoints; i++)
                {
                    /* find the index of nestest cluster centers */
                    index = find_nearest_point(feature[i],
                    nfeatures,
                    clusters,
                    nclusters);
                    /* if membership changes, increase delta by 1 */
                    if (membership[i] != index) delta += 1.0;

                    /* assign the membership to object i */
                    membership[i] = index;

                    /* update new cluster centers : sum of all objects located
                    		       within */
                    partial_new_centers_len[tid][index]++;
                    for (j=0; j<nfeatures; j++)
                        partial_new_centers[tid][index][j] += feature[i][j];
                }
            } /* end of #pragma omp parallel */

            /* let the main thread perform the array reduction */
            for (i=0; i<nclusters; i++)
            {
                for (j=0; j<nthreads; j++)
                {
                    int tid=j*CACHE_ADJUSTMENT;
                    new_centers_len[i] += partial_new_centers_len[tid][i];
                    partial_new_centers_len[tid][i] = 0.0;
                    for (k=0; k<nfeatures; k++)
                    {
                        new_centers[i][k] += partial_new_centers[tid][i][k];
                        partial_new_centers[tid][i][k] = 0.0;
                    }
                }
             }
        }

        /* replace old cluster centers with new_centers */
        for (i=0; i<nclusters; i++)
        {
            for (j=0; j<nfeatures; j++)
            {
                if (new_centers_len[i] > 0)
                    clusters[i][j] = new_centers[i][j] / new_centers_len[i];
                new_centers[i][j] = 0.0;   /* set back to 0 */
            }
            new_centers_len[i] = 0;   /* set back to 0 */
        }

        delta /= npoints;
    }
    while (delta > threshold && loop++ < max_iterations);
#else
    do
    {
        if (is_perform_atomic)
        {
#pragma omp parallel for \
                    private(i,j,index) \
                    firstprivate(npoints,nclusters,nfeatures) \
                    shared(feature,clusters,membership,new_centers,new_centers_len) \
                    schedule(static)
            for (i=0; i<npoints; i++)
            {
                /* find the index of nestest cluster centers */
                index = find_nearest_point(feature[i],
                                           nfeatures,
                                           clusters,
                                           nclusters);

                /* assign the membership to object i */
                membership[i] = index;

                /* update new cluster centers : sum of objects located within */
#pragma omp atomic
                new_centers_len[index]++;
                for (j=0; j<nfeatures; j++)
#pragma omp atomic
                    new_centers[index][j] += feature[i][j];
            }
        }
        else
        {

#pragma omp parallel \
                    shared(feature,clusters,membership,partial_new_centers,partial_new_centers_len)
            {
                int tid = omp_get_thread_num()*CACHE_ADJUSTMENT;
#pragma omp for \
                            private(i,j,index) \
                            firstprivate(npoints,nclusters,nfeatures) \
                            schedule(static)
                for (i=0; i<npoints; i++)
                {
                    /* find the index of nestest cluster centers */
                    index = find_nearest_point(feature[i],
                    nfeatures,
                    clusters,
                    nclusters);

                    /* assign the membership to object i */
                    membership[i] = index;

                    /* update new cluster centers : sum of all objects located
                    		       within */
                    partial_new_centers_len[tid][index]++;
                    for (j=0; j<nfeatures; j++)
                        partial_new_centers[tid][index][j] += feature[i][j];
                }
            } /* end of #pragma omp parallel */

            /* let the main thread perform the array reduction */
            for (i=0; i<nclusters; i++)
            {
                for (j=0; j<nthreads; j++)
                {
                    int tid=j*CACHE_ADJUSTMENT;
                    new_centers_len[i] += partial_new_centers_len[tid][i];
                    partial_new_centers_len[tid][i] = 0.0;
                    for (k=0; k<nfeatures; k++)
                    {
                        new_centers[i][k] += partial_new_centers[tid][i][k];
                        partial_new_centers[tid][i][k] = 0.0;
                    }
                }
            }
        }

        /* replace old cluster centers with new_centers */
        for (i=0; i<nclusters; i++)
        {
            for (j=0; j<nfeatures; j++)
            {
                if (new_centers_len[i] > 0)
                    clusters[i][j] = new_centers[i][j] / new_centers_len[i];
                new_centers[i][j] = 0.0;   /* set back to 0 */
            }
            new_centers_len[i] = 0;   /* set back to 0 */
        }

    }
    while (loop++ < max_iterations);
#endif
    if (_debug)
    {
        timing = omp_get_wtime() - timing;
        printf("Time for %d iterations is: %f (sec)\n",loop-1,timing);
    }

    free(new_centers[0]);
    free(new_centers);
    free(new_centers_len);
}
#else
/*----< kmeans_clustering() >---------------------------------------------*/
void kmeans_clustering( int     is_perform_atomic,  /* in:                        */
                        float **feature,            /* in: [npoints][nfeatures]   */
                        int     nfeatures,
                        int     npoints,
                        int     nclusters,
                        float **clusters,           /* in: [nclusters][nfeatures] */
                        float   threshold,
                        int    *membership,         /* out: [npoints]             */
                        int     max_iterations)
{

    int      i, j, k, index, loop=0;
    int     *new_centers_len; /* [nclusters]: no. of points in each cluster */
    float    delta;
    float  **new_centers;     /* [nclusters][nfeatures] */
    double   timing, assign_timing, reduce_timing, new_timing;
    double   assign_time, reduce_time, new_time;
    assign_time = reduce_time = new_time = 0;

    int      nthreads;
    int    **partial_new_centers_len;
    float ***partial_new_centers;
    int CACHE_ADJUSTMENT = (CACHE_BLOCK_SIZE_BYTES/(nclusters*sizeof(float)) + 1);

    nthreads = omp_get_max_threads();

    for (i=0; i<npoints; i++)
        membership[i] = -1;

    /* need to initialize new_centers_len and new_centers[0] to all 0 */
    new_centers_len = (int*) calloc(nclusters, sizeof(int));

    new_centers    = (float**) malloc(nclusters * sizeof(float*));
    new_centers[0] = (float*)  calloc(nclusters * nfeatures, sizeof(float));
    for (i=1; i<nclusters; i++)
        new_centers[i] = new_centers[i-1] + nfeatures;

    if (!is_perform_atomic)
    {
        partial_new_centers_len    = (int**) malloc(nthreads *CACHE_ADJUSTMENT*
                                     sizeof(int*));
        partial_new_centers_len[0] = (int*)  calloc(nthreads*nclusters*CACHE_ADJUSTMENT,
                                     sizeof(int));
        for (i=1; i<nthreads*CACHE_ADJUSTMENT; i++)
            partial_new_centers_len[i] = partial_new_centers_len[i-1]+nclusters;

        partial_new_centers    =(float***)malloc(nthreads *CACHE_ADJUSTMENT*
                                sizeof(float**));
        partial_new_centers[0] =(float**) malloc(nthreads*nclusters *CACHE_ADJUSTMENT*
                                sizeof(float*));
        for (i=1; i<nthreads*CACHE_ADJUSTMENT; i++)
            partial_new_centers[i] = partial_new_centers[i-1] + nclusters;
        for (i=0; i<nthreads*CACHE_ADJUSTMENT; i++)
            for (j=0; j<nclusters; j++)
                partial_new_centers[i][j] = (float*)calloc(nfeatures,
                                            sizeof(float));
    }

    if (_debug) timing = omp_get_wtime();

#if CHECK_THRESHOLD == 1
    do
    {
        delta = 0.0;

        if (is_perform_atomic)
        {
#pragma omp parallel for \
                    private(i,j,index) \
                    firstprivate(npoints,nclusters,nfeatures) \
                    shared(feature,clusters,membership,new_centers,new_centers_len) \
                    schedule(static) \
                    reduction(+:delta)
            for (i=0; i<npoints; i++)
            {
                /* find the index of nestest cluster centers */
                index = find_nearest_point(feature[i],
                                           nfeatures,
                                           clusters,
                                           nclusters);
                /* if membership changes, increase delta by 1 */
                if (membership[i] != index) delta += 1.0;

                /* assign the membership to object i */
                membership[i] = index;

                /* update new cluster centers : sum of objects located within */
#pragma omp atomic
                new_centers_len[index]++;
                for (j=0; j<nfeatures; j++)
#pragma omp atomic
                    new_centers[index][j] += feature[i][j];
            }
        }
        else
        {
            assign_timing = omp_get_wtime();
#pragma omp parallel \
                    shared(feature,clusters,membership,partial_new_centers,partial_new_centers_len)
            {
#pragma omp for \
                            private(i,j,index) \
                            firstprivate(npoints,nclusters,nfeatures) \
                            schedule(static) \
                            reduction(+:delta)
                for (i=0; i<npoints; i++)
                {
                    /* find the index of nestest cluster centers */
                    index = find_nearest_point(feature[i],
                    nfeatures,
                    clusters,
                    nclusters);
                    /* if membership changes, increase delta by 1 */
                    if (membership[i] != index) delta += 1.0;

                    /* assign the membership to object i */
                    membership[i] = index;
                }
            } /* end of #pragma omp parallel */
            assign_time += omp_get_wtime() - assign_timing;
            reduce_timing = omp_get_wtime();

#pragma omp parallel \
                    shared(feature,membership,partial_new_centers,partial_new_centers_len)
            {
                int tid = omp_get_thread_num()*CACHE_ADJUSTMENT;
#pragma omp for \
                            private(i,j,index) \
                            firstprivate(npoints,nfeatures) \
                            schedule(static)
                for (i=0; i<npoints; i++)
                {
                    /* find the index of nestest cluster centers */
                    index = membership[i];

                      /* update new cluster centers : sum of all objects located
                                 within */
                    partial_new_centers_len[tid][index]++;
                    for (j=0; j<nfeatures; j++)
                        partial_new_centers[tid][index][j] += feature[i][j];
                }
            } /* end of #pragma omp parallel */

            reduce_time += omp_get_wtime() - reduce_timing;
            new_timing = omp_get_wtime();

            /* let the main thread perform the array reduction */
            for (i=0; i<nclusters; i++)
            {
                for (j=0; j<nthreads; j++)
                {
                    int tid=j*CACHE_ADJUSTMENT;
                    new_centers_len[i] += partial_new_centers_len[tid][i];
                    partial_new_centers_len[tid][i] = 0.0;
                    for (k=0; k<nfeatures; k++)
                    {
                        new_centers[i][k] += partial_new_centers[tid][i][k];
                        partial_new_centers[tid][i][k] = 0.0;
                    }
                }
             }
        }

        /* replace old cluster centers with new_centers */
        for (i=0; i<nclusters; i++)
        {
            for (j=0; j<nfeatures; j++)
            {
                if (new_centers_len[i] > 0)
                    clusters[i][j] = new_centers[i][j] / new_centers_len[i];
                new_centers[i][j] = 0.0;   /* set back to 0 */
            }
            new_centers_len[i] = 0;   /* set back to 0 */
        }

        new_time += omp_get_wtime() - new_timing;

        delta /= npoints;
    }
    while (delta > threshold && loop++ < max_iterations);
#else
    do
    {
        if (is_perform_atomic)
        {
#pragma omp parallel for \
                    private(i,j,index) \
                    firstprivate(npoints,nclusters,nfeatures) \
                    shared(feature,clusters,membership,new_centers,new_centers_len) \
                    schedule(static)
            for (i=0; i<npoints; i++)
            {
                /* find the index of nestest cluster centers */
                index = find_nearest_point(feature[i],
                                           nfeatures,
                                           clusters,
                                           nclusters);

                /* assign the membership to object i */
                membership[i] = index;

                /* update new cluster centers : sum of objects located within */
#pragma omp atomic
                new_centers_len[index]++;
                for (j=0; j<nfeatures; j++)
#pragma omp atomic
                    new_centers[index][j] += feature[i][j];
            }
        }
        else
        {

            assign_timing = omp_get_wtime();
#pragma omp parallel \
                    shared(feature,clusters,membership,partial_new_centers,partial_new_centers_len)
            {
#pragma omp for \
                            private(i,j,index) \
                            firstprivate(npoints,nclusters,nfeatures) \
                            schedule(static)
                for (i=0; i<npoints; i++)
                {
                    /* find the index of nestest cluster centers */
                    index = find_nearest_point(feature[i],
                    nfeatures,
                    clusters,
                    nclusters);

                    /* assign the membership to object i */
                    membership[i] = index;
                }
            } /* end of #pragma omp parallel */

            assign_time += omp_get_wtime() - assign_timing;
            reduce_timing = omp_get_wtime();


#pragma omp parallel \
                    shared(feature,membership,partial_new_centers,partial_new_centers_len)
            {
                int tid = omp_get_thread_num()*CACHE_ADJUSTMENT;
#pragma omp for \
                            private(i,j,index) \
                            firstprivate(npoints,nfeatures) \
                            schedule(static)
                for (i=0; i<npoints; i++)
                {
                      /* find the index of nestest cluster centers */
                      index = membership[i];

                      /* update new cluster centers : sum of all objects located
                                 within */
                    partial_new_centers_len[tid][index]++;
                    for (j=0; j<nfeatures; j++)
                        partial_new_centers[tid][index][j] += feature[i][j];
                }
            } /* end of #pragma omp parallel */

            reduce_time += omp_get_wtime() - reduce_timing;
            new_timing = omp_get_wtime();

            /* let the main thread perform the array reduction */
            for (i=0; i<nclusters; i++)
            {
                for (j=0; j<nthreads; j++)
                {
                    int tid=j*CACHE_ADJUSTMENT;
                    new_centers_len[i] += partial_new_centers_len[tid][i];
                    partial_new_centers_len[tid][i] = 0.0;
                    for (k=0; k<nfeatures; k++)
                    {
                        new_centers[i][k] += partial_new_centers[tid][i][k];
                        partial_new_centers[tid][i][k] = 0.0;
                    }
                }
            }
        }

        /* replace old cluster centers with new_centers */
        for (i=0; i<nclusters; i++)
        {
            for (j=0; j<nfeatures; j++)
            {
                if (new_centers_len[i] > 0)
                    clusters[i][j] = new_centers[i][j] / new_centers_len[i];
                new_centers[i][j] = 0.0;   /* set back to 0 */
            }
            new_centers_len[i] = 0;   /* set back to 0 */
        }

        new_time += omp_get_wtime() - new_timing;


    }
    while (loop++ < max_iterations);
#endif
    if (_debug)
    {
        timing = omp_get_wtime() - timing;
        printf("Time for %d iterations is: %f (sec)\n",loop-1,timing);
    }

      printf("Assign time for %d iterations is: %f (sec)\n",loop-1,assign_time);
      printf("Internal Reduction time for %d iterations is: %f (sec)\n",loop-1,reduce_time);
      printf("External reduction plus setting cluster time for %d iterations is: %f (sec)\n",loop-1,new_time);


    free(new_centers[0]);
    free(new_centers);
    free(new_centers_len);
}
#endif
