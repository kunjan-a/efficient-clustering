/*************************************************************************/
/**   File:         example.c                                           **/
/**   Description:  Takes as input a file:                              **/
/**                 ascii  file: containing 1 data point per line       **/
/**                 This example performs a simple k-means clustering   **/
/**                 on the data.                                        **/
/*************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <unistd.h>
#include <fcntl.h>
#include <omp.h>

#include "kmeans.h"

extern double wtime(void);

/*---< usage() >------------------------------------------------------------*/
void usage(char *argv0)
{
    char *help ="Usage: %s [switches]\n"
        "       -n numPoints   : (mandatory) number of data points\n"
        "       -k numClusters : (mandatory) number of clusters\n"
        "       -d numAttribs  : (mandatory) number of attributes of each data point\n"
        "       -f filename:     file containing data to be clustered\n"
        "       -i iteratns    : number of iterations to be performed\n"
        "       -p nproc       : number of threads\n"
        "       -t threshold   : threshold value\n"
        "       -a             : perform atomic OpenMP pragma\n";
    fprintf(stderr, help, argv0);
    exit(-1);
}

/*---< main() >-------------------------------------------------------------*/
int main(int argc, char **argv)
{
    int     opt;
    extern char   *optarg;
    float **attributes=NULL;
    float **cluster_centres=NULL;
    int     i, j;
    int    *cluster_assign=NULL;
    int     numClusters, numAttributes, numObjects;
    int     numIter = NUM_ITERATION;
    int     nthreads = THREAD_NUM;
    int     is_perform_atomic = PERFORM_ATOMIC;
    float   threshold = THRESHOLD_VALUE;
    double  timing, io_timing, clustering_timing;
    char   *filename = 0;
    int mandatory_opts_provided = 0;
    int mandatory_options = 3;
    _debug=1;

    while ( (opt=getopt(argc,argv,"n:k:d:f:i:p:t:a"))!= EOF)
    {
        switch (opt)
        {
        case 'n':
            numObjects=atof(optarg);
            mandatory_opts_provided++;
            break;
        case 'k':
            numClusters = atoi(optarg);
            mandatory_opts_provided++;
            break;
        case 'd':
            numAttributes = atoi(optarg);
            mandatory_opts_provided++;
            break;
        case 'f':
            filename=optarg;
            break;
        case 'i':
            numIter=atof(optarg);
            break;
        case 'p':
            nthreads = atoi(optarg);
            break;
        case 't':
            threshold=atof(optarg);
            break;
        case 'a':
            is_perform_atomic = 1;
            break;
        case '?':
            usage(argv[0]);
            break;
        default:
            usage(argv[0]);
            break;
        }
    }
    if(mandatory_opts_provided != mandatory_options)  usage(argv[0]);

    setVars(numObjects, numClusters, numAttributes, &attributes, &cluster_assign);
    io_timing = omp_get_wtime();
    if(readInput(numObjects, numClusters, numAttributes, &attributes, filename))
    {

      timing            = omp_get_wtime();
      io_timing         = timing - io_timing;

      omp_set_num_threads(nthreads);
      printf("Number of threads = %d\n", omp_get_max_threads());
      printf("numObjects = %d, each of %d dimensions\n",
             numObjects,numAttributes);
      printf("total iterations are %d\n",numIter);

      printf("time for reading data values = %f (sec)\n", io_timing);

      setInitialClusters(numObjects, numClusters, numAttributes, attributes, &cluster_centres);
      printf("Initial cluster values set.\n");
/*
      printf("Data points are:\n");
      print_2d_arr(feature, nfeatures, npoints);

      printf("Initial clusters are:\n");
      print_2d_arr(clusters, nfeatures, nclusters);
*/

      timing            = omp_get_wtime();
      kmeans_clustering(is_perform_atomic, attributes, numAttributes, numObjects, numClusters, cluster_centres,
                        threshold, cluster_assign, numIter);
      clustering_timing = omp_get_wtime() - timing;
      printf("time for the complete kmeans_clustering function = %f (sec)\n", clustering_timing);

      io_timing         = omp_get_wtime();
      file_write(numClusters,numObjects,numAttributes,cluster_centres,cluster_assign);
      timing            = omp_get_wtime();
      io_timing         = timing - io_timing;
      printf("time for storing data values = %f (sec)\n", io_timing);

    }
    if(attributes != NULL)
    {
        free(attributes[0]);
        free(attributes);
        attributes=NULL;
    }
    if(cluster_assign != NULL)
    {
        free(cluster_assign);
        cluster_assign = NULL;
    }
    if(cluster_centres != NULL)
    {
        free(cluster_centres[0]);
        free(cluster_centres);
        cluster_centres=NULL;
    }

    return(0);
}

