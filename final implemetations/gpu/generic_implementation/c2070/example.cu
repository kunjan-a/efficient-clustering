/*************************************************************************/
/**   File:         example.cu                                          **/
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

#include "kmeans.h"


/*---< usage() >------------------------------------------------------------*/
void usage(char *argv0)
{
    char *help ="Usage: %s [switches]\n"
        "       -n numPoints   : (mandatory) number of data points\n"
        "       -k numClusters : (mandatory) number of clusters\n"
        "       -d numAttribs  : (mandatory) number of attributes of each data point\n"
        "       -f filename:     file containing data to be clustered\n"
        "       -i iteratns    : number of iterations to be performed\n"
        "       -p nprocAssign : number of threads per block for label assignment\n"
        "       -l distLen     : number of centroids to be processed simultaneously\n"
        "       -P nprocReduce : number of threads per block for intra-block reduction\n"
        "       -w nwarpReduce : minimum number of warps per SM for intra-block reduction\n";
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
    int    *cluster_assign=NULL;
    int     numClusters, numAttributes, numObjects;
    int     numIter = NUM_ITERATION;
    int     nthreads_assign = THREAD_NUM_ASSIGN;
    int     distLen = DIST_LEN;
    int     nthreads_reduce = THREAD_NUM_REDUCE;
    int     minWarps_reduce = MIN_WARPS_PER_SM_REDUCE;
    float   timing;
    char   *filename = 0;
    int mandatory_opts_provided = 0;
    int mandatory_options = 3;

    while ( (opt=getopt(argc,argv,"n:k:d:f:i:p:l:P:w:"))!= EOF)
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
            nthreads_assign = atoi(optarg);
            break;
        case 'l':
            distLen = atoi(optarg);
            break;
        case 'P':
            nthreads_reduce = atoi(optarg);
            break;
        case 'w':
            minWarps_reduce = atoi(optarg);
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
    if(readInput(numObjects, numClusters, numAttributes, &attributes, filename))
    {
      printf("Number of threads per block during label assignment = %d\n", nthreads_assign);
      printf("Number of threads per block during intra-block reduction = %d\n", nthreads_reduce);
      printf("Minimum number of warps per SM during intra-block reduction = %d\n", minWarps_reduce);
      printf("Number of objects = %d, each of %d dimensions\n",
             numObjects,numAttributes);
      printf("Total iterations are %d\n",numIter);
      printf("Length of distance array inside label assignment kernel is %d\n",distLen);

      setInitialClusters(numObjects, numClusters, numAttributes, attributes, &cluster_centres);
      printf("Initial cluster values set.\n");
/*
      printf("Data points are:\n");
      print_2d_arr(feature, nfeatures, npoints);

      printf("Initial clusters are:\n");
      print_2d_arr(clusters, nfeatures, nclusters);
*/

      timing = kmeans_clustering(attributes, numAttributes, numObjects, numClusters, cluster_centres, cluster_assign,
                                numIter, nthreads_assign, distLen, nthreads_reduce, minWarps_reduce);

      printf("Total time taken = %f (millisecond)\n", timing);

      file_write(numClusters,numObjects,numAttributes,cluster_centres,cluster_assign);

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

