#ifndef _H_KMEANS
#define _H_KMEANS

#ifndef FLT_MAX
#define FLT_MAX 3.40282347e+38
#endif

// Function declarations
float   euclid_dist_2        (float*, float*, int);
int     find_nearest_point   (float* , int, float**, int);

void setVars(const int numObjects, const int numClusters, const int numAttributes, float ***attributes, int **cluster_assign);
int readInput(const int numObjects, const int numClusters, const int numAttributes, float ***attributes, char *filename);
void setInitialClusters(const int numObjects, const int numClusters, const int numAttributes, float **attributes, float ***cluster_centres);
int file_write(int        numClusters,  /* no. clusters */
               int        numObjs,      /* no. data objects */
               int        numCoords,    /* no. coordinates (local) */
               float    **clusters,     /* [numClusters][numCoords] centers */
               int       *membership);  /* [numObjs] */
void print_2d_arr(float **arr, int width, int height);
void kmeans_clustering( int     is_perform_atomic,  /* in:                        */
                        float **feature,            /* in: [npoints][nfeatures]   */
                        int     nfeatures,
                        int     npoints,
                        int     nclusters,
                        float **clusters,           /* in: [nclusters][nfeatures] */
                        float   threshold,
                        int    *membership,         /* out: [npoints]             */
                        int     max_iterations);
int     _debug;

/*********************************************************************************/

// Default settings to be used
#define NUM_ITERATION 1                   
#define THREAD_NUM 2
#define PERFORM_ATOMIC 0                  // perform atomic OpenMP pragma
#define THRESHOLD_VALUE 0.001
#define CHECK_THRESHOLD 0

#define OUTPUTFILE "output.txt"           // File in which final centroids and number of members are written.

#define USE_SQRT 0                        // 1: Use square root for distance calculation
#if USE_SQRT == 1
  #define SQRT_MACRO(x)  sqrt((double)x)
  #define distance(x1,y1,x2,y2) sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1))
#else
  #define SQRT_MACRO(x)  x
  #define distance(x1,y1,x2,y2) (x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)
#endif

#define PROFILE_TIME 0                    // 1:profile internal function times too
#define DONT_CHNG_CENTROIDS 0             // 0: change centroids in each iteration as per the new mean values
                                          // 1: Leave the centroids as it is for each iteration.
#define WRITE_CENTROIDS 1                 // 1: Store the final centroid and number of members for each cluster in the file given by macro OUTPUTFILE
#define CACHE_BLOCK_SIZE_BYTES 256        // Change to a multiple of block size of system cache

#endif
