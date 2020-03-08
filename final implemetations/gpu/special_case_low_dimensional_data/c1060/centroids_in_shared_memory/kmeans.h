#ifndef _H_KMEANS
#define _H_KMEANS

#ifndef FLT_MAX
#define FLT_MAX 3.40282347e+38
#endif

// Function declarations

void setVars(const int numObjects, const int numClusters, const int numAttributes, float ***attributes, int **cluster_assign);
int readInput(const int numObjects, const int numClusters, const int numAttributes, float ***attributes, char *filename);
void setInitialClusters(const int numObjects, const int numClusters, const int numAttributes, float **attributes, float ***cluster_centres);
int file_write(int        numClusters,  /* no. clusters */
               int        numObjs,      /* no. data objects */
               int        numCoords,    /* no. coordinates (local) */
               float    **clusters,     /* [numClusters][numCoords] centers */
               int       *membership);  /* [numObjs] */
void print_2d_arr(float **arr, int width, int height);
float kmeans_clustering(float **attributes,       /* in: [numObjects][numAttributes]    */
                        int     numAttributes,
                        int     numObjects,
                        int     numClusters,
                        float **cluster_centres,  /* in: [numClusters][numAttributes]   */
                        int    *cluster_assign,   /* out: [numObjects]                  */
                        int     numIterations,
                        int     threadDim_assign);

/*********************************************************************************/

// Default settings to be used
#define NUM_ITERATION 1                   
#define THREAD_NUM_ASSIGN 256             // Default number of threads per block while assigning labels

#define OUTPUTFILE "output.txt"           // File in which final centroids and number of members are written.

#define WRITE_CENTROIDS 1                 // 1: Store the final centroid and number of members for each cluster in the file given by macro OUTPUTFILE

#define ACCESS_DATA_COALESCED             // If defined data is stored in column-major form else row-major form
//#define ACCESS_CENT_COALESCED           // If defined centroids are stored in column-major form else row-major form

#endif
