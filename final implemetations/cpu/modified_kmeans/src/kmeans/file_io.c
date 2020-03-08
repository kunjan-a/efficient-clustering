/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*   File:         file_io.c                                                 */
/*   Description:  This program reads point data from a file                 */
/*                 and write cluster output to files                         */
/*   Input file format:                                                      */
/*                 ascii  file: each line contains 1 data object             */
/*                 with attributes separated by space                        */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>  /* open() */
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>     /* read(), close() */

#include "kmeans.h"

void setVars(const int n, const int k, const int d, float ***attributes, int **cluster_assign)
{
    int i;

    *cluster_assign = (int*) malloc(n * sizeof(int));

    *attributes      = (float**)malloc(n * sizeof(float*));
    (*attributes)[0] = (float*) malloc(n * d * sizeof(float));
    for (i=1; i<n; i++)
        (*attributes)[i] = (*attributes)[i-1] + d;

}


void setInitialClusters(const int n, const int k, const int d, float **attributes, float ***cluster_centres)
{
    int i,j;
    srand(31);

    /* allocate space*/
    *cluster_centres    = (float**) malloc(k * sizeof(float*));
    (*cluster_centres)[0] = (float*)  malloc(k * d * sizeof(float));
    for (i=1; i<k; i++)
        (*cluster_centres)[i] = (*cluster_centres)[i-1] + d;

    float **clusters = *cluster_centres;
    /* randomly pick cluster centers */
    for (i=0; i<k; i++)
    {
        int pt_index = ((int)rand() % (int)(n/k))+((n/k)*i);
        for (j=0; j<d; j++){
            clusters[i][j] = attributes[pt_index][j];
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
//!Reads the input from a file with name set by input parameter filename
//! @param n number of points
//! @param k number of clusters
//! @param d number of attributes
///////////////////////////////////////////////////////////////////////////////
int readInput(const int n, const int k, const int d, float ***attributes, char *filename)
{
    int i,j;

    if(filename == 0) {
        srand(31);
        int maxi = 31;
        int l;
        for(l=0;l<n;l+=k){
          for ( i=0;i<k && i+l<n;i++)
          {
            for ( j=0; j<d ;j++ ){
              float val = (float)((rand()%(maxi*2+1))-maxi)/(maxi+1.0f);
              (*attributes)[i+l][j]=val;
            }
          }
        }
    } else  {
        FILE* fin=NULL;
        char *inputFile_name=filename;

        fin = fopen( inputFile_name, "r");
        if (fin!=NULL)
        {
          for ( i = 0; i < n; ++i)
          {
            for ( j=0; j<d ;j++ ){
              int read=fscanf(fin, "%f", &((*attributes)[i][j]));
              if( read<=0 ){
                printf("File has only %d values whereas it should have %d * %d i.e. %d values",(i*d)+j,n,d,n*d);
                printf("%s",inputFile_name);
                printf("\n");
                return 0;
              }
            }
          }

          printf("Finished reading ");
          printf("%s",inputFile_name);
          printf("\n");
          fclose(fin);
        }
        else
        {
            printf("Could not open file ");
            printf("%s",inputFile_name);
            printf("\n");
            return 0;
        }
    }
    return 1;
}



int file_write(int        numClusters,  /* no. clusters */
               int        numObjs,      /* no. data objects */
               int        numCoords,    /* no. coordinates (local) */
               float    **clusters,     /* [numClusters][numCoords] centers */
               int       *membership)   /* [numObjs] */
{
    FILE *fptr;
    int   i, j;

    char *outputFile_name = OUTPUTFILE;

    fptr = fopen( outputFile_name, "w");

    if(fptr != NULL){
        int* count = NULL;
        count = (int*) calloc(numClusters,sizeof(int));
        /* Calculate the number of points assigned to each cluster ----------*/
        for (i=0; i<numObjs; i++) {
            count[membership[i]]++;
        }

    /* output: the coordinates of the cluster centres ----------------------*/
#if WRITE_CENTROIDS == 1
        printf("Writing coordinates of K=%d cluster centers to file \"%s\"\n", numClusters, outputFile_name);
        fprintf(fptr, "%d centroids are:\n",numClusters);
        for (i=0; i<numClusters; i++) {
            for (j=0; j<numCoords; j++)
                fprintf(fptr, "%f ", clusters[i][j]);
            fprintf(fptr, "Total members: %d", count[i]);
            fprintf(fptr, "\n");
        }
#endif
        free(count);
        fclose(fptr);
    } else
    {
        printf("Could not open file ");
        printf("%s",outputFile_name);
        printf("\n");
        return 0;
    }

    return 1;
}

void print_2d_arr(float **arr, int width, int height) {
  int i,j;
  for(i=0; i<height; i++) {
    for(j=0; j<width; j++) {
      printf("%f ",arr[i][j]);
    }
    printf("\n");
  }
}
