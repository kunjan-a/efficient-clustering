***************************************************************************
README FILE

Contents: Explains how to setup and execute parallel implementation of kmeans (based on NU-MineBench 3.0)
***************************************************************************


----------------------------------------------------------------------------
COMPILATION:
----------------------------------------------------------------------------

PLEASE TRY TO USE THE FOLLOWING CONFIGURATION OF COMPILERS
* GNU GCC/G++  version 3.2 or above 
* Intel C++ Compiler version 7 or above 

kmeans:
cd src/kmeans
make example

----------------------------------------------------------------------------
EXECUTION:
----------------------------------------------------------------------------

$src/kmeans/example

NOTE: just typing the application name without any command line options would list the actual command line options that are available to the user.


       -n numPoints   : (mandatory) number of data points
       -k numClusters : (mandatory) number of clusters
       -d numAttribs  : (mandatory) number of attributes of each data point
       -f filename:     file containing data to be clustered
       -i iteratns    : number of iterations to be performed
       -p nproc       : number of threads
       -t threshold   : threshold value
       -a             : perform atomic OpenMP pragma

Options -n, -k and -d are mandatory.
If no input file is given then random values are assigned to the input data points.
Initial centroids are assigned by picking one random point each from 0 to n/k-1, n/k to 2*n/k-1 and so on.

src/kmeans/kmeans.h contains default values for remaining command-line flags.

It stores the final centroid and number of members for each cluster in the file defined by macro OUTPUTFILE inside src/kmeans/kmeans.h
***************************************************************************
