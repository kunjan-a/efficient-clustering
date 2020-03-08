***************************************************************************
README FILE

Contents: Explains how to setup and execute parallel implementation of kmeans
***************************************************************************


----------------------------------------------------------------------------
COMPILATION:
----------------------------------------------------------------------------

Each folder contains a Makefile with the sources.
e.g.
cd c1060/centroids_in_constant_memory
make

To clear the binaries ond objects use target 'clean'
e.g.
cd c1060/centroids_in_constant_memory
make clean

----------------------------------------------------------------------------
EXECUTION:
----------------------------------------------------------------------------
e.g.
cd c1060/centroids_in_constant_memory
$bin/linux/release/example

NOTE: just typing the application name without any command line options would list the actual command line options that are available to the user.


       -n numPoints   : (mandatory) number of data points
       -k numClusters : (mandatory) number of clusters
       -d numAttribs  : (mandatory) number of attributes of each data point
       -f filename:     file containing data to be clustered
       -i iteratns    : number of iterations to be performed
       -p nprocAssign : number of threads per block for label assignment

Options -n, -k and -d are mandatory.
If no input file is given then random values are assigned to the input data points.
Initial centroids are assigned by picking one random point each from 0 to n/k-1, n/k to 2*n/k-1 and so on.

kmeans.h contains default values for remaining command-line flags.

It stores the final centroid and number of members for each cluster in the file defined by macro OUTPUTFILE inside kmeans.h
***************************************************************************
