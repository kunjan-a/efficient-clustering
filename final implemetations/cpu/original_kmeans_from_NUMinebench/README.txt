***************************************************************************

README FILE

Author: Berkin Ozisikyilmaz
Email: nu-minebench AT ece DOT northwestern DOT edu
Contents: Explains how to setup and execute NU-MineBench 3.0

***************************************************************************


----------------------------------------------------------------------------
TO DOWNLOAD:
----------------------------------------------------------------------------
Go to: http://cucis.ece.northwestern.edu/projects/DMS/MineBench.html
Click on "Download" and download NU-MineBench-3.0.tar.gz



----------------------------------------------------------------------------
TO INSTALL:
----------------------------------------------------------------------------

tar -xvzf NU-MineBench-3.0.src.tar.gz
Lets call $DMHOME as the home for our data mining suite

NU-MineBench is a collection of data mining applications. Currently
there are 21 applications in the suite.


* kmeans - Partitioning based clustering application

* kmeans also contains a fuzzy based clustering application (execute with option -f to use fuzzy clustering)



----------------------------------------------------------------------------
COMPILATION:
----------------------------------------------------------------------------

PLEASE TRY TO USE THE FOLLOWING CONFIGURATION OF COMPILERS
* GNU GCC/G++  version 3.2 or above 
* Intel C++ Compiler version 7 or above 
* Intel Fortran Compiler version 8 or above
* Intel Math Kernel Library 7.2 or above


kmeans:
cd $DMHOME/src/kmeans
make example


----------------------------------------------------------------------------
EXECUTION:
----------------------------------------------------------------------------

$DMHOME/src/kmeans/example

NOTE: just typing the application name without any command
line options would list the actual command line options that are available to the
user.


***************************************************************************
