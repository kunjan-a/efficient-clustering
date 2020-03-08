.KEEP_STATE:

DFLAGS      = -D_USE_ATOMIC_PRAGMA_
DFLAGS      =

OPTFLAGS    = -O2
INCFLAGS    =
CFLAGS      = $(OPTFLAGS) $(DFLAGS) $(INCFLAGS)
LDFLAGS     = -lm

# CC = cc -g -non_shared -taso -DANSI -D__STDC__
CC = gcc -fopenmp
#CC = icc -openmp

H_FILES     = kmeans.h

SRC = kmeans_clustering.c \
              file_io.c \
              example.c

OBJ = $(SRC:%.c=%.o)

%.o: %.c $(H_FILES)
	$(CC) $(CFLAGS) -c -o $@ $<

all: example

example: $(OBJ) $(H_FILES)
	$(CC) $(OPTFLAGS) -o $@ $(OBJ) $(LDFLAGS)

clean:
	rm -rf *.o example core* .make.state \
	       *.cluster_assign \
               *.cluster_centres group.*

tar:
	tar -cf kmeans.tar $(SRC) $(H_FILES) \
			   Makefile
	gzip kmeans.tar
