.KEEP_STATE:

DFLAGS      = -D_USE_ATOMIC_PRAGMA_
DFLAGS      =

OPTFLAGS    = -O2
INCFLAGS    =
CFLAGS      = $(OPTFLAGS) $(DFLAGS) $(INCFLAGS)
LDFLAGS     = -lm

# CC = cc -g -non_shared -taso -DANSI -D__STDC__
CC = gcc
CC = icc -openmp

H_FILES     = fuzzy_kmeans.h

SRC         = fuzzy_kmeans.c \
              cluster.c \
	      kmeans_clustering.c \
              example.c

OBJ         = $(SRC:%.c=%.o)

.c.o:
	$(CC) $(CFLAGS) -c $<

all: example

example: $(OBJ) $(H_FILES)
	$(CC) $(OPTFLAGS) -o example $(OBJ) $(LDFLAGS)

clean:
	rm -rf *.o example core* .make.state \
	       *.cluster_assign \
               *.cluster_centres group.*

tar:
	tar -cf kmeans.tar $(SRC) $(H_FILES) \
			   Makefile
	gzip kmeans.tar
