#include <stdio.h>
#include <settings.h>

#define CONFLICT_FREE_INDEX(n) ( (n) + ((n) >> LOG_NUM_BANKS))


////////////////////////////////////////////////////////////////////////////////
//!Number of blocks (MAX_BLOCKS defines the upper limit)
//! @param n number of points
////////////////////////////////////////////////////////////////////////////////
__device__ __host__ int get_num_blocks(int n)
{
    //Must be atleast equal to number of GPU cores.
    //Can be more with lesser threads per block if there are too many synchronize calls required so that there is minimum relative blocking among threads.
    return 1 + ( ( ( 1 + (n-1)/POINTS_PER_THREAD ) - 1 ) >> LOG_BLOCKDIM );

}



////////////////////////////////////////////////////////////////////////////////
//!Length of the array required for barrier synchronization of blocks
//! @param numBlocks number of blocks
////////////////////////////////////////////////////////////////////////////////
__device__ __host__ int get_barrier_synch_array_length(int numBlocks)
{
    int length=0;

    while(numBlocks > 1)
    {
        length += numBlocks;
        numBlocks = (numBlocks -1 ) >> LOG_BLOCKDIM +1;
    }

    return length;

}


////////////////////////////////////////////////////////////////////////////////
//!Number of blocks rounded to nearest multiple of HalfWarp i.e. ceil(num_blocks/HalfWarp)*HalfWarp
//! @param n number of points
////////////////////////////////////////////////////////////////////////////////
__device__ __host__ unsigned int get_mem_length(int n)
{
    int blocks = get_num_blocks(n);
    return ((((blocks-1)>>LOG_HALF_WARP) + 1) << LOG_HALF_WARP);                                        //MemLength gives number of blocks rounded to nearest multiple of HalfWarp i.e. ceil(num_blocks/HalfWarp)*HalfWarp
}


/*__device__ float distance(const float &x1, const float &y1, const float &x2, const float &y2)
{
    return sqrtf((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1));
}*/


////////////////////////////////////////////////////////////////////////////////
//!Initialize all elements of an array with default values
////////////////////////////////////////////////////////////////////////////////
__global__ void initialize(int length, int *const __restrict__ array, int value)
{
    int index = (blockIdx.x << LOG_BLOCKDIM)+threadIdx.x;
    if(index < length)
        array[index]=value;

}

////////////////////////////////////////////////////////////////////////////////
//!Initialize all elements of an array with default values
////////////////////////////////////////////////////////////////////////////////
__global__ void initialize(int length, float *const __restrict__ array, float value)
{
    int index = (blockIdx.x << LOG_BLOCKDIM)+threadIdx.x;
    if(index < length)
        array[index]=value;

}

__global__ void
testBRDCST(int n, int k, float *const __restrict__ d_dataptx)
{
    // shared memory
    extern __shared__ float s_centroid[];                                                               //Stores x-coord of all clusters

    const unsigned int index = (blockIdx.x << LOG_BLOCKDIM) + threadIdx.x;                              //If we've T threads per block then index of ith thread in kth block is k*T + i. So we are maintaining an order among all threads of all blocks.

    // Same as blockIdx.x * blockDim.x + threadId.x

    float *const s_centroidx = s_centroid;                                                              //x-coord of 1st cluster's centroid

    //Assign the values of the global memory centroids to the shared memory centroids for all k clusters
    //coalesced access
    s_centroidx[threadIdx.x] = d_dataptx[threadIdx.x];
    __syncthreads();                                                                                    // BLOCKDIM-k%BLOCKDIM threads would have waited for maximum one iteration of above loop over here while others executes the above lines

//    float centroid_dist,new_centroid_dist;
    if (index<n)                                                                                        //Find centroid nearest to the datapoint at location index
    {
        float ptx = d_dataptx[index];

        for ( int rep=0; rep<1; rep++)
        {
            for (int i=0; i<BLOCKDIM; i++)
            {
#if USE_BRDCST == 0
                int indx=i+threadIdx.x;
                if (indx >= BLOCKDIM)
                    indx -= BLOCKDIM;
                ptx +=s_centroidx[indx];
#else
                ptx +=s_centroidx[i];
#endif
            }
// TODO: Shud try even after removing this syncthread. Ideally syncthread shudn't be req. here.
            __syncthreads();
            s_centroidx[threadIdx.x] = ptx/BLOCKDIM;
// TODO: Why havent we used a threa fence here.
            __syncthreads();
        }

        d_dataptx[index]=ptx;
    }
}

////////////////////////////////////////////////////////////////////////////////
//!Return threadIndex % HalfWarp.
// Useful for caoalesced acess from shared mem.
////////////////////////////////////////////////////////////////////////////////
__device__ unsigned int thrdModHalfWarp(unsigned int tid)
{

    if(HALF_WARP == 16)
    {
        switch(tid)
        {

        case 0:
        case 16:
        case 32:
        case 48:
        case 64:
        case 80:
        case 96:
        case 112:
        case 128:
        case 144:
        case 160:
        case 176:
        case 192:
        case 208:
        case 224:
        case 240:
        case 256:
        case 272:
        case 288:
        case 304:
        case 320:
        case 336:
        case 352:
        case 368:
        case 384:
        case 400:
        case 416:
        case 432:
        case 448:
        case 464:
        case 480:
        case 496:
        case 512:
            return 0;

        case 1:
        case 17:
        case 33:
        case 49:
        case 65:
        case 81:
        case 97:
        case 113:
        case 129:
        case 145:
        case 161:
        case 177:
        case 193:
        case 209:
        case 225:
        case 241:
        case 257:
        case 273:
        case 289:
        case 305:
        case 321:
        case 337:
        case 353:
        case 369:
        case 385:
        case 401:
        case 417:
        case 433:
        case 449:
        case 465:
        case 481:
        case 497:
            return 1;

        case 2:
        case 18:
        case 34:
        case 50:
        case 66:
        case 82:
        case 98:
        case 114:
        case 130:
        case 146:
        case 162:
        case 178:
        case 194:
        case 210:
        case 226:
        case 242:
        case 258:
        case 274:
        case 290:
        case 306:
        case 322:
        case 338:
        case 354:
        case 370:
        case 386:
        case 402:
        case 418:
        case 434:
        case 450:
        case 466:
        case 482:
        case 498:
            return 2;

        case 3:
        case 19:
        case 35:
        case 51:
        case 67:
        case 83:
        case 99:
        case 115:
        case 131:
        case 147:
        case 163:
        case 179:
        case 195:
        case 211:
        case 227:
        case 243:
        case 259:
        case 275:
        case 291:
        case 307:
        case 323:
        case 339:
        case 355:
        case 371:
        case 387:
        case 403:
        case 419:
        case 435:
        case 451:
        case 467:
        case 483:
        case 499:
            return 3;

        case 4:
        case 20:
        case 36:
        case 52:
        case 68:
        case 84:
        case 100:
        case 116:
        case 132:
        case 148:
        case 164:
        case 180:
        case 196:
        case 212:
        case 228:
        case 244:
        case 260:
        case 276:
        case 292:
        case 308:
        case 324:
        case 340:
        case 356:
        case 372:
        case 388:
        case 404:
        case 420:
        case 436:
        case 452:
        case 468:
        case 484:
        case 500:
            return 4;

        case 5:
        case 21:
        case 37:
        case 53:
        case 69:
        case 85:
        case 101:
        case 117:
        case 133:
        case 149:
        case 165:
        case 181:
        case 197:
        case 213:
        case 229:
        case 245:
        case 261:
        case 277:
        case 293:
        case 309:
        case 325:
        case 341:
        case 357:
        case 373:
        case 389:
        case 405:
        case 421:
        case 437:
        case 453:
        case 469:
        case 485:
        case 501:
            return 5;

        case 6:
        case 22:
        case 38:
        case 54:
        case 70:
        case 86:
        case 102:
        case 118:
        case 134:
        case 150:
        case 166:
        case 182:
        case 198:
        case 214:
        case 230:
        case 246:
        case 262:
        case 278:
        case 294:
        case 310:
        case 326:
        case 342:
        case 358:
        case 374:
        case 390:
        case 406:
        case 422:
        case 438:
        case 454:
        case 470:
        case 486:
        case 502:
            return 6;

        case 7:
        case 23:
        case 39:
        case 55:
        case 71:
        case 87:
        case 103:
        case 119:
        case 135:
        case 151:
        case 167:
        case 183:
        case 199:
        case 215:
        case 231:
        case 247:
        case 263:
        case 279:
        case 295:
        case 311:
        case 327:
        case 343:
        case 359:
        case 375:
        case 391:
        case 407:
        case 423:
        case 439:
        case 455:
        case 471:
        case 487:
        case 503:
            return 7;

        case 8:
        case 24:
        case 40:
        case 56:
        case 72:
        case 88:
        case 104:
        case 120:
        case 136:
        case 152:
        case 168:
        case 184:
        case 200:
        case 216:
        case 232:
        case 248:
        case 264:
        case 280:
        case 296:
        case 312:
        case 328:
        case 344:
        case 360:
        case 376:
        case 392:
        case 408:
        case 424:
        case 440:
        case 456:
        case 472:
        case 488:
        case 504:
            return 8;

        case 9:
        case 25:
        case 41:
        case 57:
        case 73:
        case 89:
        case 105:
        case 121:
        case 137:
        case 153:
        case 169:
        case 185:
        case 201:
        case 217:
        case 233:
        case 249:
        case 265:
        case 281:
        case 297:
        case 313:
        case 329:
        case 345:
        case 361:
        case 377:
        case 393:
        case 409:
        case 425:
        case 441:
        case 457:
        case 473:
        case 489:
        case 505:
            return 9;

        case 10:
        case 26:
        case 42:
        case 58:
        case 74:
        case 90:
        case 106:
        case 122:
        case 138:
        case 154:
        case 170:
        case 186:
        case 202:
        case 218:
        case 234:
        case 250:
        case 266:
        case 282:
        case 298:
        case 314:
        case 330:
        case 346:
        case 362:
        case 378:
        case 394:
        case 410:
        case 426:
        case 442:
        case 458:
        case 474:
        case 490:
        case 506:
            return 10;

        case 11:
        case 27:
        case 43:
        case 59:
        case 75:
        case 91:
        case 107:
        case 123:
        case 139:
        case 155:
        case 171:
        case 187:
        case 203:
        case 219:
        case 235:
        case 251:
        case 267:
        case 283:
        case 299:
        case 315:
        case 331:
        case 347:
        case 363:
        case 379:
        case 395:
        case 411:
        case 427:
        case 443:
        case 459:
        case 475:
        case 491:
        case 507:
            return 11;

        case 12:
        case 28:
        case 44:
        case 60:
        case 76:
        case 92:
        case 108:
        case 124:
        case 140:
        case 156:
        case 172:
        case 188:
        case 204:
        case 220:
        case 236:
        case 252:
        case 268:
        case 284:
        case 300:
        case 316:
        case 332:
        case 348:
        case 364:
        case 380:
        case 396:
        case 412:
        case 428:
        case 444:
        case 460:
        case 476:
        case 492:
        case 508:
            return 12;

        case 13:
        case 29:
        case 45:
        case 61:
        case 77:
        case 93:
        case 109:
        case 125:
        case 141:
        case 157:
        case 173:
        case 189:
        case 205:
        case 221:
        case 237:
        case 253:
        case 269:
        case 285:
        case 301:
        case 317:
        case 333:
        case 349:
        case 365:
        case 381:
        case 397:
        case 413:
        case 429:
        case 445:
        case 461:
        case 477:
        case 493:
        case 509:
            return 13;

        case 14:
        case 30:
        case 46:
        case 62:
        case 78:
        case 94:
        case 110:
        case 126:
        case 142:
        case 158:
        case 174:
        case 190:
        case 206:
        case 222:
        case 238:
        case 254:
        case 270:
        case 286:
        case 302:
        case 318:
        case 334:
        case 350:
        case 366:
        case 382:
        case 398:
        case 414:
        case 430:
        case 446:
        case 462:
        case 478:
        case 494:
        case 510:
            return 14;

        case 15:
        case 31:
        case 47:
        case 63:
        case 79:
        case 95:
        case 111:
        case 127:
        case 143:
        case 159:
        case 175:
        case 191:
        case 207:
        case 223:
        case 239:
        case 255:
        case 271:
        case 287:
        case 303:
        case 319:
        case 335:
        case 351:
        case 367:
        case 383:
        case 399:
        case 415:
        case 431:
        case 447:
        case 463:
        case 479:
        case 495:
        case 511:
            return 15;

        default:
            return tid%HALF_WARP;

        }
    }
    return tid&(HALF_WARP-1);
}


#if POINTS_PER_THREAD == 1
__device__ int inline load_data_points(const unsigned int n,float * const __restrict__ ptx, float * const __restrict__ pty, const unsigned int index, const float *const __restrict__ d_dataptx,
                                       const float *const __restrict__ d_datapty, float *const __restrict__ s_thrd_x, float *const __restrict__ s_thrd_y)
{

  #if TRY_CACHE_BW_DEV_AND_SHARED == 1
    __syncthreads();
  #endif

    if (index < n )
    {
        *ptx=d_dataptx[index];
        *pty=d_datapty[index];

        s_thrd_x[threadIdx.x]=*ptx;
        s_thrd_y[threadIdx.x]=*pty;
        return 1;
    }
    return 0;
}
#else


__device__ int inline get_my_point_num(const int points_remaining)
{

    if(points_remaining <= 0)
        return 0;
    else if ((POINTS_PER_THREAD << LOG_BLOCKDIM) > points_remaining)               //This is the last block and not all threads might be processing POINTS_PER_THREAD points.
    {
        int my_point_num = points_remaining >> LOG_BLOCKDIM;                       //Next three lines take care of remainder if points_remaining is not divisible by BLOCKDIM
        if( (my_point_num << LOG_BLOCKDIM) + threadIdx.x < points_remaining)       //Load balance s.t. diff. b/w no. of pts processed by any two threads of this block is <= 1.
            my_point_num++;
        return my_point_num;
    }

    return POINTS_PER_THREAD;

}


__device__ void inline load_data_points_given_point_num(const unsigned int my_point_num, float *const __restrict__ ptx, float *const __restrict__ pty, const unsigned int index,
                                                        const float *const __restrict__ d_dataptx, const float *const __restrict__ d_datapty,
                                                        float *const __restrict__ s_thrd_x, float *const __restrict__ s_thrd_y)
{
#if TRY_CACHE_BW_DEV_AND_SHARED == 1
    __syncthreads();
#endif

    switch(my_point_num)
    {
    case POINTS_PER_THREAD:
        //#pragma unroll POINTS_PER_THREAD

        for(int i=0; i<POINTS_PER_THREAD; i++)
        {
            ptx[i]=d_dataptx[index+(i<<LOG_BLOCKDIM)];
            pty[i]=d_datapty[index+(i<<LOG_BLOCKDIM)];

            s_thrd_x[threadIdx.x + (i<<LOG_BLOCKDIM)]=ptx[i];
            s_thrd_y[threadIdx.x + (i<<LOG_BLOCKDIM)]=pty[i];
        }
        break;

    case 0:
        break;

    default:              //Only load as many points as are allotted to this thread.
        for(int i=0; i<my_point_num; i++)
        {
            ptx[i]=d_dataptx[index+(i<<LOG_BLOCKDIM)];
            pty[i]=d_datapty[index+(i<<LOG_BLOCKDIM)];

            s_thrd_x[threadIdx.x + (i<<LOG_BLOCKDIM)]=ptx[i];
            s_thrd_y[threadIdx.x + (i<<LOG_BLOCKDIM)]=pty[i];
        }
    }

#if TRY_CACHE_BW_DEV_AND_SHARED == 1
    __syncthreads();
#endif
}


__device__ int inline load_data_points(const unsigned int n,float *const __restrict__ ptx, float *const __restrict__ pty, const unsigned int index,
                                       const float *const __restrict__ d_dataptx, const float *const __restrict__ d_datapty,
                                       float *const __restrict__ s_thrd_x, float *const __restrict__ s_thrd_y)
{
    const int points_remaining = n - (blockIdx.x << LOG_BLOCKDIM)*POINTS_PER_THREAD;                      //Total number of points to be processed by this and later blocks.

    int my_point_num=get_my_point_num(points_remaining);

    load_data_points_given_point_num(my_point_num, ptx, pty, index, d_dataptx, d_datapty, s_thrd_x, s_thrd_y);
    return my_point_num;
}


__device__ int inline load_data_points(const unsigned int n, const unsigned int curr_point_cycle, const unsigned int total_point_cycles,
                                       float *const __restrict__ ptx, float *const __restrict__ pty, const unsigned int index,
                                       const float *const __restrict__ d_dataptx, const float *const __restrict__ d_datapty,
                                       float *const __restrict__ s_thrd_x, float *const __restrict__ s_thrd_y)
{
    int my_point_num=POINTS_PER_THREAD;

    if(curr_point_cycle+1 == total_point_cycles)                                                            // This is the last point cycle
    {
        const int points_remaining = n - (gridDim.x*curr_point_cycle + blockIdx.x)*(POINTS_PER_THREAD << LOG_BLOCKDIM);               //Total number of points to be processed by this and later blocks.
        my_point_num=get_my_point_num(points_remaining);
    }

    load_data_points_given_point_num(my_point_num, ptx, pty, index, d_dataptx, d_datapty, s_thrd_x, s_thrd_y);
    return my_point_num;
}

#endif


__device__ void inline load_centroids_in_shared(const unsigned int k, float *const __restrict__ s_centroidx, float *const __restrict__ s_centroidy,
        volatile const float *const __restrict__ d_centroidx, volatile const float *const __restrict__ d_centroidy)
{
    // Assign the values of the global memory centroids to the shared memory centroids for all k clusters
    // coalesced access
#if TRY_CACHE_BW_DEV_AND_SHARED == 1
    __syncthreads();
#endif
    // TODO: Make this assignment go at top even b4 data point loads as syncthreads is essential after this and
    // if points are loaded b4 this then it might cause us to wait for all data points to be loaded by all the threads in the block.
    // But by pushing this code to top, we can then load points after the syncthreads and then assign closest centroids b4 needing the next synthreads.
    // Thus, more parallelism might be available.
    for(int i=threadIdx.x; i<k; i+=BLOCKDIM)
    {
        s_centroidx[i] = d_centroidx[i];
        s_centroidy[i] = d_centroidy[i];
    }

}


#if POINTS_PER_THREAD == 1
__device__ signed int inline find_closest_centroid( const unsigned int n, const unsigned int k,const float ptx, const float pty, const unsigned int index,
        volatile const float *const __restrict__ s_centroidx, volatile const float *const __restrict__ s_centroidy)
{

    if (index < n)
    {
        int closestCentroid=0;

        // Find centroid nearest to the datapoint at location: index.
        float centroid_dist,new_centroid_dist;
  #if USE_BRDCST == 0
        closestCentroid = thrdModHalfWarp(threadIdx.x);
        int centroidIdx = closestCentroid;
  #endif
        centroid_dist=distance( ptx, pty, s_centroidx[closestCentroid], s_centroidy[closestCentroid]);
        for (int i =  1; i < k; ++i)
        {
  #if USE_BRDCST == 0
            centroidIdx++;
            if (centroidIdx >= k)
                centroidIdx -= k;

            new_centroid_dist=distance( ptx, pty, s_centroidx[centroidIdx], s_centroidy[centroidIdx]);
  #else
            // Broadcast access from shared mem. This line shud be executed by all threads in warp simultaneously.
            new_centroid_dist=distance( ptx, pty, s_centroidx[i], s_centroidy[i]);
  #endif
            if (  new_centroid_dist < centroid_dist )
            {
                centroid_dist=new_centroid_dist;
  #if USE_BRDCST == 0
                closestCentroid = centroidIdx;
  #else
                closestCentroid = i;
  #endif
            }
        }
        return closestCentroid;
    }
    else
        return -1;                                                                           // Put an invalid k value.

}
#else
__device__ void inline find_closest_centroid(const unsigned int k,const float *const __restrict__ ptx, const float *const __restrict__ pty,
                                             volatile const float *const __restrict__ s_centroidx, volatile const float *const __restrict__ s_centroidy,
                                             const int my_point_num, int *const __restrict__ closestCentroid)
{
    int startIndex=0;

    // Find centroid nearest to the datapoint at location: index.
    float centroid_dist[POINTS_PER_THREAD],new_centroid_dist[POINTS_PER_THREAD];

  #if USE_BRDCST == 0
    startIndex = thrdModHalfWarp(threadIdx.x);
    int centroidIdx = startIndex;
  #endif
    float centroidx, centroidy;
    switch(my_point_num)
    {
    case POINTS_PER_THREAD:
        centroidx=s_centroidx[startIndex];
        centroidy=s_centroidy[startIndex];
//      #pragma unroll POINTS_PER_THREAD
        for(int i=0; i< POINTS_PER_THREAD ; i++)
        {
            closestCentroid[i]=startIndex;
              centroid_dist[i]=distance( ptx[i], pty[i], centroidx, centroidy);
        }

        for (int centroid =  1; centroid < k; ++centroid)
        {

  #if USE_BRDCST == 0
            centroidIdx++;
            if (centroidIdx >= k)
                centroidIdx -= k;

            centroidx=s_centroidx[centroidIdx];
            centroidy=s_centroidy[centroidIdx];
  #else
            //Broadcast access from shared mem. This line shud be executed by all threads in warp simultaneously.
            centroidx=s_centroidx[centroid];
            centroidy=s_centroidy[centroid];
  #endif
//          #pragma unroll POINTS_PER_THREAD
            for(int i=0; i<POINTS_PER_THREAD; i++)
            {
                new_centroid_dist[i]=distance( ptx[i], pty[i], centroidx, centroidy);

                if (  new_centroid_dist[i] < centroid_dist[i] )
                {
                    centroid_dist[i]=new_centroid_dist[i];
  #if USE_BRDCST == 0
                    closestCentroid[i] = centroidIdx;
  #else
                    closestCentroid[i] = centroid;
  #endif
                }
            }
        }
        break;

    case 0:
//     #pragma unroll POINTS_PER_THREAD
        for(int i=0; i<POINTS_PER_THREAD; i++)
            closestCentroid[i] = -1;

        break;

    default:
        centroidx=s_centroidx[startIndex];
        centroidy=s_centroidy[startIndex];
        for(int i=0; i<my_point_num; i++)
        {
            closestCentroid[i]=startIndex;
              centroid_dist[i]=distance( ptx[i], pty[i], centroidx, centroidy);
        }

        for (int centroid =  1; centroid < k; ++centroid)
        {
  #if USE_BRDCST == 0
            centroidIdx++;
            if (centroidIdx >= k)
                centroidIdx -= k;

            centroidx=s_centroidx[centroidIdx];
            centroidy=s_centroidy[centroidIdx];
  #else
            //Broadcast access from shared mem. This line shud be executed by all threads in warp simultaneously.
            centroidx=s_centroidx[centroid];
            centroidy=s_centroidy[centroid];
  #endif
            for(int i=0; i<my_point_num; i++)
            {
                new_centroid_dist[i]=distance( ptx[i], pty[i], centroidx, centroidy);
                if (  new_centroid_dist[i] < centroid_dist[i] )
                {
                    centroid_dist[i]=new_centroid_dist[i];
#if USE_BRDCST == 0
                    closestCentroid[i] = centroidIdx;
#else
                    closestCentroid[i] = centroid;
#endif
                }
            }
        }

        // Store closestNum as the index of the closest centroid to the datapoint.
        for(int i=my_point_num; i<POINTS_PER_THREAD; i++)
            closestCentroid[i] = -1;
    }
}
#endif


#if POINTS_PER_THREAD == 1
__device__ void inline store_nearest_in_shared(const unsigned int n, const unsigned int k,const float ptx, const float pty,
        const unsigned int index, volatile const float *const __restrict__ s_centroidx, volatile const float *const __restrict__ s_centroidy,
        int *const __restrict__ s_thrd_k)
{
    // Store the index of the centroid closest to the datapoint.
      s_thrd_k[threadIdx.x]=find_closest_centroid( n, k, ptx, pty, index, s_centroidx, s_centroidy);
}

#else

__device__ void inline store_nearest_in_shared(const unsigned int k,const float *const __restrict__ ptx, const float *const __restrict__ pty,
        volatile const float *const __restrict__ s_centroidx, volatile const float *const __restrict__ s_centroidy,
        int *const __restrict__ s_thrd_k, const int my_point_num)
{
    // Find and store closest centroid if this thread holds a data-point, else store -1
    switch(my_point_num)
    {
    case 0:
//      #pragma unroll POINTS_PER_THREAD
        for(int i=0; i<POINTS_PER_THREAD; i++)
            s_thrd_k[threadIdx.x + (i<<LOG_BLOCKDIM)]=-1;       // Put an invalid k value.

        break;
    default:
        int closestCentroid[POINTS_PER_THREAD];
        find_closest_centroid( k, ptx, pty, s_centroidx, s_centroidy, my_point_num, closestCentroid);

        // Store closestNum as the index of the closest centroid to the datapoint.
//      #pragma unroll POINTS_PER_THREAD
        for(int i=0; i<POINTS_PER_THREAD; i++)
            s_thrd_k[threadIdx.x + (i<<LOG_BLOCKDIM)]=closestCentroid[i];

    }
}
#endif


#if POINTS_PER_THREAD == 1
__device__ void inline store_nearest_in_global(const unsigned int n, const unsigned int k,const float *const __restrict__ ptx,
        const float *const __restrict__ pty, const unsigned int index, volatile const float *const __restrict__ s_centroidx,
        volatile const float *const __restrict__ s_centroidy, int *const __restrict__ d_clusterno)
{
    // Find and store closest centroid if this thread holds a data-point, else store -1

  #if TRY_CACHE_BW_DEV_AND_SHARED == 1
    __syncthreads();
  #endif

      if(index<n)
      // Store the index of the centroid closest to the datapoint.
        d_clusterno[index]=find_closest_centroid( n, k, *ptx, *pty, index, s_centroidx, s_centroidy);

}
#else


__device__ void inline store_nearest_in_global(const unsigned int k,const float *const __restrict__ ptx, const float *const __restrict__ pty,
        const unsigned int index, volatile const float *const __restrict__ s_centroidx, volatile const float *const __restrict__ s_centroidy,
        int *const __restrict__ d_clusterno, const int my_point_num)
{
    // Find and store closest centroid if this thread holds a data-point, else store -1
    int closestCentroid[POINTS_PER_THREAD];

    if(my_point_num > 0)
        find_closest_centroid(k, ptx, pty, s_centroidx, s_centroidy, my_point_num, closestCentroid);

#if TRY_CACHE_BW_DEV_AND_SHARED == 1
    __syncthreads();
#endif

    switch(my_point_num)
    {
    case POINTS_PER_THREAD:
        // Store closestNum as the index of the closest centroid to the datapoint.
//      #pragma unroll POINTS_PER_THREAD
        for(int i=0; i<POINTS_PER_THREAD; i++)
            d_clusterno[index + (i<<LOG_BLOCKDIM)]=closestCentroid[i];
        break;
    case 0:
        break;
    default:
        // Store closestNum as the index of the closest centroid to the datapoint.
//      #pragma unroll POINTS_PER_THREAD
        for(int i=0; i<my_point_num; i++)
            d_clusterno[index + (i<<LOG_BLOCKDIM)]=closestCentroid[i];
    }

}
#endif

__device__ int inline reduceThreads(const unsigned int k, float *const __restrict__ sumx, float *const __restrict__ sumy,
                                    volatile const float *const __restrict__ s_thrd_x, volatile const float *const __restrict__ s_thrd_y,
                                    volatile const int *const __restrict__ s_thrd_k)
{
    int count=0;
    if (threadIdx.x < k)
    {
#if TRY_ATOMIC_CNTR_IN_LOCAL_REDCN == 1
        __shared__ unsigned int points_checked;
        points_checked=0;
//TODO: Unroll the following for loop, if unrolling does not cause memory spilling.
        for (int i = 0; points_checked < BLOCKDIM*POINTS_PER_THREAD && i < BLOCKDIM*POINTS_PER_THREAD; ++i)
        {
            int indx=i+threadIdx.x;
            if (indx >= BLOCKDIM*POINTS_PER_THREAD)
                indx -= BLOCKDIM*POINTS_PER_THREAD;
            if (threadIdx.x == s_thrd_k[indx])                                                          // Coalesced access from shared mem.
            {
                // Only one thread from all the warps wud enter in, all other warps wud move to the next value of i.
                // In all WarpSize -1 threads wait here every time one of the threads meets the if-condition for some i.
                // Max. no. of warps that can be simultaneously in this if-cond while executing for diff. values of i is: 2*(broad. shared access) + 3*floating pt. arithmetic instr

                *sumx = (*sumx) + s_thrd_x[indx];
                *sumy = (*sumy) + s_thrd_y[indx];
                count++;
                atomicInc(&points_checked,BLOCKDIM*POINTS_PER_THREAD);
            }
        }
#else
//TODO: Unroll the following for loop, if unrolling does not cause memory spilling.
        for (int i = 0; i < BLOCKDIM*POINTS_PER_THREAD; ++i)
        {
  #if USE_BRDCST == 0
            int indx=i+threadIdx.x;
            if (indx >= BLOCKDIM*POINTS_PER_THREAD)
                indx -= BLOCKDIM*POINTS_PER_THREAD;
            if (threadIdx.x == s_thrd_k[indx])                                                          // Coalesced access from shared mem.
            {
                // Only one thread from all the warps wud enter in, all other warps wud move to the next value of i.
                // In all WarpSize -1 threads wait here every time one of the threads meets the if-condition for some i.
                // Max. no. of warps that can be simultaneously in this if-cond while ececuting for diff. values of i is: 2*(broad. shared access) + 3*floating pt. arithmetic instr

                *sumx = (*sumx) + s_thrd_x[indx];
                *sumy = (*sumy) + s_thrd_y[indx];
                count++;
            }
  #else
        // Broadcast access from shared mem. This line shud be executed by all threads in warp simultaneously.
        if (threadIdx.x == s_thrd_k[i])
        {
            *sumx = (*sumx) + s_thrd_x[i];
            *sumy = (*sumy) + s_thrd_y[i];
            count++;
        }
  #endif
        }

#endif

    }

    return count;
}

/* This code is to be used when tackling fewer threads as compared to centroids. Correct for valatiles then.
#if TRY_ATOMIC_CNTR_IN_LOCAL_REDCN == 1
__device__ int inline reduceThreads(const unsigned int checkFor, float *const __restrict__ sumx, float *const __restrict__ sumy,
                                    const float *const __restrict__ s_thrd_x, const float *const __restrict__ s_thrd_y,
                                    const int *const __restrict__ s_thrd_k, unsigned int * const __restrict__ points_checked)
{
    int count=0;

    {
//TODO: Unroll the following for loop, if unrolling does not cause memory spilling.
        for (int i = 0; (*points_checked) < BLOCKDIM*POINTS_PER_THREAD && i < BLOCKDIM*POINTS_PER_THREAD; ++i)
        {
            int indx=i+checkFor;
            if (indx >= BLOCKDIM*POINTS_PER_THREAD)
                indx -= BLOCKDIM*POINTS_PER_THREAD;
            if (checkFor == s_thrd_k[indx])                                                          // Coalesced access from shared mem.
            {
                // Only one thread from all the warps wud enter in, all other warps wud move to the next value of i.
                // In all WarpSize -1 threads wait here every time one of the threads meets the if-condition for some i.
                // Max. no. of warps that can be simultaneously in this if-cond while executing for diff. values of i is: 2*(broad. shared access) + 3*floating pt. arithmetic instr

                *sumx = (*sumx) + s_thrd_x[indx];
                *sumy = (*sumy) + s_thrd_y[indx];
                count++;
                atomicInc(points_checked,BLOCKDIM*POINTS_PER_THREAD);
            }
        }
    }
    return count;
}
#else


__device__ int inline reduceThreads(const unsigned int checkFor, float *const __restrict__ sumx, float *const __restrict__ sumy,
                                    const float *const __restrict__ s_thrd_x, const float *const __restrict__ s_thrd_y,
                                    const int *const __restrict__ s_thrd_k)
{
    int count=0;

    {
//TODO: Unroll the following for loop, if unrolling does not cause memory spilling.
        for (int i = 0; i < BLOCKDIM*POINTS_PER_THREAD; ++i)
        {
#if USE_BRDCST == 0
            int indx=i+checkFor;
            if (indx >= BLOCKDIM*POINTS_PER_THREAD)
                indx -= BLOCKDIM*POINTS_PER_THREAD;
            if (checkFor == s_thrd_k[indx])                                                          // Coalesced access from shared mem.
            {
                // Only one thread from all the warps wud enter in, all other warps wud move to the next value of i.
                // In all WarpSize -1 threads wait here every time one of the threads meets the if-condition for some i.
                // Max. no. of warps that can be simultaneously in this if-cond while ececuting for diff. values of i is: 2*(broad. shared access) + 3*floating pt. arithmetic instr

                *sumx = (*sumx) + s_thrd_x[indx];
                *sumy = (*sumy) + s_thrd_y[indx];
                count++;
            }
#else
        // Broadcast access from shared mem. This line shud be executed by all threads in warp simultaneously.
            if (checkFor == s_thrd_k[i])
            {
                *sumx = (*sumx) + s_thrd_x[i];
                *sumy = (*sumy) + s_thrd_y[i];
                count++;
            }
#endif
        }
    }

    return count;
}
#endif

*/

__device__ void inline reduceBlocks(const int k,const int goalVal,unsigned int blocks,const unsigned int storageIndex, float *const __restrict__ sumx,
                                    float *const __restrict__ sumy,int *const __restrict__ count,volatile float *const __restrict__ d_sumx,
                                    volatile float *const __restrict__ d_sumy, volatile int *const __restrict__ d_count,unsigned int reductionDist,
                                    volatile int *const __restrict__ Arrayin)//, int *d_reducedCounts, clock_t *const d_timeVar)
{
    // Perform reduction till you have no one left on your right to reduce with.
    // At that stage store your own reduced values in global mem. and return.
    // Only blockIdx 0, wont store its reduced values as it will directly use them to find centroids.
    // Hence, we wait till blockIdx 0 is the only block left in this loop.
    while(blocks > 1)
    {

        if((blockIdx.x & ((reductionDist<<1) -1)) == 0)//% (reductionDist<<1) == 0 )                                 // (x % 2^k == 0) is same as (x & ((2^k)-1) == 0)
        {
            unsigned int reduceWithBlockIdx=blockIdx.x+reductionDist;
            if(reduceWithBlockIdx < gridDim.x)
            {

//                if(threadIdx.x < k)
//                {
//                    int *storeAt;
//                    storeAt = d_reducedCounts + ((k+2)<<1)*blockIdx.x/reductionDist;
//                    d_reducedCounts += ((k+2)<<1)*blocks;
//                    if(reductionDist>1)
//                    {
//                        storeAt += ((k+2)<<1)*blockIdx.x/reductionDist + (k+2)<<1;
//                        d_reducedCounts += ((k+2)<<1)*blocks;
//                    }
//
//                    if(threadIdx.x == 0)
//                    {
//                        *storeAt = blockIdx.x;
//                        *(storeAt+1) = blockIdx.x;
//                        *(storeAt + k+2)= blockIdx.x;
//                        *(storeAt+k+2+1) = blockIdx.x;
//                    }
//
//                    *(storeAt + threadIdx.x+2) = -1;
//                    *(storeAt + k+2+threadIdx.x+2) = *sumx;
//                }

                if(threadIdx.x==0)
                {
                    while(Arrayin[reduceWithBlockIdx]!=goalVal);
                }
                __syncthreads();                                                                                     // All threads wait for the block with which we have to reduce to finish its own global reduction

                if(threadIdx.x < k)                                                                                  // Perform reduction for each cluster
                {
                    const unsigned int reduceWithStorageIndex = (reduceWithBlockIdx>>1)*k + threadIdx.x;
                    clock_t read_time=clock();

                    *sumx=(*sumx)+d_sumx[reduceWithStorageIndex];
                    *sumy=(*sumy)+d_sumy[reduceWithStorageIndex];
                    *count=*count+d_count[reduceWithStorageIndex];

//                    int *storeAt;
//                    storeAt = d_reducedCounts + ((k+2)<<1)*blockIdx.x/(reductionDist<<1);
//                    storeAt += ((k+2)<<1)*blockIdx.x/(reductionDist<<1) ;
//
//                    if(threadIdx.x == 0)
//                    {
//                        *storeAt = blockIdx.x;
//                        *(storeAt+1) = reduceWithBlockIdx;
//                        *(storeAt + k+2)= blockIdx.x;
//                        *(storeAt+k+2+1) = reduceWithBlockIdx;
//                        if(blockIdx.x==0 && reduceWithBlockIdx==2)
//                            d_timeVar[0]=read_time;
//                    }
//
//                    *(storeAt + threadIdx.x+2) = reduceWithStorageIndex;
//                    *(storeAt + k+2+threadIdx.x+2) = t1;

                }
#if TRY_CACHE_BW_DEV_AND_SHARED == 1
                __syncthreads();
#endif
                blocks=blocks - (blocks>>1);
                reductionDist=reductionDist<<1;
                continue;
            }
        }

        {
            // I didnt reduce with anyone. So its time I store my own value and wait for first block to finish which will
            // indicate that whole reduction is finished
#if TRY_CACHE_BW_DEV_AND_SHARED == 1
            __syncthreads();
#endif
            if(threadIdx.x < k)                                                                                   // Store the reduced values in the global mem.
            {
                d_sumx[storageIndex]=*sumx;
                d_sumy[storageIndex]=*sumy;
                d_count[storageIndex]=*count;
            }

            __threadfence();                                                                             // threadfence has to be b4 syncthreads so that all threads have made public their changes.
            __syncthreads();                                                                             // Ensure sum values are seen by other blocks b4 Arrayin is seen by them.
//            clock_t update_time=clock();

//            if(threadIdx.x < k)
//            {
//                int *storeAt;
//                storeAt = d_reducedCounts + ((k+2)<<1)*blockIdx.x/reductionDist;
//                if(reductionDist>1)
//                {
//                    storeAt += ((k+2)<<1)*blockIdx.x/reductionDist + ((k+2)<<1);
//                }
//
//                if(threadIdx.x == 0)
//                {
//                    *storeAt = blockIdx.x;
//                    *(storeAt+1) = blockIdx.x;
//                    *(storeAt + k+2)= blockIdx.x;
//                    *(storeAt+k+2+1) = blockIdx.x;
//                    if(blockIdx.x==2)
//                        d_timeVar[1]=update_time;
//                }
//
//                *(storeAt + threadIdx.x+2) = storageIndex;
//                *(storeAt + k+2+threadIdx.x+2) = *sumx;
//            }

            if(threadIdx.x==0)
            {
                Arrayin[blockIdx.x]=goalVal;
            }
            break;

        }
    }
}



__device__ void inline reduceBlocks1(const int k,const int goalVal,unsigned int blocks,const unsigned int storageIndex,
                                     float *const __restrict__ sumx,float *const __restrict__ sumy,int *const __restrict__ count,volatile float *const __restrict__ d_sumx,
                                     volatile float *const __restrict__ d_sumy, volatile int *const __restrict__ d_count,unsigned int reductionDist,volatile int *const __restrict__ Arrayin,
                                     int *const __restrict__ d_dbgIter, const int max_iter, const int logBase2Blocks)
{
    // Perform reduction till you have no one left on your right to reduce with.
    // At that stage store your own reduced values in global mem. and return.
    // Only blockIdx 0, wont store its reduced values as it will directly use them to find centroids.
    // Hence, we wait till blockIdx 0 is the only block left in this loop.
    int loops=0;
//    blocks=4;
//    bool continueWhile=false;
    while(blocks > 1)
    {

        if((blockIdx.x & ((reductionDist<<1) -1)) == 0)//% (reductionDist<<1) == 0 )                                                                       // (x % 2^k == 0) is same as (x & ((2^k)-1) == 0)
        {
            unsigned int reduceWithBlockIdx=blockIdx.x+reductionDist;
            if(reduceWithBlockIdx < gridDim.x)
            {

                if(threadIdx.x==0)
                {
                    while(Arrayin[reduceWithBlockIdx]!=goalVal);
                    d_dbgIter[blockIdx.x*(max_iter*logBase2Blocks*4) + (goalVal*logBase2Blocks*4) + loops*4 + 0]=reduceWithBlockIdx;
                    d_dbgIter[blockIdx.x*(max_iter*logBase2Blocks*4) + (goalVal*logBase2Blocks*4) + loops*4 + 1]=(reduceWithBlockIdx>>1)*k + threadIdx.x;
                    d_dbgIter[blockIdx.x*(max_iter*logBase2Blocks*4) + (goalVal*logBase2Blocks*4) + loops*4 + 2]=d_count[(reduceWithBlockIdx>>1)*k + threadIdx.x];
                    d_dbgIter[blockIdx.x*(max_iter*logBase2Blocks*4) + (goalVal*logBase2Blocks*4) + loops*4 + 3]=*count;
                    loops++;
                }
                __syncthreads();                                                                                     // All threads wait for the block with which we have to reduce to finish its own global reduction

                if(threadIdx.x < k)                                                                                  // Perform reduction for each cluster
                {
                    const unsigned int reduceWithStorageIndex = (reduceWithBlockIdx>>1)*k + threadIdx.x;
                    *sumx=*sumx+d_sumx[reduceWithStorageIndex];
                    *sumy=*sumy+d_sumy[reduceWithStorageIndex];
                    *count=*count+d_count[reduceWithStorageIndex];
                }
#if TRY_CACHE_BW_DEV_AND_SHARED == 1
                __syncthreads();
#endif
                blocks=blocks - (blocks>>1);
                reductionDist=reductionDist<<1;
                continue;//continueWhile=true;//
            }
        }

//        if(!continueWhile && (blockIdx.x!=0))
        {
            // I didnt reduce with anyone. So its time I store my own value and wait for first block to finish which will
            // indicate that whole reduction is finished
#if TRY_CACHE_BW_DEV_AND_SHARED == 1
            __syncthreads();
#endif
            if(threadIdx.x < k)                                                                                   // Store the reduced values in the global mem.
            {
                d_sumx[storageIndex]=*sumx;
                d_sumy[storageIndex]=*sumy;
                d_count[storageIndex]=*count;
            }

            __syncthreads();                                                                                      // Make sure all threads have stored the reduced vale b4 we declare that the block is done.
            __threadfence();                                                                                      // Ensure sum values are seen by other blocks b4 Arrayin is seen by them.
            if(threadIdx.x==0)
            {
                Arrayin[blockIdx.x]=goalVal;
                d_dbgIter[blockIdx.x*(max_iter*logBase2Blocks*4) + (goalVal*logBase2Blocks*4) + loops*4 + 0]=0;
                d_dbgIter[blockIdx.x*(max_iter*logBase2Blocks*4) + (goalVal*logBase2Blocks*4) + loops*4 + 3]=*count;
//                d_dbgIter[blockIdx.x*(max_iter<<1) + (goalVal<<1)]=goalVal;
            }

            break;//blocks=0;//

        }
//        else
//          continueWhile=false;
    }
}



//GPU lock-free synchronization function
// Function is not in working condition.
// See the TODO for modifying recursive implementation below.
__device__ void gpu_sync(const int goalVal,const unsigned int globalIndex, const int level, int *const __restrict__ Arrayin, int *const __restrict__ Arrayout)
{
// thread ID in a block
    int checkForBlock = globalIndex;
    int nBlockNum = gridDim.x;
    int bid = blockIdx.x;

    for(int i=level; i>1; i--)
    {
        bid+=nBlockNum;                                                                                       // When optimizing with start address as multiple of 32, pass a ptr array pointing to first addr of each level
        nBlockNum=((nBlockNum-1) >> LOG_BLOCKDIM) + 1;
    }

    // only thread 0 is used for synchronization
    if (threadIdx.x == 0)
    {
        Arrayin[bid] = goalVal;
    }

    // If my globalIndex is less than no. of blocks then synchronize for block no.:globalIndex
    if (globalIndex < nBlockNum)
    {
        checkForBlock+=bid-blockIdx.x;
        while (Arrayin[checkForBlock] != goalVal)
        {
            // BUSY WAIT
        }
    }
    __syncthreads();

    // If this was not the only block doing synchronization then ensure others have also finished.
    if (nBlockNum > BLOCKDIM)
    {
        //TODO: recursion not allowed. Make it iterative.
        //__gpu_sync(goalVal,globalIndex,level+1,Arrayin,Arrayout);

    }

    if (globalIndex < nBlockNum)
    {
        Arrayout[checkForBlock] = goalVal;
    }

    if (threadIdx.x == 0)
    {
        while (Arrayout[bid] != goalVal)
        {
            // BUSY WAIT
        }
    }
    __syncthreads();

}



__device__ void inline reduceBlocks_and_storeNewCentroids(const int k,const int numIter,unsigned int blocks,const unsigned int storageIndex,
        float *const __restrict__ sumx,float *const __restrict__ sumy,int *const __restrict__ count,
        float *const __restrict__ d_sumx, float *const __restrict__ d_sumy, int *const __restrict__ d_count,
        volatile int *const __restrict__ d_syncArr, float *const __restrict__ d_centroidx,
        float *const __restrict__ d_centroidy)//, int *d_reducedCounts, clock_t *const d_timeVar)
{
    unsigned int reducnDist=1;
    reduceBlocks(k,numIter,gridDim.x,storageIndex,sumx,sumy,count,d_sumx,d_sumy,d_count,reducnDist,d_syncArr);//, d_reducedCounts, d_timeVar);//,d_dbgIter,*max_iter);  // Local reduction done for all centroids.

    //Global reduction done for all blocks. Now store the new centroid values in global memory
    if(blockIdx.x==0)
    {
        if(threadIdx.x < k)
        {
            if(*count != 0)
            {
#if DONT_CHNG_CENTROIDS == 0
                d_centroidx[threadIdx.x] = __fdividef( *sumx, __int2float_rn(*count));
                d_centroidy[threadIdx.x] = __fdividef( *sumy, __int2float_rn(*count));
#endif
            }

//            int valuesets=gridDim.x;
//            for( int blocks=gridDim.x; blocks>1; blocks-=(blocks>>1))
//                valuesets+=(blocks>>1)<<1;
//            valuesets=(valuesets-1)<<1;
//
//            int *storeAt;
//            storeAt = d_reducedCounts+valuesets*(k+2);// + (k+1)*blockIdx.x/reductionDist;
//
//            if(threadIdx.x == 0)
//            {
//                *storeAt = blockIdx.x;
//                *(storeAt+1)=blockIdx.x;
//                *(storeAt+k+2)=blockIdx.x;
//                *(storeAt+k+2+1)=blockIdx.x;
//            }
//
//            *(storeAt + threadIdx.x+2) = threadIdx.x;
//            *(storeAt + k+2+threadIdx.x+2) = *sumx;
//
//
        }

        __threadfence();                                                                                      //Ensure sum values are seen by other blocks b4 Arrayin is seen by them.
        __syncthreads();

        // Declare reduction done by thread with greatest index so that if no. of centroids is less than threads,
        // then by the time last thread sets the global value other threads from other warps would load the new
        // centroid values into the shared memory.
        if(threadIdx.x == BLOCKDIM -1)
        {
            d_syncArr[0]=numIter;                                                                                   // Declares global reduction over when it has stored new centroids in global mem.
//              d_dbgIter[numIter<<1]=d_dbgIter[(numIter<<1)+1]=numIter;
        }
        //No need for syncthreads() as other threads can go on and load the new centroid values into shared mem. from device mem.
    }
    else
    {
//            if(blockIdx.x==2)
//             {
//               d_syncArr[2]=7; 20
//               __threadfence();
//             }
        if(threadIdx.x==0)
        {
            while(d_syncArr[0]!=numIter);                                                                           // All blocks wait for blockIdx 0, to store new centroids.
//              d_dbgIter[blockIdx.x*((*max_iter)<<1) + (numIter<<1)+1]=numIter;
        }
        __syncthreads();                                                                                          // One threads checks value from global and others just wait for it.
    }
    //Central Barrier synchronization between all blocks is not required as all threads simply wait for thread 0 to finish.
    //gpu_sync(numIter,index,1,d_syncArrIn, d_syncArrOut);

}


// Right now this code only works when
// 1. Number of threads >= k
// 2. Number of points can be > threads*blocks, i.e each thread may have to process more than one point till the shared mem allows.
// 3. All the centroids can be accomodated in shared mem. at once.
__global__ void __launch_bounds__(BLOCKDIM, MIN_NUM_BLOCKS_PER_MULTIPROC)
cluster(int n, int k,int *const __restrict__ max_iter, const float *const __restrict__ d_dataptx, const float *const __restrict__ d_datapty,
        float *const __restrict__ d_centroidx, float *const __restrict__ d_centroidy, int *const __restrict__ d_syncArr,
        float *const __restrict__ d_sumx, float *const __restrict__ d_sumy, int *const __restrict__ d_count,int *const __restrict__ d_clusterno)//, int *const d_reducedCounts, clock_t *const d_timeVar)//, int *const d_dbgIter)
{
    // shared memory
    // Stores x-coord of all clusters followed by y-coord of all clusters and so on.
    extern __shared__ float s_centroid[];

    // No point of going for short int as each bank in shared mem. is 32-bit wide and only one thread
    // can access one bank at a time, so no. of accesses wont be reduced
    __shared__ signed int s_thrd_k[(BLOCKDIM)*POINTS_PER_THREAD];                                           // Store the nearest cluster of point checked by this thrd.

    __shared__ float s_thrd[(BLOCKDIM<<1)*POINTS_PER_THREAD];                                               // Store the x-coord of data point processed, for all thrds followed by y-coord.

    float *const s_centroidx = s_centroid;                                                                  // x-coord of 1st cluster's centroid
    float *const s_centroidy = s_centroid+k;                                                                // y-coord of 1st cluster's centroid

    float *const s_thrd_x = s_thrd;                                                                         // x-coord of points checked by thrds
    float *const s_thrd_y = s_thrd+(POINTS_PER_THREAD<<LOG_BLOCKDIM);                                       // y-coord of points checked by thrds


  #if POINTS_PER_THREAD == 1
      float ptx,pty;
  #else
      float ptx[POINTS_PER_THREAD],pty[POINTS_PER_THREAD];
  #endif

    // If we've T threads per block then index of ith thread in kth block is k*T + i. So we are maintaining an order among all threads of all blocks.
    // This is same as blockIdx.x * BLOCKDIM + threadId.x
    const unsigned int index = (POINTS_PER_THREAD << LOG_BLOCKDIM)*blockIdx.x + threadIdx.x;

  #if POINTS_PER_THREAD == 1
    load_data_points(n,&ptx, &pty, index, d_dataptx, d_datapty, s_thrd_x, s_thrd_y);                            //Number of points to be processed by this thread.
  #else
    const int my_point_num = load_data_points(n,ptx, pty, index, d_dataptx, d_datapty, s_thrd_x, s_thrd_y);     //Number of points to be processed by this thread.
  #endif

    bool repeat=true;
    int numIter;
    for ( numIter = 0; repeat && (numIter < *max_iter); ++numIter )
    {
        load_centroids_in_shared(k, s_centroidx, s_centroidy, d_centroidx, d_centroidy);
        __syncthreads();


        // Find closest centroid if this thread holds a data-point and set it in shared mem.
  #if POINTS_PER_THREAD == 1
          store_nearest_in_shared(n, k, ptx, pty, index, s_centroidx, s_centroidy, s_thrd_k);
  #else
          store_nearest_in_shared(k, ptx, pty, s_centroidx, s_centroidy, s_thrd_k, my_point_num);
  #endif
        __syncthreads();                                                                                    // Wait for all thrds to finish setting closest k value


        // Do parallel reduction with each thread doing reduction for a distinct centroid.
        const unsigned int storageIndex = (blockIdx.x>>1)*k + threadIdx.x;                                        // The sumx, sumy and count value will be stored at this index in global mem.
        int count=0;
        float sumx=0;
        float sumy =0;

        count += reduceThreads(k, &sumx, &sumy, s_thrd_x, s_thrd_y, s_thrd_k);

        reduceBlocks_and_storeNewCentroids(k,numIter,gridDim.x,storageIndex,&sumx,&sumy,&count,d_sumx,d_sumy,d_count,d_syncArr, d_centroidx, d_centroidy);//, d_reducedCounts, d_timeVar);//,d_dbgIter,*max_iter);
    }

    // Final centroids have been found. Now set the cluster no. in d_clusterno array in device memory.
    {
        load_centroids_in_shared(k, s_centroidx, s_centroidy, d_centroidx, d_centroidy);
        __syncthreads();

        // Find closest centroid if this thread holds a data-point
  #if POINTS_PER_THREAD == 1
        store_nearest_in_global(n, k, &ptx, &pty, index, s_centroidx, s_centroidy, d_clusterno);
  #else
        store_nearest_in_global(k, ptx, pty, index, s_centroidx, s_centroidy, d_clusterno, my_point_num);
  #endif

    }

    if(blockIdx.x == 0 && threadIdx.x == 0)
        *max_iter = numIter;
}

// Right now this code only works when
// 1. Number of threads >= k
// 2. Number of points can be > threads*blocks, i.e each thread may have to process more than one point till the shared mem allows.
// 3. All the centroids can be accomodated in shared mem. at once.
__global__ void __launch_bounds__(BLOCKDIM, MIN_NUM_BLOCKS_PER_MULTIPROC)
cluster1(int n, int k,int *const __restrict__ max_iter, const float *const __restrict__ d_dataptx, const float *const __restrict__ d_datapty,
         float *const __restrict__ d_centroidx, float *const __restrict__ d_centroidy, int *const __restrict__ d_syncArr,
         float *const __restrict__ d_sumx, float *const __restrict__ d_sumy, int *const __restrict__ d_count,int *const __restrict__ d_clusterno)//, int *const d_reducedCounts, clock_t *const d_timeVar)//, int *const d_dbgIter)
{
    // shared memory
    // Stores x-coord of all clusters followed by y-coord of all clusters i.e. x and y coord of a cluster are separated by k.
    extern __shared__ float s_centroid[];

    // No point of going for short int as each bank in shared mem. is 32-bit wide and only one thread
    // can access one bank at a time, so no. of accesses wont be reduced
    __shared__ signed int s_thrd_k[(BLOCKDIM)*POINTS_PER_THREAD];                                           // Store the nearest cluster of point checked by this thrd.

    __shared__ float s_thrd[(BLOCKDIM<<1)*POINTS_PER_THREAD];                                               // Store the x-coord of data point processed, for all thrds followed by y-coord.

    float *const s_centroidx = s_centroid;                                                                  // x-coord of 1st cluster's centroid
    float *const s_centroidy = s_centroid+k;                                                                // y-coord of 1st cluster's centroid

    float *const s_thrd_x = s_thrd;                                                                         // x-coord of points checked by thrds
    float *const s_thrd_y = s_thrd+(POINTS_PER_THREAD<<LOG_BLOCKDIM);                                       // y-coord of points checked by thrds


  #if POINTS_PER_THREAD == 1
    float ptx,pty;
  #else
    float ptx[POINTS_PER_THREAD],pty[POINTS_PER_THREAD];
  #endif

    const unsigned int total_point_cycles=(n-1)/((POINTS_PER_THREAD<<LOG_BLOCKDIM)*gridDim.x) + 1;

    bool repeat=true;
    int numIter;
    for ( numIter = 0; repeat && (numIter < *max_iter); ++numIter )
    {
        load_centroids_in_shared(k, s_centroidx, s_centroidy, d_centroidx, d_centroidy);

        __syncthreads();

        float sumx=0;
        float sumy =0;
        int count=0;
        const unsigned int storageIndex = (blockIdx.x>>1)*k + threadIdx.x;                         // The sumx, sumy and count value will be stored at this index in global mem.
        for( int curr_point_cycle=0; curr_point_cycle < total_point_cycles; curr_point_cycle++)
        {

            // If we've T threads per block then index of ith thread in kth block is k*T + i. So we are maintaining an order among all threads of all blocks.
            // This is same as blockIdx.x * BLOCKDIM + threadId.x
            unsigned int index = (POINTS_PER_THREAD << LOG_BLOCKDIM)*(curr_point_cycle*gridDim.x + blockIdx.x) + threadIdx.x;

  #if POINTS_PER_THREAD == 1
            load_data_points(n,&ptx, &pty, index, d_dataptx, d_datapty, s_thrd_x, s_thrd_y);                            //Number of points to be processed by this thread.
  #else
            int my_point_num = load_data_points(n,curr_point_cycle, total_point_cycles, ptx, pty, index, d_dataptx, d_datapty, s_thrd_x, s_thrd_y);     //Number of points to be processed by this thread.
  #endif


            // Find closest centroid if this thread holds a data-point and set it in shared mem.
  #if POINTS_PER_THREAD == 1
            store_nearest_in_shared(n, k, ptx, pty, index, s_centroidx, s_centroidy, s_thrd_k);
  #else
            store_nearest_in_shared(k, ptx, pty, s_centroidx, s_centroidy, s_thrd_k, my_point_num);
  #endif
            __syncthreads();                                                                                    // Wait for all thrds to finish setting closest k value

            // Do parallel reduction with each thread doing reduction for a distinct centroid.
            count += reduceThreads(k, &sumx, &sumy, s_thrd_x, s_thrd_y, s_thrd_k);

        }

        reduceBlocks_and_storeNewCentroids(k,numIter,gridDim.x,storageIndex,&sumx,&sumy,&count,d_sumx,d_sumy,d_count,d_syncArr, d_centroidx, d_centroidy);//, d_reducedCounts, d_timeVar);//,d_dbgIter,*max_iter);

    }

    // Iterations are over and final centroids have been found. Now set the cluster no. in d_clusterno array in device memory.
    {
        load_centroids_in_shared(k, s_centroidx, s_centroidy, d_centroidx, d_centroidy);
        __syncthreads();
        for( int curr_point_cycle=0; curr_point_cycle < total_point_cycles; curr_point_cycle++)
        {
            // If we've T threads per block then index of ith thread in kth block is k*T + i. So we are maintaining an order among all threads of all blocks.
            // This is same as blockIdx.x * BLOCKDIM + threadId.x
            unsigned int index = (POINTS_PER_THREAD << LOG_BLOCKDIM)*(curr_point_cycle*gridDim.x + blockIdx.x) + threadIdx.x;

  #if POINTS_PER_THREAD == 1
            load_data_points(n,&ptx, &pty, index, d_dataptx, d_datapty, s_thrd_x, s_thrd_y);                            //Number of points to be processed by this thread.
  #else
            const int my_point_num = load_data_points(n,ptx, pty, index, d_dataptx, d_datapty, s_thrd_x, s_thrd_y);     //Number of points to be processed by this thread.
  #endif

            // Find closest centroid if this thread holds a data-point
  #if POINTS_PER_THREAD == 1
            store_nearest_in_global(n, k, &ptx, &pty, index, s_centroidx, s_centroidy, d_clusterno);
  #else
            store_nearest_in_global(k, ptx, pty, index, s_centroidx, s_centroidy, d_clusterno, my_point_num);
  #endif
        }

    }

    if(blockIdx.x == 0 && threadIdx.x == 0)
        *max_iter = numIter;
}

__device__ int inline init_data(const int curr_point_cycle, const int total_point_cycles, const int n, float *const __restrict__ ptx,
                                float *const __restrict__ pty, unsigned int *const __restrict__ index, const float *const __restrict__ d_dataptx,
                                const float *const __restrict__ d_datapty, float *const __restrict__ s_thrd_x, float *const __restrict__ s_thrd_y)
{

    // If we've T threads per block then index of ith thread in kth block is k*T + i. So we are maintaining an order among all threads of all blocks.
    // This is same as blockIdx.x * BLOCKDIM + threadId.x
    *index = (POINTS_PER_THREAD << LOG_BLOCKDIM)*(curr_point_cycle*gridDim.x + blockIdx.x) + threadIdx.x;

  #if POINTS_PER_THREAD == 1
    int my_point_num = load_data_points(n,ptx, pty, *index, d_dataptx, d_datapty, s_thrd_x, s_thrd_y);                            //Number of points to be processed by this thread.
  #else
    int my_point_num = load_data_points(n,curr_point_cycle, total_point_cycles, ptx, pty, *index, d_dataptx, d_datapty, s_thrd_x, s_thrd_y);     //Number of points to be processed by this thread.
  #endif

    return my_point_num;
}


__device__ inline void reduceThreads_and_setCentroidVars(const int n, const int k, float *const __restrict__ ptx, float *const __restrict__ pty, float *const __restrict__ s_centroidx,
                                                         float *const __restrict__ s_centroidy, float *const __restrict__ s_thrd_x, float *const __restrict__ s_thrd_y,
                                                         int *const __restrict__ s_thrd_k, const unsigned int *const __restrict__ index, const int *const __restrict__ my_point_num,
                                                         float *const __restrict__ sumx, float *const __restrict__ sumy, int *const __restrict__ count)
{
    // Find closest centroid if this thread holds a data-point and set it in shared mem.
  #if POINTS_PER_THREAD == 1
    store_nearest_in_shared(n, k, *ptx, *pty, *index, s_centroidx, s_centroidy, s_thrd_k);
  #else
    store_nearest_in_shared(k, ptx, pty, s_centroidx, s_centroidy, s_thrd_k, *my_point_num);
  #endif
    __syncthreads();                                                                                    // Wait for all thrds to finish setting closest k value

    // Do parallel reduction with each thread doing reduction for a distinct centroid.
    *count += reduceThreads(k, sumx, sumy, s_thrd_x, s_thrd_y, s_thrd_k);
}



__device__ void inline init_centroids(const int k, float *const __restrict__ s_centroidx, float *const __restrict__ s_centroidy,
                                      const float *const __restrict__ d_centroidx, const float *const __restrict__ d_centroidy,
                                      float *const __restrict__ sumx, float *const __restrict__ sumy, int *const __restrict__ count)
{

    *sumx=0;
    *sumy =0;
    load_centroids_in_shared(k, s_centroidx, s_centroidy, d_centroidx, d_centroidy);
    *count=0;
    __syncthreads();

}

// Right now this code only works when
// 1. Number of threads >= k
// 2. Number of points can be > threads*blocks, i.e each thread may have to process more than one point till the shared mem allows.
// 3. All the centroids can be accomodated in shared mem. at once.
__global__ void __launch_bounds__(BLOCKDIM, MIN_NUM_BLOCKS_PER_MULTIPROC)
cluster1_load_optimized(int n, int k,int *const __restrict__ max_iter, const float *const __restrict__ d_dataptx, const float *const __restrict__ d_datapty,
                       float *const __restrict__ d_centroidx, float *const __restrict__ d_centroidy, int *const __restrict__ d_syncArr,
                       float *const __restrict__ d_sumx, float *const __restrict__ d_sumy, int *const __restrict__ d_count,int *const __restrict__ d_clusterno)//, int *const d_reducedCounts, clock_t *const d_timeVar)//, int *const d_dbgIter)
{
    // shared memory
    // Stores x-coord of all clusters followed by y-coord of all clusters i.e. x and y coord of a cluster are separated by k.
    extern __shared__ float s_centroid[];

    // No point of going for short int as each bank in shared mem. is 32-bit wide and only one thread
    // can access one bank at a time, so no. of accesses wont be reduced
    __shared__ signed int s_thrd_k[(BLOCKDIM)*POINTS_PER_THREAD];                                           // Store the nearest cluster of point checked by this thrd.

    __shared__ float s_thrd[(BLOCKDIM<<1)*POINTS_PER_THREAD];                                               // Store the x-coord of data point processed, for all thrds followed by y-coord.

    float *const s_centroidx = s_centroid;                                                                  // x-coord of 1st cluster's centroid
    float *const s_centroidy = s_centroid+k;                                                                // y-coord of 1st cluster's centroid

    float *const s_thrd_x = s_thrd;                                                                         // x-coord of points checked by thrds
    float *const s_thrd_y = s_thrd+(POINTS_PER_THREAD<<LOG_BLOCKDIM);                                       // y-coord of points checked by thrds

  #if POINTS_PER_THREAD == 1
    float ptx,pty;
  #else
    float ptx[POINTS_PER_THREAD],pty[POINTS_PER_THREAD];
  #endif

    const unsigned int total_point_cycles=(n-1)/((POINTS_PER_THREAD<<LOG_BLOCKDIM)*gridDim.x) + 1;

    bool repeat=true;
    int numIter=0;

    int count=0;
    const unsigned int storageIndex = (blockIdx.x>>1)*k + threadIdx.x;                            // The sumx, sumy and count value will be stored at this index in global mem.

    int curr_point_cycle=0;
    unsigned int index;

    float sumx=0;
    float sumy =0;
  #if POINTS_PER_THREAD == 1
    int my_point_num = init_data(curr_point_cycle, total_point_cycles, n, &ptx, &pty, &index, d_dataptx, d_datapty, s_thrd_x, s_thrd_y);
  #else
    int my_point_num = init_data(curr_point_cycle, total_point_cycles, n, ptx, pty, &index, d_dataptx, d_datapty, s_thrd_x, s_thrd_y);
  #endif

    for ( numIter = 0; repeat && (numIter < *max_iter); ++numIter )
    {
        {
            // ************ 0th, 2nd and so on all even iterations ************
            init_centroids(k, s_centroidx, s_centroidy, d_centroidx, d_centroidy, &sumx, &sumy, &count);
            // We already have points loaded for first point_cycle
  #if POINTS_PER_THREAD == 1
            reduceThreads_and_setCentroidVars(n, k, &ptx, &pty, s_centroidx, s_centroidy, s_thrd_x, s_thrd_y, s_thrd_k, &index, &my_point_num, &sumx, &sumy, &count);
  #else
            reduceThreads_and_setCentroidVars(n, k, ptx, pty, s_centroidx, s_centroidy, s_thrd_x, s_thrd_y, s_thrd_k, &index, &my_point_num, &sumx, &sumy, &count);
  #endif


            for( curr_point_cycle=1; curr_point_cycle < total_point_cycles; curr_point_cycle++)
            {
  #if POINTS_PER_THREAD == 1
                init_data(curr_point_cycle, total_point_cycles, n, &ptx, &pty, &index, d_dataptx, d_datapty, s_thrd_x, s_thrd_y);
                reduceThreads_and_setCentroidVars(n, k, &ptx, &pty, s_centroidx, s_centroidy, s_thrd_x, s_thrd_y, s_thrd_k, &index, &my_point_num, &sumx, &sumy, &count);
  #else
                int my_point_num = init_data(curr_point_cycle, total_point_cycles, n, ptx, pty, &index, d_dataptx, d_datapty, s_thrd_x, s_thrd_y);
                reduceThreads_and_setCentroidVars(n, k, ptx, pty, s_centroidx, s_centroidy, s_thrd_x, s_thrd_y, s_thrd_k, &index, &my_point_num, &sumx, &sumy, &count);
  #endif
            }
            reduceBlocks_and_storeNewCentroids(k,numIter,gridDim.x,storageIndex,&sumx,&sumy,&count,d_sumx,d_sumy,d_count,d_syncArr, d_centroidx, d_centroidy);//, d_reducedCounts, d_timeVar);//,d_dbgIter,*max_iter);
        }

        if(++numIter < *max_iter && repeat)
        {
            // ************ 1st, 3rd and so on all odd iterations ************

            init_centroids(k, s_centroidx, s_centroidy, d_centroidx, d_centroidy, &sumx, &sumy, &count);
            // We already have points loaded for last point_cycle
  #if POINTS_PER_THREAD == 1
            reduceThreads_and_setCentroidVars(n, k, &ptx, &pty, s_centroidx, s_centroidy, s_thrd_x, s_thrd_y, s_thrd_k, &index, &my_point_num, &sumx, &sumy, &count);
  #else
            reduceThreads_and_setCentroidVars(n, k, ptx, pty, s_centroidx, s_centroidy, s_thrd_x, s_thrd_y, s_thrd_k, &index, &my_point_num, &sumx, &sumy, &count);
  #endif
            for( curr_point_cycle=total_point_cycles-2; curr_point_cycle >= 0; curr_point_cycle--)
            {
  #if POINTS_PER_THREAD == 1
                init_data(curr_point_cycle, total_point_cycles, n, &ptx, &pty, &index, d_dataptx, d_datapty, s_thrd_x, s_thrd_y);
                reduceThreads_and_setCentroidVars(n, k, &ptx, &pty, s_centroidx, s_centroidy, s_thrd_x, s_thrd_y, s_thrd_k, &index, &my_point_num, &sumx, &sumy, &count);
  #else
                int my_point_num = init_data(curr_point_cycle, total_point_cycles, n, ptx, pty, &index, d_dataptx, d_datapty, s_thrd_x, s_thrd_y);
                reduceThreads_and_setCentroidVars(n, k, ptx, pty, s_centroidx, s_centroidy, s_thrd_x, s_thrd_y, s_thrd_k, &index, &my_point_num, &sumx, &sumy, &count);
  #endif
            }

            reduceBlocks_and_storeNewCentroids(k,numIter,gridDim.x,storageIndex,&sumx,&sumy,&count,d_sumx,d_sumy,d_count,d_syncArr, d_centroidx, d_centroidy);//, d_reducedCounts, d_timeVar);//,d_dbgIter,*max_iter);

        }

    }

    // Iterations are over and final centroids have been found. Now set the cluster no. in d_clusterno array in device memory.
    {
        load_centroids_in_shared(k, s_centroidx, s_centroidy, d_centroidx, d_centroidy);
        __syncthreads();

        // Store nearest in global for last loaded point.
        // Find closest centroid if this thread holds a data-point
  #if POINTS_PER_THREAD == 1
        store_nearest_in_global(n, k, &ptx, &pty, index, s_centroidx, s_centroidy, d_clusterno);
  #else
        store_nearest_in_global(k, ptx, pty, index, s_centroidx, s_centroidy, d_clusterno, my_point_num);
  #endif

        // Assuming last loaded points were of first point cycle
        int nextCycle=1;
        int goTill=total_point_cycles;

        if(curr_point_cycle==total_point_cycles)
        {
            // Last loaded points were of last point cycle
            nextCycle=0;
            goTill=total_point_cycles-1;
        }

        for( int curr_point_cycle=nextCycle; curr_point_cycle < goTill; curr_point_cycle++)
        {
  #if POINTS_PER_THREAD == 1
            init_data(curr_point_cycle, total_point_cycles, n, &ptx, &pty, &index, d_dataptx, d_datapty, s_thrd_x, s_thrd_y);
            // Find closest centroid if this thread holds a data-point
            store_nearest_in_global(n, k, &ptx, &pty, index, s_centroidx, s_centroidy, d_clusterno);
  #else
            init_data(curr_point_cycle, total_point_cycles, n, ptx, pty, &index, d_dataptx, d_datapty, s_thrd_x, s_thrd_y);
            // Find closest centroid if this thread holds a data-point
            store_nearest_in_global(k, ptx, pty, index, s_centroidx, s_centroidy, d_clusterno, my_point_num);
  #endif
        }

    }

    if(blockIdx.x == 0 && threadIdx.x == 0)
        *max_iter = numIter;
}

//
//// Right now this code only works when
//// 1. Number of threads >= k
//// 2. Number of points <= threads*blocks, so that each thread needs to process just one point.
//
//__global__ void /*__launch_bounds__(256, 2)*/
//testGlblReductnForCluster(int n, int k,int *const __restrict__ max_iter, const float *const __restrict__ d_dataptx, const float *const __restrict__ d_datapty,
//                          float *const __restrict__ d_reducex, float *const __restrict__ d_reducey, int *const __restrict__ d_reduceCount, int *const __restrict__ d_syncArr,
//                          float *const __restrict__ d_sumx, float *const __restrict__ d_sumy, int *const __restrict__ d_count, int *const __restrict__ d_dbgIter, int logBase2Blocks)
//{
//
//    __shared__ float s_thrd[BLOCKDIM<<1];                                                                 // Store the x-coord of data point processed, for all thrds followed by y-coord.
//
//
//    float ptx,pty;
//    ptx=pty=0.0f;
//    // If we've T threads per block then index of ith thread in kth block is k*T + i. So we are maintaining an order among all threads of all blocks.
//    // This is same as blockIdx.x * blockDim.x + threadId.x
//    const unsigned int index = (blockIdx.x << LOG_BLOCKDIM) + threadIdx.x;
//#if TRY_CACHE_BW_DEV_AND_SHARED == 1
//    __syncthreads();
//#endif
//    if (index <n )
//    {
//        ptx=d_dataptx[index];
//        pty=d_datapty[index];
//    }
//#if TRY_CACHE_BW_DEV_AND_SHARED == 1
//    __syncthreads();
//#endif
//
//    const unsigned int storageIndex=(blockIdx.x>>1)*k + threadIdx.x;                                             // The sumx, sumy and count value will be stored at this index in global mem.
//
//    bool repeat=true;
//    int numIter;
//    for ( numIter = 0; repeat && (numIter < *max_iter); ++numIter )
//    {
//
//        float *const s_thrd_x = s_thrd;                                                                     // x-coord of points checked by thrds
//        float *const s_thrd_y = s_thrd+BLOCKDIM;                                                            // y-coord of points checked by thrds
//
//        // Find closest centroid if this thread holds a data-point
//        s_thrd_x[threadIdx.x]=ptx;
//        s_thrd_y[threadIdx.x]=pty;
//        __syncthreads();                                                                                    // Wait for all thrds to finish setting closest k value
//
//
//        // Do parallel reduction with each thread doing reduction for a distinct centroid.
//        float sumx=0;
//        float sumy =0;
//        int count=0;
//        if (threadIdx.x < k)
//        {
//            for (int i = 0; i < BLOCKDIM; ++i)
//            {
//#if USE_BRDCST == 0
//                int indx=i+threadIdx.x;
//                if (indx >= BLOCKDIM)
//                    indx -= BLOCKDIM;
//                sumx+=s_thrd_x[indx];
//                sumy+=s_thrd_y[indx];
//                count++;
//#else
//                // Broadcast access from shared mem. This line shud be executed by all threads in warp simultaneously.
//                sumx+=s_thrd_x[i];
//                sumy+=s_thrd_y[i];
//                count++;
//#endif
//            }
//        }
//
//        const unsigned int reducnDist=1;
//        reduceBlocks1(k,numIter,gridDim.x,storageIndex,&sumx,&sumy,&count,d_sumx,d_sumy,d_count,reducnDist,d_syncArr,d_dbgIter,*max_iter,logBase2Blocks);  // Local reduction done for all centroids.
//
//
//        //Global reduction done for all blocks. Now store the new centroid values in global memory
//        if(blockIdx.x==0)
//        {
//            if(threadIdx.x < k)
//            {
//                if(count != 0)
//                {
//#if DONT_CHNG_CENTROIDS == 0
//                    d_reducex[(numIter*k)+threadIdx.x] = sumx;//__fdividef( sumx, __int2float_rn(count));
//                    d_reducey[(numIter*k)+threadIdx.x] = sumy;//__fdividef( sumy, __int2float_rn(count));
//                    d_reduceCount[(numIter*k)+threadIdx.x] = count;
//#endif
//                }
//            }
//
//            __syncthreads();
//            __threadfence();                                                                                      //Ensure sum values are seen by other blocks b4 Arrayin is seen by them.
//
//            // Declare reduction finished by thread with greatest index so that if no. of centroids is less than threads,
//            // then by the time last thread sets the global value other threads would load the new centroid values
//            // into the shared memory.
//            if(threadIdx.x == BLOCKDIM -1)
//            {
//                d_syncArr[0]=numIter;                                                                                   // Declares global reduction over when it has stored new centroids in global mem.
////              d_dbgIter[numIter<<1]=d_dbgIter[(numIter<<1)+1]=numIter;
//            }
//            //No need for syncthreads() as other threads can go on and load the new centroid values into shared mem. from device mem.
//        }
//        else
//        {
//            if(threadIdx.x==0)
//            {
//                while(d_syncArr[0]!=numIter);                                                                           // All blocks wait for blockIdx 0, to store new centroids.
////              d_dbgIter[blockIdx.x*((*max_iter)<<1) + (numIter<<1)+1]=numIter;
//            }
//            __syncthreads();                                                                                          // One threads checks value from global and others just wait for it.
//        }
//        //Central Barrier synchronization between all blocks is not required as all threads simply wait for thread 0 to finish.
//        //gpu_sync(numIter,index,1,d_syncArrIn, d_syncArrOut);
//    }
//
//    if(blockIdx.x == 0 && threadIdx.x == 0)
//        *max_iter = numIter;
//}
//
//
//
//__device__ void inline reduceBlocks0(const int k,const int goalVal,unsigned int blocks,const unsigned int storageIndex,
//                                     float *const __restrict__ sumx,float *const __restrict__ sumy,int *const __restrict__ count,volatile float *const __restrict__ d_sumx,
//                                     volatile float *const __restrict__ d_sumy, volatile int *const __restrict__ d_count,unsigned int reductionDist,volatile int *const __restrict__ Arrayin, int *d_reducedCounts, clock_t *const d_timeVar)
//{
//    // Perform reduction till you have no one left on your right to reduce with.
//    // At that stage store your own reduced values in global mem. and return.
//    // Only blockIdx 0, wont store its reduced values as it will directly use them to find centroids.
//    // Hence, we wait till blockIdx 0 is the only block left in this loop.
//    while(blocks > 1)
//    {
//
//        if((blockIdx.x & ((reductionDist<<1) -1)) == 0)//% (reductionDist<<1) == 0 )                                 // (x % 2^k == 0) is same as (x & ((2^k)-1) == 0)
//        {
//            unsigned int reduceWithBlockIdx=blockIdx.x+reductionDist;
//            if(reduceWithBlockIdx < gridDim.x)
//            {
//
//                if(threadIdx.x < k)
//                {
//                    int *storeAt;
//                    storeAt = d_reducedCounts + ((k+2)<<1)*blockIdx.x/reductionDist;
//                    d_reducedCounts += ((k+2)<<1)*blocks;
//                    if(reductionDist>1)
//                    {
//                        storeAt += ((k+2)<<1)*blockIdx.x/reductionDist + (k+2)<<1;
//                        d_reducedCounts += ((k+2)<<1)*blocks;
//                    }
//
//                    if(threadIdx.x == 0)
//                    {
//                        *storeAt = blockIdx.x;
//                        *(storeAt+1) = blockIdx.x;
//                        *(storeAt + k+2)= blockIdx.x;
//                        *(storeAt+k+2+1) = blockIdx.x;
//                    }
//
//                    *(storeAt + threadIdx.x+2) = -1;
//                    *(storeAt + k+2+threadIdx.x+2) = *count;
//                }
//
//                if(threadIdx.x==0)
//                {
//                    while(Arrayin[reduceWithBlockIdx]!=goalVal);
//                }
//                __syncthreads();                                                                                     // All threads wait for the block with which we have to reduce to finish its own global reduction
//
//                if(threadIdx.x < k)                                                                                  // Perform reduction for each cluster
//                {
//                    const unsigned int reduceWithStorageIndex = (reduceWithBlockIdx>>1)*k + threadIdx.x;
//                    clock_t read_time=clock();
//                    float t1=d_sumx[reduceWithStorageIndex];
//                    *sumx=(*sumx)+t1;
//                    *sumy=(*sumy)+d_sumy[reduceWithStorageIndex];
//                    int t2=d_count[reduceWithStorageIndex];
//                    *count=(*count)+t2;
//
//                    int *storeAt;
//                    storeAt = d_reducedCounts + ((k+2)<<1)*blockIdx.x/(reductionDist<<1);
//                    storeAt += ((k+2)<<1)*blockIdx.x/(reductionDist<<1) ;
//
//                    if(threadIdx.x == 0)
//                    {
//                        *storeAt = blockIdx.x;
//                        *(storeAt+1) = reduceWithBlockIdx;
//                        *(storeAt + k+2)= blockIdx.x;
//                        *(storeAt+k+2+1) = reduceWithBlockIdx;
//                        if(blockIdx.x==0 && reduceWithBlockIdx==2)
//                            d_timeVar[0]=read_time;
//                    }
//
//                    *(storeAt + threadIdx.x+2) = reduceWithStorageIndex;
//                    *(storeAt + k+2+threadIdx.x+2) = t2;
//
//                }
//#if TRY_CACHE_BW_DEV_AND_SHARED == 1
//                __syncthreads();
//#endif
//                blocks=blocks - (blocks>>1);
//                reductionDist=reductionDist<<1;
//                continue;
//            }
//        }
//
//        {
//            // I didnt reduce with anyone. So its time I store my own value and wait for first block to finish which will
//            // indicate that whole reduction is finished
//#if TRY_CACHE_BW_DEV_AND_SHARED == 1
//            __syncthreads();
//#endif
//            if(threadIdx.x < k)                                                                                   // Store the reduced values in the global mem.
//            {
////                *sumx=25;
////                *sumy=35;
////                *count=45;
//                d_sumx[storageIndex]=25;//*sumx;
//                d_sumy[storageIndex]=35;//*sumy;
//                d_count[storageIndex]=45;//*count;
//            }
//
//            __threadfence();                                                                             // threadfence has to be b4 syncthreads so that all threads have made public their changes.
//            __syncthreads();                                                                             // Ensure sum values are seen by other blocks b4 Arrayin is seen by them.
//            clock_t update_time=clock();
//
//            if(threadIdx.x < k)
//            {
//                int *storeAt;
//                storeAt = d_reducedCounts + ((k+2)<<1)*blockIdx.x/reductionDist;
//                if(reductionDist>1)
//                {
//                    storeAt += ((k+2)<<1)*blockIdx.x/reductionDist + ((k+2)<<1);
//                }
//
//                if(threadIdx.x == 0)
//                {
//                    *storeAt = blockIdx.x;
//                    *(storeAt+1) = blockIdx.x;
//                    *(storeAt + k+2)= blockIdx.x;
//                    *(storeAt+k+2+1) = blockIdx.x;
//                    if(blockIdx.x==2)
//                        d_timeVar[1]=update_time;
//                }
//
//                *(storeAt + threadIdx.x+2) = storageIndex;
//                *(storeAt + k+2+threadIdx.x+2) = *count;
//            }
//
//            if(threadIdx.x==0)
//            {
//                Arrayin[blockIdx.x]+=goalVal+2;
//            }
//            break;
//
//        }
//    }
//}
//
//
//
//__device__ void inline reduceBlocks_and_storeNewCentroids0(const int k,const int numIter,unsigned int blocks,const unsigned int storageIndex,
//        float *const __restrict__ sumx,float *const __restrict__ sumy,int *const __restrict__ count,
//        float *const __restrict__ d_sumx, float *const __restrict__ d_sumy, int *const __restrict__ d_count,
//        volatile int *const __restrict__ d_syncArr, float *const __restrict__ d_centroidx,
//        float *const __restrict__ d_centroidy, int *d_reducedCounts, clock_t *const d_timeVar)
//{
//    unsigned int reducnDist=1;
//    reduceBlocks0(k,numIter,gridDim.x,storageIndex,sumx,sumy,count,d_sumx,d_sumy,d_count,reducnDist,d_syncArr, d_reducedCounts, d_timeVar);//,d_dbgIter,*max_iter);  // Local reduction done for all centroids.
//
//
//    //Global reduction done for all blocks. Now store the new centroid values in global memory
//    if(blockIdx.x==0)
//    {
//        if(threadIdx.x < k)
//        {
//            if(*count != 0)
//            {
//#if DONT_CHNG_CENTROIDS == 0
//                d_centroidx[threadIdx.x] = *sumx;//__fdividef( *sumx, __int2float_rn(*count));
//                d_centroidy[threadIdx.x] = *sumy;//__fdividef( *sumy, __int2float_rn(*count));
//#endif
//            }
//
//            int valuesets=gridDim.x;
//            for( int blocks=gridDim.x; blocks>1; blocks-=(blocks>>1))
//                valuesets+=(blocks>>1)<<1;
//            valuesets=(valuesets-1)<<1;
//
//            int *storeAt;
//            storeAt = d_reducedCounts+valuesets*(k+2);// + (k+1)*blockIdx.x/reductionDist;
//
//            if(threadIdx.x == 0)
//            {
//                *storeAt = blockIdx.x;
//                *(storeAt+1)=blockIdx.x;
//                *(storeAt+k+2)=blockIdx.x;
//                *(storeAt+k+2+1)=blockIdx.x;
//            }
//
//            *(storeAt + threadIdx.x+2) = threadIdx.x;
//            *(storeAt + k+2+threadIdx.x+2) = *count;
//
//            d_sumx[threadIdx.x]=*sumx;
//            d_sumy[threadIdx.x]=*sumy;
//            d_count[threadIdx.x]=*count;
//
//        }
//
//        __threadfence();                                                                                      //Ensure sum values are seen by other blocks b4 Arrayin is seen by them.
//        __syncthreads();
//
//        // Declare reduction done by thread with greatest index so that if no. of centroids is less than threads,
//        // then by the time last thread sets the global value other threads from of warps would load the new
//        // centroid values into the shared memory.
//        if(threadIdx.x == BLOCKDIM -1)
//        {
//            d_syncArr[0]=numIter;                                                                                   // Declares global reduction over when it has stored new centroids in global mem.
////              d_dbgIter[numIter<<1]=d_dbgIter[(numIter<<1)+1]=numIter;
//        }
//        //No need for syncthreads() as other threads can go on and load the new centroid values into shared mem. from device mem.
//    }
//    else
//    {
////            if(blockIdx.x==2)
////             {
////               d_syncArr[2]=7; 20
////               __threadfence();
////             }
//        if(threadIdx.x==0)
//        {
//            while(d_syncArr[0]!=numIter);                                                                           // All blocks wait for blockIdx 0, to store new centroids.
////              d_dbgIter[blockIdx.x*((*max_iter)<<1) + (numIter<<1)+1]=numIter;
//        }
//        __syncthreads();                                                                                          // One threads checks value from global and others just wait for it.
//    }
//    //Central Barrier synchronization between all blocks is not required as all threads simply wait for thread 0 to finish.
//    //gpu_sync(numIter,index,1,d_syncArrIn, d_syncArrOut);
//
//}
//
//
//
//// Right now this code only works when
//// 1. Number of threads >= k
//// 2. Number of points can be > threads*blocks, i.e each thread may have to process more than one point till the shared mem allows.
//// 3. All the centroids can be accomodated in shared mem. at once.
//__global__ void __launch_bounds__(BLOCKDIM, MIN_NUM_BLOCKS_PER_MULTIPROC)
//cluster0(int n, int k,int *const __restrict__ max_iter, const float *const __restrict__ d_dataptx, const float *const __restrict__ d_datapty,
//         float *const __restrict__ d_centroidx, float *const __restrict__ d_centroidy, int *const __restrict__ d_syncArr,
//         float *const __restrict__ d_sumx, float *const __restrict__ d_sumy, int *const __restrict__ d_count,int *const __restrict__ d_clusterno, int *const d_reducedCounts, clock_t *const d_timeVar)//, int *const d_dbgIter)
//{
//    // shared memory
//    // Stores x-coord of all clusters followed by y-coord of all clusters i.e. x and y coord of a cluster are separated by k.
//    extern __shared__ float s_centroid[];
//
//    __shared__ float s_thrd[(BLOCKDIM<<1)*POINTS_PER_THREAD];                                               // Store the x-coord of data point processed, for all thrds followed by y-coord.
//    // No point of going for short int as each bank in shared mem. is 32-bit wide and only one thread
//    // can access one bank at a time, so no. of accesses wont be reduced
//    __shared__ signed int s_thrd_k[(BLOCKDIM)*POINTS_PER_THREAD];                                           // Store the nearest cluster of point checked by this thrd.
//
//    float *const s_centroidx = s_centroid;                                                                  // x-coord of 1st cluster's centroid
//    float *const s_centroidy = s_centroid+k;                                                                // y-coord of 1st cluster's centroid
//
//    float *const s_thrd_x = s_thrd;                                                                         // x-coord of points checked by thrds
//    float *const s_thrd_y = s_thrd+(POINTS_PER_THREAD<<LOG_BLOCKDIM);                                       // y-coord of points checked by thrds
////
////
//#if POINTS_PER_THREAD == 1
//    float ptx,pty;
//#else
//    float ptx[POINTS_PER_THREAD],pty[POINTS_PER_THREAD];
//#endif
////
//    // If we've T threads per block then index of ith thread in kth block is k*T + i. So we are maintaining an order among all threads of all blocks.
//    // This is same as blockIdx.x * BLOCKDIM + threadId.x
//    const unsigned int index = (POINTS_PER_THREAD << LOG_BLOCKDIM)*blockIdx.x + threadIdx.x;
//
//#if POINTS_PER_THREAD == 1
//    load_data_points(n,&ptx, &pty, index, d_dataptx, d_datapty, s_thrd_x, s_thrd_y);   //Number of points to be processed by this thread.
//#else
//    const int my_point_num = load_data_points(n,ptx, pty, index, d_dataptx, d_datapty, s_thrd_x, s_thrd_y);     //Number of points to be processed by this thread.
//#endif
//
//
//    bool repeat=true;
//    int numIter;
//    for ( numIter = 0; repeat && (numIter < *max_iter); ++numIter )
//    {
//        load_centroids_in_shared(k, s_centroidx, s_centroidy, d_centroidx, d_centroidy);
//        __syncthreads();
//
//
//        // Find closest centroid if this thread holds a data-point and set it in shared mem.
//#if POINTS_PER_THREAD == 1
//        store_nearest_in_shared(n, k, ptx, pty, index, s_centroidx, s_centroidy, s_thrd_k);
//#else
//        store_nearest_in_shared(k, ptx, pty, s_centroidx, s_centroidy, s_thrd_k, my_point_num);
//#endif
//        __syncthreads();                                                                                          // Wait for all thrds to finish setting closest k value
//
//
//        // Do parallel reduction with each thread doing reduction for a distinct centroid.
//        const unsigned int storageIndex = (blockIdx.x>>1)*k + threadIdx.x;                                        // The sumx, sumy and count value will be stored at this index in global mem.
////
//        float sumx=0.0f;
//        float sumy =0.0f;
//        int count=0;
//
////        if(index<n)
////        {
////          sumx=d_dataptx[index];
////          sumy =d_datapty[index];
////          count=1;
////        }
//        count += reduceThreads(k, &sumx, &sumy, s_thrd_x, s_thrd_y, s_thrd_k);
////        sumx=threadIdx.x+1.0f;
////        sumy=threadIdx.x+1.0f;
////        count=1;
//        __syncthreads();
//        reduceBlocks_and_storeNewCentroids0(k,1,gridDim.x,storageIndex,&sumx,&sumy,&count,d_sumx,d_sumy,d_count,d_syncArr, d_centroidx, d_centroidy, d_reducedCounts, d_timeVar);//,d_dbgIter,*max_iter);
//    }
////
////    // Final centroids have been found. Now set the cluster no. in d_clusterno array in device memory.
////    {
//////        load_centroids_in_shared(k, s_centroidx, s_centroidy, d_centroidx, d_centroidy);
////         __syncthreads();
////
////        // Find closest centroid if this thread holds a data-point
////#if POINTS_PER_THREAD == 1
////        store_nearest_in_global(n, k, &ptx, &pty, index, s_centroidx, s_centroidy, d_clusterno);
////#else
////        store_nearest_in_global(k, ptx, pty, index, s_centroidx, s_centroidy, d_clusterno, my_point_num);
////#endif
////
////    }
//
//    if(blockIdx.x == 0 && threadIdx.x == 0)
//        *max_iter = 1;
//}
//
