#define DONT_USE_LOAD_OPTIMIZED 1
__device__ INLINE int get_my_point_num(const int n, int *const RESTRICT startIndex)
{
  int totalThreads = BLOCKDIM * gridDim.x;
  int my_point_num = n/totalThreads;
  //points that wud still be left is: n - my_point_num*totalThreads
  int extra_points = n - my_point_num*totalThreads;

  int foreign_thrds_b4_me = BLOCKDIM*blockIdx.x;
  *startIndex= my_point_num * foreign_thrds_b4_me  + threadIdx.x;
  if(extra_points <= foreign_thrds_b4_me)
    *startIndex+=extra_points;
  else
  {
    *startIndex+=foreign_thrds_b4_me;    
    if(extra_points > (blockIdx.x<<LOG_BLOCKDIM) + threadIdx.x)
      my_point_num++;
    
  }
  return my_point_num;
}

__global cluster(const int n,....)
{
  
  int startIndex;
  const int my_point_num = get_my_point_num(n,&startIndex);

  if(my_point_num > POINTS_PER_THREAD)  //Load in cycles
  {
    int point_cycles=my_point_num/POINTS_PER_THREAD; //definitely > 1
    int curr_cycle=0;

  #if DONT_USE_LOAD_OPTIMIZED == 1
    for(int curr_iter=0; curr_iter<*max_iter; curr_iter++)//For each iteration
    {
      for(curr_cycle=0;curr_cycle<point_cycles; curr_cycle++)
      {
        point_num=POINTS_PER_THREAD;
        //////////////////////////////////////
        // load_data_points(point_num,startIndex+curr_cycle*POINTS_PER_THREAD,...);
        ///// Do all the stuff for these /////
        //////////////////////////////////////
      }

      point_num=my_point_num - point_cycles*POINTS_PER_THREAD;
      //This value of point_num is definitely less than POINTS_PER_THREAD
      {
        //////////////////////////////////////
        // load_data_points(point_num,startIndex+point_cycles*POINTS_PER_THREAD,...);
        ///// Do all the stuff for these /////
        //////////////////////////////////////
      }
    }  //End of iteration
  #else

    ///////////////////////////////////////////
    //////////// LOAD OPTIMIZED ///////////////
    ///////////////////////////////////////////

    int point_num=POINTS_PER_THREAD;
    //////////////////////////////////////
    //load_data_points(point_num,startIndex,...);
    ///// Do all the stuff for these /////
    //////////////////////////////////////

    //This is for the case my_point_num was a multiple of POINTS_PER_THREAD
    int start_odd_iter = point_cycles-2; 
    if(point_cycles*POINTS_PER_THREAD < my_point_num)
      start_odd_iter = point_cycles-1;
    
    for(int curr_iter=0; curr_iter<*max_iter; curr_iter+=2)//For each iteration
    {
      //For each even iteration i.e 0th, 2nd, 4th ...
      //This is perfectly safe as we know point_cycles > 1
      for(curr_cycle=1;curr_cycle<point_cycles; curr_cycle++)
      {
        point_num=POINTS_PER_THREAD;
        //////////////////////////////////////
        //load_data_points(point_num,startIndex+curr_cycle*POINTS_PER_THREAD,...);
        ///// Do all the stuff for these /////
        //////////////////////////////////////
      }

      point_num=my_point_num - point_cycles*POINTS_PER_THREAD;
      //This value of point_num is definitely less than POINTS_PER_THREAD
      {
        //////////////////////////////////////
        //load_data_points(point_num,startIndex+point_cycles*POINTS_PER_THREAD,...);
        ///// Do all the stuff for these /////
        //////////////////////////////////////
      }
      //End of iteration

      if(++curr_iter < *max_iter)
      {   //For each odd iteration i.e 1st, 3rd, 5th ...
      
        for(curr_cycle=start_odd_iter;curr_cycle>=0; curr_cycle--)
        {
          point_num=POINTS_PER_THREAD;
          //////////////////////////////////////
          //load_data_points(point_num,startIndex+curr_cycle*POINTS_PER_THREAD,...);
          ///// Do all the stuff for these /////
          //////////////////////////////////////
        }
      
      }   //End of iteration
    }
  #endif
  }
  else    //my_point_num <= POINTS_PER_THREAD
  {
    //////////////////////////////////////////////////////
    //Only one time loading of data points would suffice//
    //load_data_points(my_point_num,startIndex,...);
    //////////////////////////////////////////////////////
  //For each iteration
  //End of iteration

  }
}

__device__ load_data_points(const int my_point_num, const int startIndex, ...)
{

  switch(my_point_num)
  {
    case 0:break;
    case POINTS_PER_THREAD:
        int index = startIndex;

        #pragma unroll DEF_NUMATTRIB  //DEF_NUMATTRIB
        for(int attr_iter=0;attr_iter<DEF_NUMATTRIB; attr_iter++)
        {
          index = startIndex;
          #pragma unroll POINTS_PER_THREAD  //POINTS_PER_THREAD
          for(int point_iter=0; point_iter<POINTS_PER_THREAD; point_iter++)
          {
            pt[attr_iter][point_iter]=D_DATAPT_AT(attr_iter,index);
            index+=BLOCKDIM;
          }
        }
    default:
        int index = startIndex;

        #pragma unroll DEF_NUMATTRIB  //DEF_NUMATTRIB
        for(int attr_iter=0;attr_iter<DEF_NUMATTRIB; attr_iter++)
        {
          index = startIndex;
          #pragma unroll POINTS_PER_THREAD  //POINTS_PER_THREAD
          for(int point_iter=0; point_iter<my_point_num; point_iter++)
          {
            pt[attr_iter][point_iter]=D_DATAPT_AT(attr_iter,index);
            index+=BLOCKDIM;
          }
        }
  }
}


