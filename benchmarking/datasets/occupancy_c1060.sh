#!/bin/bash

echo "Args should be regPerThread sharedPerBlock used by the kernel."

#thrdPerBlock=$1
#blocks=$2
regPerThrd=$1
sharedPerBlock=$2

sms=30
maxThrdPerBlock=512
maxRegPerSm=16384
maxSharedPerSm=16384
maxThrdPerSm=1024
maxBlocksPerSm=8
gw=2				#warp allocation granularity
gt=512				#thread allocation granularity
gs=512				#shared memory allocation granularity
echo "Actual shared usage per block is :ceil("$sharedPerBlock","$gs"), where ceil(x,y) is x rounded to nearest multiple of y:"
if (( sharedPerBlock%gs > 0 ))
then
    sharedPerBlock=$(( sharedPerBlock + gs-(sharedPerBlock%gs) ))
fi
echo $sharedPerBlock" bytes."


for ((thrdPerBlock=64; thrdPerBlock<=maxThrdPerBlock; thrdPerBlock*=2))
do
  for ((blocks_iter=1; blocks_iter<=maxBlocksPerSm; blocks_iter+=1))
  do
    blocksPerSm=0
    blocks=$((blocks_iter*sms));
    output=""
    output1=""
    warpsPerBlock=$((thrdPerBlock/32))
    if (( warpsPerBlock%gw > 0 ))
    then
	warpsPerBlock=$(( warpsPerBlock + gw - (warpsPerBlock%gw) ))
    fi
    regPerBlock=$(( warpsPerBlock*32*regPerThrd ))
    if ((regPerBlock%gt > 0))
    then
	regPerBlock=$(( regPerBlock + gt - (regPerBlock%gt) ))
    fi
    echo "********* "$thrdPerBlock" threadsPerBlock, "$blocks" totalBlocks, "$blocks_iter" blocksPerSM, "$warpsPerBlock" warpsPerBlock, "$((warpsPerBlock*blocks_iter))" totalWarpsForOneSM, "$regPerBlock" regPerBlock, "$sharedPerBlock" sharedPerBlock **********"
    if ((regPerBlock>maxRegPerSm)) || ((sharedPerBlock>maxSharedPerSm))
    then
      output="Not possible"
      output=$output" Per block register usage is:"$((thrdPerBlock*regPerThrd))" while max. allowed is:"$maxregPerSm
      output=$output" Per block shared usage is:"$((thrdPerBlock*regPerThrd))" while max. allowed is:"$maxSharedPerSm
      blocks=0
    fi

    for ((i=1; blocks>0; i+=1))
    do
      if ((i*regPerBlock<=maxRegPerSm)) && ((i*sharedPerBlock<=maxSharedPerSm)) && ((i*thrdPerBlock<=maxThrdPerSm)) && ((i<=maxBlocksPerSm))
      then
	blocksPerSm=$i
	blocks=$((blocks-sms))
#	echo $blocks" blocks left"
      else
	i=$blocksPerSm
	echo $i" blocks, "$((i*thrdPerBlock/32))" warps, "$((i*thrdPerBlock))" thrdPerSm, "$((i*regPerBlock))" regPerSm, "$((i*sharedPerBlock))" sharedPerSm."
	output=$output$((i*thrdPerBlock/32))" warps,"
	output1=$output1$i" blocks,"
        i=0
#	sleep 1s
      fi
    done

    if ((blocksPerSm>0))
    then
	i=$blocksPerSm
	echo $i" blocks, "$((i*thrdPerBlock/32))" warps, "$((i*thrdPerBlock))" thrdPerSm, "$((i*regPerBlock))" regPerSm, "$((i*sharedPerBlock))" sharedPerSm."
	output=$output$((i*thrdPerBlock/32))" warps"
        output1=$output1$i" blocks"
    fi
    echo "-- "$output1" "$output" --"
#    sleep 1s
  done
done
