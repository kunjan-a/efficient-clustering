To find the occupancy we use occupancy.sh and pass to it registerPerThread usage (count) and sharedPerblock usage (memory in bytes) of the kernel.

e.g.
./occupancy_c1060.sh 29 136| grep "\-\-" | sed -e s/\ warps//g | sed -e s/\ blocks//g | sed -e s/\-\-//g

The above cmd will give the expected block assignment and warp assignment on a SM for C1060 with threadsPerBlock varying from 64 to 512 and number of blocks varying from 30 to 240


./occupancy_c1060.sh 29 136| grep "\*" |cut -d\  -f2,4,6,8,10,12,14

The above cmd will give the following values for C1060 with threadsPerBlock varying from 64 to 512 and number of blocks varying from 30 to 240:
threadsPerBlock, totalBlocks, blocksPerSM, warpsPerBlock, totalWarpsForOneSM, regPerBlock, sharedPerBlock
