for i in 1 2
do
  bin/grandbfs --struconly=1 --niter=1000 --mxvmode=$i --timing=2 /data-2/topc-datasets/kron_g500-logn21.mtx > result/grandbfs-$i
done
