bin/gbfs --struconly=0 --opreuse=false --endbit=0 --mxvmode=1 --reduce=1 --timing=1 /data-2/topc-datasets/kron_g500-logn21.mtx

bin/gbfs --struconly=1 --opreuse=false --endbit=0 --mxvmode=1 --reduce=1 --timing=1 /data-2/topc-datasets/kron_g500-logn21.mtx

bin/gbfs --struconly=1 --opreuse=false --endbit=1 --mxvmode=1 --reduce=1 --timing=1 /data-2/topc-datasets/kron_g500-logn21.mtx

bin/gbfs --struconly=1 --opreuse=false --endbit=1 --earlyexit=false --reduce=1 --mxvmode=1 --timing=1 /data-2/topc-datasets/kron_g500-logn21.mtx

bin/gbfs --struconly=1 --opreuse=false --endbit=1 --earlyexit=true --reduce=1 --mxvmode=0 --timing=1 /data-2/topc-datasets/kron_g500-logn21.mtx

bin/gbfs --struconly=1 --opreuse=true --endbit=1 --earlyexit=true --reduce=1 --mxvmode=0 --timing=1 /data-2/topc-datasets/kron_g500-logn21.mtx
#bin/gbfs --struconly=1 --split=true --opreuse=true --earlyexit=true --mxvmode=0 --timing=1 /data-2/gunrock_dataset/large/kron_g500-logn21/kron_g500-logn21.mtx
