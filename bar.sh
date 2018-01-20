bin/gbfs --struconly=0 --opreuse=false --mxvmode=1 --timing=1 /data-2/gunrock_dataset/large/kron_g500-logn21/kron_g500-logn21.mtx

bin/gbfs --struconly=1 --opreuse=false --mxvmode=1 --timing=1 /data-2/gunrock_dataset/large/kron_g500-logn21/kron_g500-logn21.mtx

bin/gbfs --struconly=1 --opreuse=true --mxvmode=1 --timing=1 /data-2/gunrock_dataset/large/kron_g500-logn21/kron_g500-logn21.mtx

bin/gbfs --struconly=1 --opreuse=true --earlyexit=false --mxvmode=0 --timing=1 /data-2/gunrock_dataset/large/kron_g500-logn21/kron_g500-logn21.mtx

bin/gbfs --struconly=1 --opreuse=true --earlyexit=true --mxvmode=0 --timing=1 /data-2/gunrock_dataset/large/kron_g500-logn21/kron_g500-logn21.mtx
#bin/gbfs --struconly=1 --split=true --opreuse=true --earlyexit=true --mxvmode=0 --timing=1 /data-2/gunrock_dataset/large/kron_g500-logn21/kron_g500-logn21.mtx
