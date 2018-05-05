# Baseline benchmark
echo Baseline
bin/gbfs --struconly=0 --opreuse=false --mask=0 --earlyexit=false --mxvmode=1 --timing=1 /data-2/topc-datasets/kron_g500-logn21.mtx

# Test for structure-only
echo Struc-Only
bin/gbfs --struconly=1 --opreuse=false --mask=0 --earlyexit=false --mxvmode=1 --timing=1 /data-2/topc-datasets/kron_g500-logn21.mtx

# Test for direction-optimized
echo Direction-Opt
bin/gbfs --struconly=1 --opreuse=false --mask=0 --earlyexit=false --mxvmode=1 --timing=2 /data-2/topc-datasets/kron_g500-logn21.mtx
bin/gbfs --struconly=1 --opreuse=false --mask=0 --earlyexit=false --mxvmode=2 --timing=2 /data-2/topc-datasets/kron_g500-logn21.mtx

# Test for masking
echo Masking
bin/gbfs --struconly=1 --opreuse=false --mask=1 --earlyexit=false --mxvmode=1 --timing=2 /data-2/topc-datasets/kron_g500-logn21.mtx
bin/gbfs --struconly=1 --opreuse=false --mask=1 --earlyexit=false --mxvmode=2 --timing=2 /data-2/topc-datasets/kron_g500-logn21.mtx

# Test for early-exit
echo Early-Exit
bin/gbfs --struconly=1 --opreuse=false --mask=1 --earlyexit=true --mxvmode=1 --timing=2 /data-2/topc-datasets/kron_g500-logn21.mtx
bin/gbfs --struconly=1 --opreuse=false --mask=1 --earlyexit=true --mxvmode=2 --timing=2 /data-2/topc-datasets/kron_g500-logn21.mtx
bin/gbfs --struconly=1 --opreuse=false --mask=1 --earlyexit=true --mxvmode=0 --timing=1 /data-2/topc-datasets/kron_g500-logn21.mtx

# Test for operand-reuse
echo Op-Reuse
bin/gbfs --struconly=1 --opreuse=true --mask=1 --earlyexit=true --mxvmode=0 --timing=1 /data-2/topc-datasets/kron_g500-logn21.mtx
