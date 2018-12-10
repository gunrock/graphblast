bin/gbfs --timing 2 --max_niter 6 --earlyexit 1 --mxvmode 0 --struconly 1 --niter 0 --opreuse 1 /data/ctcyang/GraphBLAS/data/large/kron_g500-logn21/kron_g500-logn21.mtx

bin/gsssp --timing 2 --max_niter 5 --niter 0 --mxvmode 0 /data/ctcyang/GraphBLAS/data/large/kron_g500-logn21/kron_g500-logn21.mtx

bin/glgc --timing 2 --max_niter 5 --niter 0 --mxvmode 0 /data/ctcyang/GraphBLAS/data/large/kron_g500-logn21/kron_g500-logn21.mtx

bin/gmis --niter 0 --mxvmode 0 --directed 2 data/small/test_bc.mtx

bin/ggc --niter 0 --mxvmode 0 --directed 2 data/small/test_bc.mtx
