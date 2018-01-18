

ARCH="GEN_SM35"

echo "data, milliseconds, gflops"

if [ "$ARCH" = "GEN_SM20" ] ; then
    ./test ../../dataset/small/test_cc.mtx
    ./test ../../dataset/small/test_bc.mtx
    ./test ../../dataset/small/test_pr.mtx
    ./test ../../dataset/small/chesapeake.mtx
fi


for i in kron_g500-logn16 kron_g500-logn17 kron_g500-logn18 kron_g500-logn19 kron_g500-logn20 kron_g500-logn21
do
	if [ "$ARCH" = "GEN_SM25" ] ; then
        benchmark/test /data/gunrock_dataset/large/$i/$i.mtx -undirected
    fi
done

for i in 579593	897318 666033 194754 796384 924094 932129 912391 344516
do
    if [ "$ARCH" = "GEN_SM30" ] ; then
	    ./test /data/gunrock_dataset/large/kron_g500-logn20/kron_g500-logn20.mtx -source $i -undirected
	fi
done

for i in ak2010 belgium_osm coAuthorsDBLP delaunay_n13 delaunay_n21 webbase-1M soc-LiveJournal1 kron_g500-logn21
do
    if [ "$ARCH" = "GEN_SM35" ] ; then
        bin/gbfs --struconly=true --mxvmode=1 --timing=1 --transpose=1 --directed=1 /data-2/gunrock_dataset/large/$i/$i.mtx
    else
        if [ "$ARCH" = "GEN_SM40" ] ; then
            ./test /data/gunrock_dataset/large/kron_g500-logn21/kron_g500-logn21.mtx
        fi
    fi
done

for i in 2-bitcoin 6-roadnet 4-pld 
do
    if [ "$ARCH" = "GEN_SM45" ] ; then
        ./test /data/PPOPP15/$i.mtx
    fi
done
