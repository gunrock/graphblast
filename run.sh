

ARCH="GEN_SM35"

echo "data, milliseconds, gflops"
for i in ak2010 #mc2depi pwtk pdb1HYS consph hood webbase-1M #belgium_osm coAuthorsDBLP delaunay_n13 delaunay_n21 webbase-1M soc-LiveJournal1 kron_g500-logn21
do
  if [ "$ARCH" = "GEN_SM15" ] ; then
    #for tb in 1 2 4 8 16 32
	  #do
			for nt in 32 64 128 256 512 1024
			do
        #benchmark_spgemm/test --iter=10 --device=0 --major=cusparse /data/gunrock_dataset/large/$i/$i.mtx
        #benchmark_spgemm/test --iter=10 --split=true /data-2/gunrock_dataset/large/$i/$i.mtx
        benchmark_spmm/test --nt=$nt /data-2/gunrock_dataset/large/$i/$i.mtx
		  done
	  #done	
  fi
done

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
        benchmark_spmm/test --mode=cusparse --nt=64 /data/gunrock_dataset/large/$i/$i.mtx
    else
        if [ "$ARCH" = "GEN_SM40" ] ; then
            #./test /data/gunrock_dataset/large/soc-LiveJournal1/soc-LiveJournal1.mtx
            ./test /data/gunrock_dataset/large/kron_g500-logn21/kron_g500-logn21.mtx
            ./test /data/gunrock_dataset/large/kron_g500-logn21/kron_g500-logn21.mtx -undirected
            break
        fi
    fi
done

for i in 2-bitcoin 6-roadnet 4-pld 
do
    if [ "$ARCH" = "GEN_SM45" ] ; then
        ./test /data/PPOPP15/$i.mtx
    fi
done
