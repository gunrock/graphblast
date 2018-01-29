ARCH="GEN_SM45"

for i in kron_g500-logn16 kron_g500-logn17 kron_g500-logn18 kron_g500-logn19 kron_g500-logn20 kron_g500-logn21
do
	if [ "$ARCH" = "GEN_SM25" ] ; then
        benchmark/test /data/gunrock_dataset/large/$i/$i.mtx -undirected
    fi
done

for i in ak2010 belgium_osm coAuthorsDBLP delaunay_n13 delaunay_n21 webbase-1M soc-LiveJournal1 kron_g500-logn21
do
    if [ "$ARCH" = "GEN_SM35" ] ; then
        bin/gbfs --struconly=true --mxvmode=2 --timing=1 --transpose=1 --directed=1 /data-2/gunrock_dataset/large/$i/$i.mtx
    fi
done

for j in 1 #0 1 2
do
  #for i in soc-orkut soc-LiveJournal1 hollywood-2009 indochina-2004 kron_g500-logn21 roadNet-CA
  for i in soc-orkut soc-LiveJournal1 hollywood-2009 indochina-2004 kron_g500-logn21 rmat_n22_e64 rmat_n23_e32 rmat_n24_e16 rgg_n24_0.000548 roadNet-CA road_usa
  do
    if [ "$ARCH" = "GEN_SM45" ] ; then
      bin/gbfs --sort=0 --struconly=true --mxvmode=$j --timing=1 /data-2/topc-datasets/$i.mtx
    fi
  done
done

# IPDPS 2016
#for i in soc-orkut hollywood-2009 indochina-2004 kron_g500-logn21 rgg_n24_0.000548 roadNet-CA
for i in soc-orkut soc-LiveJournal1 hollywood-2009 indochina-2004 kron_g500-logn21 rmat_n22_e64 rmat_n23_e32 rmat_n24_e16 rgg_n24_0.000548 roadNet-CA road_usa
do
  for j in 0 1 2
  do
    if [ "$ARCH" = "GEN_SM50" ] ; then
      bin/gbfs --struconly=true --mxvmode=$j --niter=1 --timing=2 /data-2/topc-datasets/$i.mtx
    fi
  done
done
