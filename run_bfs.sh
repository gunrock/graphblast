TEST="1"

for file in ak2010 belgium_osm cit-Patents coAuthorsDBLP delaunay_n13 delaunay_n21 delaunay_n24 europe_osm hollywood-2009 indochina-2004 kron_g500-logn21 roadNet-CA road_usa soc-LiveJournal1 soc-orkut
do
  if [ "$TEST" = "0" ] ; then
    bin/gbfs --timing 0 --earlyexit 1 --mxvmode 1 --struconly 1 --niter 0 --opreuse 1 /data/ctcyang/GraphBLAS/dataset/large/$file/$file.mtx
  fi
done

for file in soc-orkut soc-LiveJournal1 hollywood-2009 indochina-2004 kron_g500-logn21 rmat_n22_e64 rmat_n23_e32 rmat_n24_e16 rgg_n24_0.000548 roadNet-CA road_usa
do
  if [ "$TEST" = "1" ] ; then
    bin/gbfs --timing 0 --earlyexit 1 --mxvmode 1 --struconly 1 --niter 0 --opreuse 1 /data/ctcyang/GraphBLAS/dataset/large/$file/$file.mtx
  fi
done
