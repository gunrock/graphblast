TEST="1"
DATA="/data-2/gunrock_dataset/large"
DATA2="/data-2/topc-datasets"

for file in ak2010 belgium_osm coAuthorsDBLP delaunay_n10 delaunay_n13 delaunay_n21 webbase-1M soc-LiveJournal1 kron_g500-logn21
do
  if [ "$TEST" = "0" ] ; then
    echo bin/gsssp --timing 0 --mxvmode 0 --directed 2 --niter 5 --memusage 0.5 $DATA/$file/$file.mtx
    bin/gsssp --timing 0 --mxvmode 0 --directed 2 --niter 5 --memusage 0.5 $DATA/$file/$file.mtx
  fi
done

for file in soc-orkut soc-LiveJournal1 hollywood-2009 indochina-2004
do
  if [ "$TEST" = "1" ] ; then
    for switch in 0.2 0.15 0.1 0.5 0.01
    do
      echo bin/gsssp --timing 1 --mxvmode 0 --niter 5 --directed 2 --memusage 0.5 --switchpoint $switch $DATA/$file.mtx
      bin/gsssp --timing 1 --mxvmode 0 --niter 5 --directed 2 --memusage 0.5 --switchpoint $switch $DATA2/$file.mtx
    done
  fi
done

for file in rmat_n22_e64 rmat_n23_e32 rmat_n24_e16
do
  if [ "$TEST" = "1" ] ; then 

for file in rgg_n24_0.000548 road_usa
do
  if [ "$TEST" = "2" ] ; then
    echo bin/gsssp --timing 1 --mxvmode 1 --directed 2 --niter 5 $DATA2/$file.mtx
    bin/gsssp --timing 1 --mxvmode 1 --directed 2 --niter 5 $DATA2/$file.mtx
  fi
done

for file in test_bc test_cc test_mesh test_mis test_pr small chesapeake
do
  if [ "$TEST" = "3" ] ; then
    echo data/small/$file.mtx
    bin/gsssp --timing 1 --mxvmode 0 --niter 5 data/small/$file.mtx
  fi
done
