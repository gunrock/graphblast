TEST="3"
DATA="/data/gunrock_dataset/large"
DATA2="/data/topc-datasets"

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
    echo bin/gsssp --timing 0 --mxvmode 0 --niter 5 --directed 2 --memusage 0.5 --switchpoint 0.1 $DATA/$file.mtx
    bin/gsssp --timing 0 --mxvmode 0 --niter 5 --directed 2 --memusage 0.5 --switchpoint 0.1 $DATA2/$file.mtx
  fi
done

if [ "$TEST" = "1" ] ; then
  echo bin/gsssp --timing 0 --mxvmode 0 --niter 5 --directed 2 --memusage 0.5 --switchpoint 0.1 /data/topc-datasets/rmat_n22_e64.mtx
  bin/gsssp --timing 0 --mxvmode 0 --niter 5 --directed 2 --memusage 0.5 --switchpoint 0.1 /data/topc-datasets/rmat_n22_e64.mtx

  echo bin/gsssp --timing 0 --mxvmode 0 --niter 5 --directed 2 --memusage 0.5 --switchpoint 0.05 /data/topc-datasets/rmat_n23_e32.mtx
  bin/gsssp --timing 0 --mxvmode 0 --niter 5 --directed 2 --memusage 0.5 --switchpoint 0.05 /data/topc-datasets/rmat_n23_e32.mtx

  echo bin/gsssp --timing 0 --mxvmode 0 --niter 5 --directed 2 --memusage 0.5 --switchpoint 0.025 /data/topc-datasets/rmat_n24_e16.mtx
  bin/gsssp --timing 0 --mxvmode 0 --niter 5 --directed 2 --memusage 0.3 --switchpoint 0.025 /data/topc-datasets/rmat_n24_e16.mtx
fi

for file in rgg_n24_0.000548 road_usa
do
  if [ "$TEST" = "3" ] ; then
    echo bin/gsssp --timing 0 --mxvmode 1 --directed 2 --niter 5 $DATA2/$file.mtx
    bin/gsssp --timing 0 --mxvmode 1 --directed 2 --niter 5 $DATA2/$file.mtx
  fi
done

for file in test_bc test_cc test_mesh test_mis test_pr small chesapeake
do
  if [ "$TEST" = "4" ] ; then
    echo data/small/$file.mtx
    bin/gsssp --timing 0 --mxvmode 0 --niter 5 data/small/$file.mtx
  fi
done
