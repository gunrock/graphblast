for file in /data/ctcyang/GraphBLAS/data/gc-data/*/
do
  folder=$(basename $file)
    bin/ggc --niter 10 --mxvmode 0 --timing 1 --directed 2 /data/ctcyang/GraphBLAS/data/gc-data/$folder/$folder.mtx
done
