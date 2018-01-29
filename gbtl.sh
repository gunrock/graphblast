for j in 0 1 2
do
  for i in Journals G43 ship_003 belgium_osm roadNet-CA delaunay_n24
  do
    bin/gbfs --sort=0 --struconly=true --mxvmode=$j --timing=1 /data-2/gunrock_dataset/large/benchmark6/$i/$i.mtx
  done
done

for i in Journals G43 ship_003 belgium_osm roadNet-CA delaunay_n24
do
  bin/gbfs --sort=0 --struconly=true --mxvmode=1 --timing=1 /data-2/gunrock_data
set/large/benchmark6/$i/$i.mtx
done
