OPTION="2"

if [ "$OPTION" = "2" ] ; then
  cat result/spmm_schema
fi

for file in /data-2/gunrock_dataset/large/benchmark/*/
do
  if [ "$OPTION" = "2" ] ; then
    folder=$(basename $file)
    bin/gbspmm --tb=64 --nt=64 /data-2/gunrock_dataset/large/benchmark/$folder/$folder.mtx
  fi
done

for i in ak2010 belgium_osm coAuthorsDBLP delaunay_n13 delaunay_n21 webbase-1M soc-LiveJournal1 kron_g500-logn21
do
  if [ "$OPTION" = "1" ] ; then
	  ./gpu_spmv --mtx=/data-2/gunrock_dataset/large/$i/$i.mtx --fp32
  fi
done

