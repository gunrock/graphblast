OPTION="5"

if [ "$OPTION" = "2" ] || ["$OPTION" = "3" ] || ["$OPTION" = "4" ] || ["$OPTION" = "5" ] || ["$OPTION" = "6"] ; then
  cat result/spmm_schema
fi

for file in audikw_1 bmwcra_1 bone010 crankseg_2 F1 ldoor #inline_1
do
	for i in CSR CUSPARSECSR CUSPARSECSR2 #SELLP
	do
	  if [ "$OPTION" = "7" ] ; then
			echo "Running $i"
		  ../magma-2.3.0/build/sparse/testing/testing_ssolver --solver LOBPCG --ev 32 --maxiter 10 --format $i /data-2/ctcyang/euro-dataset/$file.mtx
	  fi
	done
done

for file in /data-2/gunrock_dataset/large/benchmark4/*/
do
  if [ "$OPTION" = "6" ] ; then
    folder=$(basename $file)
    bin/gbspmm --tb=32 --nt=128 /data-2/gunrock_dataset/large/benchmark4/$folder/$folder.mtx
  fi
done

for file in /data-2/gunrock_dataset/large/benchmark3/*/
do
  if [ "$OPTION" = "5" ] ; then
    folder=$(basename $file)
    bin/gbspmm --tb=32 --nt=128 /data-2/gunrock_dataset/large/benchmark3/$folder/$folder.mtx
  fi
done

for file in /data-2/gunrock_dataset/large/benchmark2/*/
do
  if [ "$OPTION" = "5" ] ; then
    folder=$(basename $file)
    bin/gbspmm --tb=4 --nt=256 /data-2/gunrock_dataset/large/benchmark2/$folder/$folder.mtx
  fi
done

for file in /data-2/gunrock_dataset/large/*/
do
  if [ "$OPTION" = "3" ] ; then
    folder=$(basename $file)
    bin/gbspmm --tb=64 --nt=64 /data-2/gunrock_dataset/large/$folder/$folder.mtx
  fi
done

for file in /data-2/gunrock_dataset/large/benchmark/*/
do
  if [ "$OPTION" = "2" ] ; then
    folder=$(basename $file)
    bin/gbspmm --tb=4 --nt=256 /data-2/gunrock_dataset/large/benchmark/$folder/$folder.mtx
  fi
done

for i in ak2010 belgium_osm coAuthorsDBLP delaunay_n13 delaunay_n21 webbase-1M soc-LiveJournal1 kron_g500-logn21
do
  if [ "$OPTION" = "1" ] ; then
	  ./gpu_spmv --mtx=/data-2/gunrock_dataset/large/$i/$i.mtx --fp32
  fi
done

