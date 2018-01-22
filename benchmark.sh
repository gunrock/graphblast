for j in 0 1 2
do
  for i in gpushbench gpullbench gspmspvbench gspmvbench
  #for i in soc-orkut soc-LiveJournal1 hollywood-2009 indochina-2004 kron_g500-logn21 rmat_n22_e64 rmat_n23_e32 rmat_n24_e16 rgg_n24_0.000548 road_usa roadNet-CA
  do
    bin/$i --struconly=0 /data-2/topc-datasets/kron_g500-logn21.mtx > result/$i
  done
done
