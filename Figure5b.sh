echo filename, num_rows, num_cols, num_nonzeros, row_length_mean, row_length_std_dev, row_length_variation, row_length_skewness, small, big, method1, avg_ms, gflops, gbps, method2, avg_ms, gflops, gbps, method3, avg_ms, gflops, gbps,
for file in /data/ctcyang/GraphBLAS/dataset/europar/lowd/*/
do
  folder=$(basename $file)
	bin/gbspmm --tb=32 --nt=128 /data/ctcyang/GraphBLAS/dataset/europar/lowd/$folder/$folder.mtx
done
