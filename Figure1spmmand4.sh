echo filename, num_rows, num_cols, num_nonzeros, row_length_mean, row_length_std_dev, row_length_variation, row_length_skewness, small, big, method1, avg_ms, gflops, gbps, method2, avg_ms, gflops, gbps
for row in 8388608 2097152 524288 131072 32768 8192 2048 512 128 32 8 2
do
	  bin/gbdense --dense=$row
done
