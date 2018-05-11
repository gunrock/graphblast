echo filename, num_rows, num_cols, num_nonzeros, row_length_mean, row_length_std_dev, row_length_variation, row_length_skewness, small, big, method1, avg_ms, gflops, gbps, method2, avg_ms, gflops, gbps
for row in 2 8 32 128 512 2048 8192 32768 131072 524288 2097152 8388608
do
	  bin/gbdense --dense=$row
done
