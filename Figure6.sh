echo Note: must download matrices first under ./dataset/europar/large/ using scripts ./dataset/europar/large/DownloadFigure6.sh and ./dataset/europar/large/ExtractFigure6.sh

echo filename, num_rows, num_cols, num_nonzeros, row_length_mean, row_length_std_dev, row_length_variation, row_length_skewness, small, big, method1, avg_ms, gflops, gbps, method2, avg_ms, gflops, gbps
for file in /data/ctcyang/GraphBLAS/dataset/europar/large/*/
do
  folder=$(basename $file)
	bin/gbspmm /data/ctcyang/GraphBLAS/dataset/europar/large/$folder/$folder.mtx
done
