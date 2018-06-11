mkdir -p /data/ctcyang/GraphBLAS/dataset/europar/matrix
for size in 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000 16000 17000 18000 19000 20000
do
  mkdir -p /data/ctcyang/GraphBLAS/dataset/europar/matrix/$size
  ./gen_matrix 100000 100000 $size > /data/ctcyang/GraphBLAS/dataset/europar/matrix/$size/$size.mtx
done
