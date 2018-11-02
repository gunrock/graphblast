Environment Variables
=====================
GraphBLAS has several settings that you can change with environment variables.

For example, you can set these environment ariables in Linux or macOS by:
```
export GRB_SPARSE_MATRIX_FORMAT=1
```

## Sparse matrix storage format

* GRB_SPARSE_MATRIX_FORMAT
  - Values: 0 (CSRCSC), 1 (CSRONLY) ```(default=1)```
  - The storage format that is used to store sparse matrix. CSRCSC will double t
he storage requirements, but may allow more optimizations such as direction-opti
mization when doing mxv and vxm. This overrides mxvmode settings in Descriptor, because missing CSC data structure makes column-based mxv and row-based vxm impossible without materializing this data structure, which is both memory-intensive and inefficient.

## Matrix-vector load-balance algorithm

* GRB_LOAD_BALANCE_MODE
  - Values: 0 (SIMPLE), 1 (LB) ```(default=1)```
  - The matrix-vector multiplication algorithm that is used. SIMPLE does not do any load-balancing, so it is good for road network graphs. LB does merge-path load-balancing as well as direction-optimization, so it is good for power law graphs.

## Utility functions

* GRB_UTIL_IGNORE_SELFLOOP
  - Values: 0, 1 ```(default=1)```
  - Since we are dealing with graphs, the MTX loader will eliminate any edges going from vertex *i* to vertex *i* (self-loops), unless the user sets this option to 0.
