echo filename, num_rows, num_cols, num_nonzeros, row_length_mean, row_length_std_dev, row_length_variation, row_length_skewness, small, big, method1, avg_ms, gflops, gbps, method2, avg_ms, gflops, gbps
for row in 2 8 32 128 512 2048 8192 32768 131072 524288 2097152 8388608
do
  #nvprof --profile-from-start off --metrics achieved_occupancy,eligible_warps_per_cycle bin/gbspmm --iter=1 /data-2/gunrock_dataset/large/benchmark5/$folder/$folder.mtx
  #nvprof --profile-from-start off --metrics avg_threads_executed_per_instruction,not_predicated_off_thread_inst_executed,thread_inst_executed bin/gbcusparse2 --iter=1
  #nvprof --profile-from-start off --metrics gld_transactions_per_request,gst_transactions_per_request bin/gbdensecusparse2 --dense=$row --iter=1
  nvprof --profile-from-start off --metrics achieved_occupancy,warp_nonpred_execution_efficiency bin/gbdensecusparse2 --dense=$row --iter=1
  #nvprof --profile-from-start off --metrics ipc,issued_ipc,executed_ipc,warp_issue_efficiency bin/gbdensecusparse2 --dense=$row --iter=1
  #nvprof --profile-from-start off --metrics warp_execution_efficiency,warp_nonpred_execution_efficiency bin/gbdensecusparse2 --dense=$row --iter=1
  #nvprof --profile-from-start off --metrics achieved_occupancy,eligible_warps_per_cycle bin/gbdensecusparse2 --dense=$row --iter=1
  #nvprof --profile-from-start off --metrics dram_read_throughput,dram_write_throughput,dram_read_transactions,dram_write_transactions,dram_utilization bin/gbdensecusparse2 --dense=$row --iter=1
done

