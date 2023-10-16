file=../winogrande.jsonl
output_file=../13b_winogrande_0bit_output.jsonl


echo "start running ${file}"

ARGS="--model-name ../models/facebook_opt_13b \
--model-type opt \
--seed 42 \
--fp16 \
--num-layers 40 \
--max-layers 96 \
--budget 22800 \
--num-iters 1000000 \
--dist-url tcp://127.0.0.1:9031 \
--token-micro-batch-size 1 \
--world-size 1 --pipeline-group-size 1 --data-group-size 1 \
--pp-mode pipe_sync_sample_mask_token_pipe \
--infer-data ${file} \
--output-path ${output_file}"

(trap 'kill 0' SIGINT; \
python dist_inference_runner.py $(echo ${ARGS}) --cuda-id 6 --rank 0 \
    & \
wait)