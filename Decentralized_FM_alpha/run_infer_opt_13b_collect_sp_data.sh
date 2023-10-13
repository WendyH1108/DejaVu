
file=../c4_train.jsonl
    
echo "start running ${file}"

ARGS="--model-name ../models/facebook_opt_13b \
--model-type opt-save \
--seed 42 \
--fp16 \
--num-layers 40 \
--max-layers 96 \
--budget 22800 \
--num-iters 2000 \
--dist-url tcp://127.0.0.1:9032 \
--token-micro-batch-size 1 \
--world-size 2 --pipeline-group-size 2 --data-group-size 1 \
--pp-mode pipe_sync_sample_mask_token_pipe \
--infer-data ${file}"

(trap 'kill 0' SIGINT; \
python dist_inference_runner.py $(echo ${ARGS}) --cuda-id 0 --rank 0 \
    &
python dist_inference_runner.py $(echo ${ARGS}) --cuda-id 1 --rank 1 \
    & \
wait)

